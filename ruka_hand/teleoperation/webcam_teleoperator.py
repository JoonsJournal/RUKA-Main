#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebCam Teleoperator - MediaPipe 기반 RUKA 로봇 손 원격조정

이 모듈은 웹캠과 MediaPipe를 사용하여 RUKA 로봇 손을 실시간으로 제어합니다.

주요 기능:
1. 웹캠에서 손 추적 (MediaPipe Hands)
2. 21개 랜드마크를 로봇 제어 데이터로 변환
3. Oculus 텔레오퍼레이터와 동일한 좌표계 변환 로직 사용
4. 실시간 로봇 손 제어

작성자: 이동준
수정: 에러 처리 강화 및 좌표 정규화 개선
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

from copy import deepcopy as copy
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation

from gesture_ctrled_ruka.HandTrackingModule import HandDetector
from ruka_hand.control.operator import RUKAOperator
from ruka_hand.utils.constants import *
from ruka_hand.utils.timer import FrequencyTimer
from ruka_hand.utils.vectorops import *

# =============================================================================
# MediaPipe 랜드마크를 Oculus 형식으로 매핑
# =============================================================================
# MediaPipe 21개 랜드마크 → Oculus 25개 키포인트 매핑
# MediaPipe: 0(손목), 1-4(엄지), 5-8(검지), 9-12(중지), 13-16(약지), 17-20(새끼)
# Oculus: 손목 + 각 손가락 5개 관절

MEDIAPIPE_TO_OCULUS_MAPPING = {
	# 손목
	0: 0,    # 손목 (Wrist)

	# 엄지 (Thumb) - MediaPipe 1,2,3,4 → Oculus 1,2,3,4
	1: 1,    # 엄지 CMC
	2: 2,    # 엄지 MCP
	3: 3,    # 엄지 IP
	4: 4,    # 엄지 끝

	# 검지 (Index) - MediaPipe 5,6,7,8 → Oculus 5,6,7,8
	5: 5,    # 검지 MCP
	6: 6,    # 검지 PIP
	7: 7,    # 검지 DIP
	8: 8,    # 검지 끝

	# 중지 (Middle) - MediaPipe 9,10,11,12 → Oculus 9,10,11,12
	9: 9,    # 중지 MCP
	10: 10,  # 중지 PIP
	11: 11,  # 중지 DIP
	12: 12,  # 중지 끝

	# 약지 (Ring) - MediaPipe 13,14,15,16 → Oculus 13,14,15,16
	13: 13,  # 약지 MCP
	14: 14,  # 약지 PIP
	15: 15,  # 약지 DIP
	16: 16,  # 약지 끝

	# 새끼 (Pinky) - MediaPipe 17,18,19,20 → Oculus 17,18,19,20
	17: 17,  # 새끼 MCP
	18: 18,  # 새끼 PIP
	19: 19,  # 새끼 DIP
	20: 20,  # 새끼 끝
}

# =============================================================================
# WebCamTeleoperator 클래스
# =============================================================================

class WebCamTeleoperator:
	"""
	웹캠 기반 RUKA 로봇 손 원격조정 클래스
	
	MediaPipe로 손을 추적하고 Oculus 텔레오퍼레이터와 동일한
	좌표계 변환을 적용하여 로봇 손을 제어합니다.
	"""

	def __init__(
		self,
		camera_id=0,
		frequency=30,
		moving_average_limit=10,
		hands=["left", "right"],
		detection_confidence=0.7,
		tracking_confidence=0.7,
		coordinate_scale=1000.0,  # 픽셀 → 미터 변환 스케일
	):
		"""
		WebCamTeleoperator 초기화
		
		Parameters:
		-----------
		camera_id : int
			웹캠 ID (기본값: 0 = 기본 카메라)
		frequency : int
			제어 주파수 (Hz)
		moving_average_limit : int
			이동평균 필터 크기
		hands : list
			제어할 손 ["left", "right"]
		detection_confidence : float
			MediaPipe 손 검출 신뢰도 (0.0~1.0)
		tracking_confidence : float
			MediaPipe 손 추적 신뢰도 (0.0~1.0)
		coordinate_scale : float
			좌표 정규화 스케일 (기본값: 1000.0)
			일반적으로 손 크기는 약 200픽셀 = 0.2미터
		"""
		
		# 타이머 초기화
		self.timer = FrequencyTimer(frequency)
		self.frequency = frequency
		
		# 좌표 변환 스케일
		self.coordinate_scale = coordinate_scale
		
		# 웹캠 초기화
		print(f"[INFO] 웹캠 초기화 중... (카메라 ID: {camera_id})")
		self.cap = cv2.VideoCapture(camera_id)
		
		if not self.cap.isOpened():
			raise RuntimeError(
				f"[ERROR] 웹캠 열기 실패 (카메라 ID: {camera_id})\n"
				f"  해결 방법:\n"
				f"    1. 카메라가 연결되어 있는지 확인\n"
				f"    2. 다른 프로그램이 카메라를 사용 중인지 확인\n"
				f"    3. 카메라 권한 확인 (Linux: /dev/video*)\n"
				f"    4. 다른 카메라 ID로 시도 (예: -c 1)"
			)
		
		# 웹캠 설정
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		self.cap.set(cv2.CAP_PROP_FPS, 60)
		
		# 실제 설정된 값 확인
		actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
		
		print(f"  ✓ 웹캠 해상도: {actual_width}x{actual_height}")
		print(f"  ✓ 웹캠 FPS: {actual_fps}")
		
		# MediaPipe HandDetector 초기화
		try:
			self.detector = HandDetector(
				staticMode=False,
				maxHands=2,
				modelComplexity=1,
				detectionCon=detection_confidence,
				minTrackCon=tracking_confidence
			)
			print(f"  ✓ MediaPipe HandDetector 초기화 완료")
		except Exception as e:
			self.cap.release()
			raise RuntimeError(
				f"[ERROR] MediaPipe 초기화 실패: {e}\n"
				f"  해결 방법:\n"
				f"    1. mediapipe 설치 확인: pip install mediapipe\n"
				f"    2. opencv-python 설치 확인: pip install opencv-python\n"
				f"    3. HandTrackingModule.py 파일 존재 확인"
			)
		
		# Oculus와 동일한 너클 포인트 사용
		self.knuckle_points = (
			5,   # 검지 MCP (OCULUS_JOINTS["knuckles"][0])
			17,  # 새끼 MCP (OCULUS_JOINTS["knuckles"][-2])
		)
		
		# 이동평균 필터
		self.moving_average_limit = moving_average_limit
		self.coord_moving_average_queues = {"left": [], "right": []}
		self.frame_moving_average_queues = {"left": [], "right": []}
		
		# 제어할 손
		self.hand_names = hands
		self.hands = {}  # RUKAOperator 인스턴스 저장
		
		# FPS 계산용
		self.frame_count = 0
		self.start_time = None
		
		print("=" * 60)
		print("WebCam Teleoperator 초기화 완료")
		print(f"카메라 ID: {camera_id}")
		print(f"제어 주파수: {frequency} Hz")
		print(f"제어 대상: {hands}")
		print(f"좌표 스케일: {coordinate_scale}")
		print("=" * 60)

	def _init_hands(self):
		"""RUKAOperator 초기화"""
		print("\n[INFO] 로봇 손 초기화 중...")
		
		for hand_name in self.hand_names:
			try:
				self.hands[hand_name] = RUKAOperator(
					hand_type=hand_name,
					moving_average_limit=5,
				)
				print(f"  ✓ {hand_name.upper()} 로봇 손 초기화 완료")
			except Exception as e:
				print(f"  ✗ {hand_name.upper()} 로봇 손 초기화 실패: {e}")
				# 한쪽 손 실패해도 계속 진행
				continue
		
		if not self.hands:
			raise RuntimeError(
				"[ERROR] 로봇 손 초기화 실패\n"
				"  해결 방법:\n"
				"    1. Dynamixel 모터 전원 확인\n"
				"    2. USB 연결 확인\n"
				"    3. constants.py의 USB_PORTS 설정 확인"
			)
		
		print(f"[INFO] {len(self.hands)}개 로봇 손 초기화 완료\n")

	def _mediapipe_to_oculus_format(self, lmList):
		"""
		MediaPipe 랜드마크를 Oculus 형식으로 변환
		
		Parameters:
		-----------
		lmList : list
			MediaPipe 21개 랜드마크 [[x, y, z], ...]
		
		Returns:
		--------
		numpy.ndarray
			Oculus 형식 키포인트 (21, 3)
		"""
		# MediaPipe는 21개, Oculus도 주요 관절 21개 사용
		keypoints = np.zeros((21, 3))
		
		for mp_idx, oculus_idx in MEDIAPIPE_TO_OCULUS_MAPPING.items():
			if mp_idx < len(lmList):
				# MediaPipe는 픽셀 좌표 [x, y, z]
				# z는 손목 기준 상대 깊이
				keypoints[oculus_idx] = lmList[mp_idx][:3]
		
		return keypoints

	def _translate_coords(self, hand_coords):
		"""
		손목을 원점으로 하는 상대 좌표계로 변환
		(Oculus 텔레오퍼레이터와 동일)
		"""
		return copy(hand_coords) - hand_coords[0]

	def _get_hand_dir_frame(
		self, origin_coord, index_knuckle_coord, pinky_knuckle_coord, hand_name
	):
		"""
		손 방향 프레임 계산
		(Oculus 텔레오퍼레이터와 동일)
		
		Returns:
		--------
		list
			[origin, X축, Y축, Z축]
		"""
		
		if hand_name == "left":
			palm_normal = normalize_vector(
				np.cross(index_knuckle_coord, pinky_knuckle_coord)
			)  # Unity space - Y
		else:
			palm_normal = normalize_vector(
				np.cross(pinky_knuckle_coord, index_knuckle_coord)
			)  # Unity space - Y
		
		palm_direction = normalize_vector(
			index_knuckle_coord + pinky_knuckle_coord
		)  # Unity space - Z
		
		if hand_name == "left":
			cross_product = normalize_vector(
				index_knuckle_coord - pinky_knuckle_coord
			)  # Unity space - X
		else:
			cross_product = normalize_vector(
				pinky_knuckle_coord - index_knuckle_coord
			)  # Unity space - X
		
		return [origin_coord, cross_product, palm_normal, palm_direction]

	def _get_ordered_joints(self, projected_translated_coords):
		"""
		관절 데이터를 5x5x3 형태로 재구성
		(Oculus 텔레오퍼레이터와 동일)
		"""
		# HAND_JOINTS 정의에 따라 추출
		extracted_joints = {
			joint: projected_translated_coords[indices]
			for joint, indices in HAND_JOINTS.items()
		}
		
		# 5개 손가락 순서로 연결
		ordered_joints = np.concatenate(
			[extracted_joints[joint] * 100.0 for joint in HAND_JOINTS],
			axis=0,
		)
		
		# (5, 5, 3) 형태로 재구성
		reshaped_joints = ordered_joints.reshape(5, 5, 3)
		
		return reshaped_joints

	def transform_keypoints(self, hand_coords, hand_name):
		"""
		키포인트 좌표계 변환
		(Oculus 텔레오퍼레이터와 동일)
		
		Parameters:
		-----------
		hand_coords : numpy.ndarray
			손 키포인트 좌표 (21, 3)
		hand_name : str
			"left" or "right"
		
		Returns:
		--------
		tuple
			(ordered_joints, hand_dir_frame)
		"""
		# 1. 손목 기준 상대 좌표로 변환
		translated_coords = self._translate_coords(hand_coords)

		# 2. 손 방향 프레임 계산
		hand_dir_frame = self._get_hand_dir_frame(
			hand_coords[0],
			translated_coords[self.knuckle_points[0]],
			translated_coords[self.knuckle_points[1]],
			hand_name,
		)
		
		# 3. 회전 변환 적용
		transformation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
		rotation_matrix = np.array(hand_dir_frame[1:])
		transformed_rotation_matrix = transformation_matrix @ rotation_matrix
		projected_translated_coords = (
			translated_coords @ transformed_rotation_matrix.T
		)
		
		# 4. 관절 데이터 재구성
		ordered_joints = self._get_ordered_joints(projected_translated_coords)
		
		return ordered_joints, hand_dir_frame

	def _operate_hand(self, hand_name, transformed_hand_coords):
		"""
		로봇 손 제어
		(Oculus 텔레오퍼레이터와 동일)
		"""
		if hand_name in self.hands.keys():
			# 이동평균 필터 적용
			transformed_hand_coords = moving_average(
				transformed_hand_coords,
				self.coord_moving_average_queues[hand_name],
				self.moving_average_limit,
			)
			
			# 로봇 제어 명령
			try:
				self.hands[hand_name].step(transformed_hand_coords)
			except Exception as e:
				print(f"[WARNING] {hand_name} 로봇 제어 실패: {e}")

	def _process_frame(self, img):
		"""
		프레임 처리 및 손 검출
		
		Parameters:
		-----------
		img : numpy.ndarray
			입력 이미지
		
		Returns:
		--------
		tuple
			(hand_data dict, processed_img)
		"""
		try:
			# MediaPipe로 손 검출
			hands, img = self.detector.findHands(img, draw=True, flipType=True)
		except Exception as e:
			print(f"[WARNING] 손 검출 실패: {e}")
			return {}, img
		
		# 검출된 손 데이터 저장
		hand_data = {}
		
		if hands:
			for hand in hands:
				try:
					hand_type = hand["type"].lower()  # "Left" or "Right" → "left" or "right"
					lmList = hand["lmList"]  # 21개 랜드마크
					
					# MediaPipe → Oculus 형식 변환
					oculus_format = self._mediapipe_to_oculus_format(lmList)
					
					# 미터 단위로 정규화 (픽셀 → 미터)
					# coordinate_scale로 조절 가능 (기본값 1000.0)
					oculus_format = oculus_format / self.coordinate_scale
					
					hand_data[hand_type] = oculus_format
				except Exception as e:
					print(f"[WARNING] {hand_type} 손 데이터 변환 실패: {e}")
					continue
		
		return hand_data, img

	def _run_robots(self):
		"""메인 제어 루프"""
		# 프레임 읽기
		success, img = self.cap.read()
		
		if not success:
			print("[WARNING] 웹캠 프레임 읽기 실패")
			return None
		
		# 손 검출 및 처리
		hand_data, img = self._process_frame(img)
		
		# 각 손에 대해 처리
		for hand_name in ["left", "right"]:
			if hand_name in hand_data:
				try:
					# 좌표계 변환
					transformed_hand_coords, _ = self.transform_keypoints(
						hand_data[hand_name], hand_name
					)
					
					# 로봇 제어
					self._operate_hand(hand_name, transformed_hand_coords)
				except Exception as e:
					print(f"[WARNING] {hand_name} 손 처리 실패: {e}")
					continue
		
		return img

	def _cleanup(self):
		"""리소스 정리"""
		print("\n[INFO] 리소스 정리 중...")
		
		# 로봇 손 토크 비활성화
		for hand_name, hand in self.hands.items():
			try:
				print(f"  → {hand_name.upper()} 토크 비활성화...")
				hand.operator.set_torque_enabled(False)
				print(f"    ✓ {hand_name.upper()} 토크 비활성화 완료")
			except Exception as e:
				print(f"    ✗ {hand_name.upper()} 토크 비활성화 실패: {e}")
		
		# 웹캠 해제
		if self.cap is not None and self.cap.isOpened():
			self.cap.release()
			print("  ✓ 웹캠 해제 완료")
		
		# OpenCV 창 닫기
		cv2.destroyAllWindows()
		print("  ✓ OpenCV 창 닫기 완료")
		
		print("[INFO] 리소스 정리 완료")
		print("=" * 60)

	def run(self):
		"""
		메인 실행 루프
		
		웹캠에서 손을 추적하고 로봇을 제어합니다.
		'q' 키를 누르면 종료합니다.
		"""
		
		try:
			# 로봇 손 초기화
			self._init_hands()
			
			print("\n[INFO] 텔레오퍼레이션 시작")
			print("[INFO] 종료하려면 'q' 키를 누르세요")
			print("=" * 60)
			
			self.frame_count = 0
			self.start_time = time.time()
			
			while True:
				# 타이머 시작
				self.timer.start_loop()
				
				# 로봇 제어
				img = self._run_robots()
				
				if img is not None:
					# FPS 계산 (실제 경과 시간 기반)
					self.frame_count += 1
					elapsed_time = time.time() - self.start_time
					fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
					
					# FPS 표시
					cv2.putText(
						img,
						f"FPS: {fps:.1f}",
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,
						(0, 255, 0),
						2
					)
					
					# 제어 상태 표시
					status_text = "Controlling: "
					active_hands = []
					for hand_name in self.hand_names:
						if hand_name in self.hands.keys():
							active_hands.append(hand_name.upper())
					
					status_text += ", ".join(active_hands) if active_hands else "NONE"
					
					cv2.putText(
						img,
						status_text,
						(10, 70),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.7,
						(0, 255, 255),
						2
					)
					
					# 화면 표시
					cv2.imshow("WebCam Teleoperator - RUKA Hand", img)
				
				# 타이머 종료
				self.timer.end_loop()
				
				# 'q' 키로 종료
				key = cv2.waitKey(1) & 0xFF
				if key == ord('q'):
					print("\n[INFO] 사용자 종료 요청")
					break
				elif key == ord('r'):
					# 'r' 키로 FPS 카운터 리셋
					self.frame_count = 0
					self.start_time = time.time()
					print("[INFO] FPS 카운터 리셋")
		
		except KeyboardInterrupt:
			print("\n[INFO] Ctrl+C로 종료")
		
		except Exception as e:
			print(f"\n[ERROR] 예상치 못한 오류: {e}")
			import traceback
			traceback.print_exc()
		
		finally:
			# 리소스 정리
			self._cleanup()


# =============================================================================
# 메인 실행
# =============================================================================

def main():
	"""
	WebCam Teleoperator 실행

	사용법:
	-------
	python webcam_teleoperator.py

	종료:
	-----
	- 'q' 키 입력
	- Ctrl+C
	
	키보드 단축키:
	-------------
	- 'q': 종료
	- 'r': FPS 카운터 리셋
	"""

	try:
		# 텔레오퍼레이터 생성
		teleoperator = WebCamTeleoperator(
			camera_id=0,                    # 기본 웹캠
			frequency=30,                   # 30Hz (MediaPipe 권장)
			moving_average_limit=10,        # 이동평균 필터
			hands=["left", "right"],        # 양손 제어
			detection_confidence=0.7,       # 검출 신뢰도
			tracking_confidence=0.7,        # 추적 신뢰도
			coordinate_scale=1000.0,        # 좌표 정규화 스케일
		)

		# 실행
		teleoperator.run()
	
	except Exception as e:
		print(f"\n[ERROR] 초기화 실패: {e}")
		import traceback
		traceback.print_exc()
		return 1
	
	return 0


if __name__ == "__main__":
	exit(main())