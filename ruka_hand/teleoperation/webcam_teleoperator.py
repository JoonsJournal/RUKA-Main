#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
WebCam Teleoperator - MediaPipe 기반 RUKA 로봇 손 원격조정
이 모듈은 웹캠과 MediaPipe를 사용하여 RUKA 로봇 손을 실시간으로 제어합니다.
주요 기능:
1. 웹캠에서 손 추적 (MediaPipe Hands)
1. 21개 랜드마크를 로봇 제어 데이터로 변환
1. Oculus 텔레오퍼레이터와 동일한 좌표계 변환 로직 사용
1. 실시간 로봇 손 제어
작성자: 이동준
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

from copy import deepcopy as copy
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gesture_ctrled_ruka.HandTrackingModule import HandDetector
from ruka_hand.control.operator import RUKAOperator
from ruka_hand.utils.constants import *
from ruka_hand.utils.timer import FrequencyTimer
from ruka_hand.utils.vectorops import *

# =============================================================================
# MediaPipe 21개 랜드마크를 손가락별로 재구성
# =============================================================================
# MediaPipe: 0(손목), 1-4(엄지), 5-8(검지), 9-12(중지), 13-16(약지), 17-20(새끼)
# 목표: 각 손가락마다 5개 관절 (손목 포함) → (5, 5, 3) 형태

MEDIAPIPE_FINGER_INDICES = {
    "thumb": [0, 1, 2, 3, 4],    # 손목 + 엄지 4개
    "index": [0, 5, 6, 7, 8],    # 손목 + 검지 4개
    "middle": [0, 9, 10, 11, 12], # 손목 + 중지 4개
    "ring": [0, 13, 14, 15, 16],  # 손목 + 약지 4개
    "pinky": [0, 17, 18, 19, 20], # 손목 + 새끼 4개
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
		"""
		
		# 타이머 초기화
		self.timer = FrequencyTimer(frequency)
		self.frequency = frequency
		
		# 웹캠 초기화
		self.cap = cv2.VideoCapture(camera_id)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		self.cap.set(cv2.CAP_PROP_FPS, 60)
		
		# MediaPipe HandDetector 초기화
		self.detector = HandDetector(
			staticMode=False,
			maxHands=2,
			modelComplexity=1,
			detectionCon=detection_confidence,
			minTrackCon=tracking_confidence
		)
		
		# 너클 포인트 (검지 MCP, 새끼 MCP)
		self.knuckle_points = (5, 17)
		
		# 이동평균 필터
		self.moving_average_limit = moving_average_limit
		self.coord_moving_average_queues = {"left": [], "right": []}
		self.frame_moving_average_queues = {"left": [], "right": []}
		
		# 제어할 손
		self.hand_names = hands
		
		print("=" * 60)
		print("WebCam Teleoperator 초기화 완료")
		print(f"카메라 ID: {camera_id}")
		print(f"제어 주파수: {frequency} Hz")
		print(f"제어 대상: {hands}")
		print("=" * 60)

	def _init_hands(self):
		"""RUKAOperator 초기화"""
		self.hands = {}
		for hand_name in self.hand_names:
			self.hands[hand_name] = RUKAOperator(
				hand_type=hand_name,
				moving_average_limit=5,
			)
			print(f"[INFO] {hand_name} 로봇 손 초기화 완료")

	def _mediapipe_to_finger_keypoints(self, lmList):
		"""
		MediaPipe 21개 랜드마크를 (5, 5, 3) 형태로 변환
		
		Parameters:
		-----------
		lmList : list
			MediaPipe 21개 랜드마크 [[x, y, z], ...]
		
		Returns:
		--------
		numpy.ndarray
			손가락별 키포인트 (5, 5, 3)
			- 5개 손가락 (엄지, 검지, 중지, 약지, 새끼)
			- 각 손가락당 5개 관절
			- 각 관절당 3D 좌표 (x, y, z)
		"""
		keypoints = np.zeros((5, 5, 3))
		
		finger_names = ["thumb", "index", "middle", "ring", "pinky"]
		
		for finger_idx, finger_name in enumerate(finger_names):
			indices = MEDIAPIPE_FINGER_INDICES[finger_name]
			for joint_idx, mp_idx in enumerate(indices):
				if mp_idx < len(lmList):
					# MediaPipe 좌표 [x, y, z] 추출
					keypoints[finger_idx, joint_idx] = lmList[mp_idx][:3]
		
		return keypoints

	def _translate_coords(self, hand_coords):
		"""
		손목을 원점으로 하는 상대 좌표계로 변환
		(Oculus 텔레오퍼레이터와 동일)
		"""
		# 각 손가락의 첫 번째 관절(손목)을 기준으로 변환
		wrist = hand_coords[0, 0]  # (3,)
		translated = copy(hand_coords)
		for finger_idx in range(5):
			translated[finger_idx] = translated[finger_idx] - wrist
		return translated

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

	def transform_keypoints(self, hand_coords, hand_name):
		"""
		키포인트 좌표계 변환
		(Oculus 텔레오퍼레이터와 동일)
		
		Parameters:
		-----------
		hand_coords : numpy.ndarray
			손 키포인트 좌표 (5, 5, 3)
		hand_name : str
			"left" or "right"
		
		Returns:
		--------
		tuple
			(transformed_keypoints, hand_dir_frame)
		"""
		# 1. 손목 기준 상대 좌표로 변환
		translated_coords = self._translate_coords(hand_coords)

		# 2. 손 방향 프레임 계산 (검지 MCP, 새끼 MCP 사용)
		wrist_pos = hand_coords[0, 0]  # 원본 손목 위치
		index_knuckle = translated_coords[1, 1]  # 검지 MCP (상대 좌표)
		pinky_knuckle = translated_coords[4, 1]  # 새끼 MCP (상대 좌표)
		
		hand_dir_frame = self._get_hand_dir_frame(
			wrist_pos,
			index_knuckle,
			pinky_knuckle,
			hand_name,
		)
		
		# 3. 회전 변환 적용
		transformation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
		rotation_matrix = np.array(hand_dir_frame[1:])
		transformed_rotation_matrix = transformation_matrix @ rotation_matrix
		
		# 각 손가락의 모든 관절에 회전 적용
		projected_coords = np.zeros_like(translated_coords)
		for finger_idx in range(5):
			projected_coords[finger_idx] = (
				translated_coords[finger_idx] @ transformed_rotation_matrix.T
			)
		
		# 센티미터 단위로 스케일링 (미터 → cm)
		projected_coords = projected_coords * 100.0
		
		return projected_coords, hand_dir_frame

	def _operate_hand(self, hand_name, transformed_hand_coords):
		"""
		로봇 손 제어
		(Oculus 텔레오퍼레이터와 동일)
		"""
		if hand_name in self.hands.keys():
			try:
				print(f"[DEBUG] {hand_name} 손 제어 시작")
				print(f"[DEBUG] 변환된 좌표 형태: {transformed_hand_coords.shape}")
				print(f"[DEBUG] 변환된 좌표 범위: min={transformed_hand_coords.min():.2f}, max={transformed_hand_coords.max():.2f}")
				
				# 이동평균 필터 적용
				transformed_hand_coords = moving_average(
					transformed_hand_coords,
					self.coord_moving_average_queues[hand_name],
					self.moving_average_limit,
				)
				
				print(f"[DEBUG] 필터링 후 좌표 범위: min={transformed_hand_coords.min():.2f}, max={transformed_hand_coords.max():.2f}")
				
				# 로봇 제어 명령
				self.hands[hand_name].step(transformed_hand_coords)
				print(f"[DEBUG] {hand_name} 손 제어 명령 전송 완료")
			except Exception as e:
				print(f"[WARNING] {hand_name} 손 처리 실패: {e}")
				import traceback
				traceback.print_exc()

	def _process_frame(self, img):
		"""
		프레임 처리 및 손 검출
		
		Parameters:
		-----------
		img : numpy.ndarray
			입력 이미지
		
		Returns:
		--------
		dict
			{"left": hand_coords, "right": hand_coords}
		"""
		# MediaPipe로 손 검출
		hands, img = self.detector.findHands(img, draw=True, flipType=True)
		
		# 검출된 손 데이터 저장
		hand_data = {}
		
		if hands:
			print(f"[DEBUG] 감지된 손 개수: {len(hands)}")
			for hand in hands:
				hand_type = hand["type"].lower()  # "Left" or "Right" → "left" or "right"
				lmList = hand["lmList"]  # 21개 랜드마크
				
				print(f"[DEBUG] {hand_type} 손 감지됨, 랜드마크 수: {len(lmList)}")
				
				# MediaPipe → (5, 5, 3) 형식 변환
				finger_keypoints = self._mediapipe_to_finger_keypoints(lmList)
				
				print(f"[DEBUG] {hand_type} 키포인트 형태: {finger_keypoints.shape}")
				print(f"[DEBUG] {hand_type} 키포인트 범위: min={finger_keypoints.min():.2f}, max={finger_keypoints.max():.2f}")
				
				# 미터 단위로 정규화 (픽셀 → 미터)
				# 일반적으로 손 크기는 약 200픽셀 = 0.2미터
				finger_keypoints = finger_keypoints / 1000.0
				
				hand_data[hand_type] = finger_keypoints
		else:
			print("[DEBUG] 손이 감지되지 않음")
		
		return hand_data, img

	def _run_robots(self):
		"""메인 제어 루프"""
		# 프레임 읽기
		success, img = self.cap.read()
		
		if not success:
			print("[ERROR] 웹캠 프레임 읽기 실패")
			return None
		
		# 손 검출 및 처리
		hand_data, img = self._process_frame(img)
		
		print(f"[DEBUG] 처리된 손 데이터: {list(hand_data.keys())}")
		
		# 각 손에 대해 처리
		for hand_name in self.hand_names:
			if hand_name in hand_data:
				print(f"[DEBUG] {hand_name} 손 변환 시작")
				# 좌표계 변환
				transformed_hand_coords, _ = self.transform_keypoints(
					hand_data[hand_name], hand_name
				)
				
				# 로봇 제어
				self._operate_hand(hand_name, transformed_hand_coords)
			else:
				print(f"[DEBUG] {hand_name} 손 데이터 없음")
		
		return img

	def run(self):
		"""
		메인 실행 루프
		
		웹캠에서 손을 추적하고 로봇을 제어합니다.
		'q' 키를 누르면 종료합니다.
		"""
		
		# 로봇 손 초기화
		self._init_hands()
		
		print("\n[INFO] 텔레오퍼레이션 시작")
		print("[INFO] 종료하려면 'q' 키를 누르세요")
		print("=" * 60)
		
		frame_count = 0
		
		try:
			while True:
				# 타이머 시작
				self.timer.start_loop()
				
				# 로봇 제어
				img = self._run_robots()
				
				if img is not None:
					# FPS 계산
					frame_count += 1
					fps = frame_count / (frame_count / self.frequency)
					
					# FPS 표시
					cv2.putText(
						img,
						f"FPS: {int(fps)}",
						(10, 30),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,
						(0, 255, 0),
						2
					)
					
					# 제어 상태 표시
					status_text = "Controlling: "
					for hand_name in self.hand_names:
						if hand_name in self.hands.keys():
							status_text += f"{hand_name.upper()} "
					
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
				if cv2.waitKey(1) & 0xFF == ord('q'):
					print("\n[INFO] 사용자 종료 요청")
					break
		
		except KeyboardInterrupt:
			print("\n[INFO] Ctrl+C로 종료")
		
		finally:
			# 리소스 정리
			print("\n[INFO] 리소스 정리 중...")
			
			# 로봇 토크 비활성화
			for hand_name in self.hands.keys():
				try:
					print(f" -> {hand_name.upper()} 토크 비활성화...")
					if hasattr(self.hands[hand_name], 'controller'):
						self.hands[hand_name].controller.hand.disable_torque()
						print(f"     ✓ {hand_name.upper()} 토크 비활성화 완료")
				except Exception as e:
					print(f"     ✗ {hand_name.upper()} 토크 비활성화 실패: {e}")
			
			# 웹캠 해제
			self.cap.release()
			print("     ✓ 웹캠 해제 완료")
			
			# OpenCV 창 닫기
			cv2.destroyAllWindows()
			print("     ✓ OpenCV 창 닫기 완료")
			
			print(" [INFO] 리소스 정리 완료")
			print("=" * 60)


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
	"""

	# 텔레오퍼레이터 생성
	teleoperator = WebCamTeleoperator(
		camera_id=0,                    # 기본 웹캠
		frequency=30,                   # 30Hz (MediaPipe 권장)
		moving_average_limit=10,        # 이동평균 필터
		hands=["left", "right"],        # 양손 제어
		detection_confidence=0.7,       # 검출 신뢰도
		tracking_confidence=0.7,        # 추적 신뢰도
	)

	# 실행
	teleoperator.run()

if __name__ == "__main__":
	main()