#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RUKA 웹캠 텔레오퍼레이션 제스처-모터 매핑 검증 프로그램

이 프로그램은 웹캠 기반 텔레오퍼레이션에서 사람의 손 동작이 
RUKA 로봇 손에 정확하게 매핑되는지 검증합니다.

검증 항목:
1. Open Hand (손 펴기) - tension_limits와 비교
2. Closed Hand Type A (엄지가 안쪽) - curl_limits와 비교  
3. Closed Hand Type B (엄지가 바깥쪽) - curl_limits와 비교
4. Interactive 중간 검증 - 실시간 모터 값 확인

작성자: 이동준
수정: webcam_teleoperator.py 방식으로 변환 (HAND_JOINTS 사용 안 함)
"""

import os
import sys
import time
import threading
from copy import deepcopy as copy
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# 프로젝트 경로 추가
sys.path.insert(0, '/mnt/project')

try:
    from HandTrackingModule import HandDetector
    from ruka_hand.control.operator import RUKAOperator
    from ruka_hand.utils.constants import *
    from ruka_hand.utils.timer import FrequencyTimer
    from ruka_hand.utils.vectorops import *
    from ruka_hand.utils.file_ops import get_repo_root
except ImportError as e:
    print(f"[WARNING] 일부 모듈 임포트 실패: {e}")
    print("[INFO] 시뮬레이션 모드로 실행됩니다.")

# =============================================================================
# 설정 상수 - webcam_teleoperator.py 방식 사용
# =============================================================================

# MediaPipe 21개 랜드마크를 손가락별로 재구성 (webcam_teleoperator.py와 동일)
MEDIAPIPE_FINGER_INDICES = {
    "thumb": [0, 1, 2, 3, 4],    # 손목 + 엄지 4개
    "index": [0, 5, 6, 7, 8],    # 손목 + 검지 4개
    "middle": [0, 9, 10, 11, 12], # 손목 + 중지 4개
    "ring": [0, 13, 14, 15, 16],  # 손목 + 약지 4개
    "pinky": [0, 17, 18, 19, 20], # 손목 + 새끼 4개
}

# 손가락 이름
FINGER_NAMES = ["엄지(Thumb)", "검지(Index)", "중지(Middle)", "약지(Ring)", "새끼(Pinky)"]

# 검증 설정
VALIDATION_REPEAT = 3  # 각 포즈 검증 반복 횟수
WAIT_SECONDS = 2       # 포즈 유지 대기 시간
TOLERANCE_PERCENT = 15  # 모터 값 허용 오차 (%)

# =============================================================================
# GestureMappingValidator 클래스
# =============================================================================

class GestureMappingValidator:
    """
    제스처-모터 매핑 검증 클래스
    
    주요 기능:
    1. 손 포즈 검출 (펴짐/접힘)
    2. 모터 값 수집 및 비교
    3. 대화형 검증 인터페이스
    4. 검증 결과 리포트 생성
    """
    
    def __init__(
        self,
        camera_id=0,
        hand_type="right",
        frequency=30,
        detection_confidence=0.7,
        tracking_confidence=0.7,
        simulation_mode=False,
    ):
        """
        초기화
        
        Args:
            camera_id: 웹캠 ID
            hand_type: "right" or "left"
            frequency: 제어 주파수 (Hz)
            detection_confidence: 손 검출 신뢰도
            tracking_confidence: 손 추적 신뢰도
            simulation_mode: True면 실제 로봇 연결 없이 테스트
        """
        
        print("\n" + "=" * 70)
        print("RUKA 제스처-모터 매핑 검증 프로그램")
        print("Gesture-Motor Mapping Validation Tool")
        print("=" * 70)
        
        self.hand_type = hand_type
        self.frequency = frequency
        self.simulation_mode = simulation_mode
        
        # 타이머 초기화
        self.timer = FrequencyTimer(frequency)
        
        # 웹캠 초기화
        print(f"\n[초기화] 웹캠 설정 중... (ID: {camera_id})")
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        print("  ✓ 웹캠 초기화 완료")
        
        # MediaPipe HandDetector 초기화
        print("[초기화] MediaPipe HandDetector 설정 중...")
        self.detector = HandDetector(
            staticMode=False,
            maxHands=2,
            modelComplexity=1,
            detectionCon=detection_confidence,
            minTrackCon=tracking_confidence
        )
        print("  ✓ HandDetector 초기화 완료")
        
        # 너클 포인트 (검지 MCP, 새끼 MCP) - webcam_teleoperator와 동일
        self.knuckle_points = (5, 17)
        
        # 이동평균 필터
        self.moving_average_limit = 10
        self.coord_moving_average_queues = {"left": [], "right": []}
        
        # 캘리브레이션 데이터 로드
        self._load_calibration_data()
        
        # 로봇 연결 (시뮬레이션 모드가 아닐 때)
        self.robot = None
        if not simulation_mode:
            self._init_robot()
        else:
            print("\n[INFO] 시뮬레이션 모드 - 로봇 연결 없음")
        
        # 검증 결과 저장
        self.validation_results = {
            "open_hand": [],
            "closed_hand_type_a": [],
            "closed_hand_type_b": [],
            "interactive": [],
        }
        
        # 엄지 굽힘 속도 추적 (Type A/B 구분용)
        self.thumb_curl_history = []
        self.thumb_curl_timestamps = []
        
        # 키보드 인터럽트 플래그
        self.paused = False
        self.quit_flag = False
        
        print("\n" + "=" * 70)
        print("검증 프로그램 초기화 완료")
        print("=" * 70)
    
    def _load_calibration_data(self):
        """캘리브레이션 데이터 로드"""
        print(f"\n[초기화] 캘리브레이션 데이터 로드 중...")
        
        try:
            repo_root = get_repo_root()
        except:
            repo_root = "/mnt/project"
        
        # Curl Limits 로드
        curl_path = f"{repo_root}/motor_limits/{self.hand_type}_curl_limits.npy"
        if os.path.exists(curl_path):
            self.curl_limits = np.load(curl_path)
            print(f"  ✓ Curl Limits 로드: {curl_path}")
            print(f"    값 범위: [{self.curl_limits.min():.0f}, {self.curl_limits.max():.0f}]")
        else:
            print(f"  ⚠ Curl Limits 파일 없음, 기본값 사용")
            self.curl_limits = np.ones(11) * 2000  # 기본값
        
        # Tension Limits 로드
        tens_path = f"{repo_root}/motor_limits/{self.hand_type}_tension_limits.npy"
        if os.path.exists(tens_path):
            self.tension_limits = np.load(tens_path)
            print(f"  ✓ Tension Limits 로드: {tens_path}")
            print(f"    값 범위: [{self.tension_limits.min():.0f}, {self.tension_limits.max():.0f}]")
        else:
            print(f"  ⚠ Tension Limits 파일 없음, 기본값 사용")
            self.tension_limits = np.ones(11) * 1000  # 기본값
    
    def _init_robot(self):
        """로봇 초기화"""
        print(f"\n[초기화] RUKA 로봇 연결 중... ({self.hand_type})")
        try:
            self.robot = RUKAOperator(
                hand_type=self.hand_type,
                moving_average_limit=5,
            )
            print(f"  ✓ 로봇 연결 성공")
        except Exception as e:
            print(f"  ✗ 로봇 연결 실패: {e}")
            print(f"  → 시뮬레이션 모드로 전환")
            self.simulation_mode = True
            self.robot = None
    
    # =========================================================================
    # 좌표 변환 메서드 (webcam_teleoperator.py와 동일)
    # =========================================================================
    
    def _mediapipe_to_finger_keypoints(self, lmList):
        """
        MediaPipe 21개 랜드마크를 (5, 5, 3) 형태로 변환
        (webcam_teleoperator.py와 동일)
        
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
        (webcam_teleoperator.py와 동일)
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
        (webcam_teleoperator.py와 동일)
        
        Returns:
        --------
        list
            [origin, X축, Y축, Z축]
        """
        
        if hand_name == "left":
            palm_normal = normalize_vector(
                np.cross(index_knuckle_coord, pinky_knuckle_coord)
            )
        else:
            palm_normal = normalize_vector(
                np.cross(pinky_knuckle_coord, index_knuckle_coord)
            )
        
        palm_direction = normalize_vector(
            index_knuckle_coord + pinky_knuckle_coord
        )
        
        if hand_name == "left":
            cross_product = normalize_vector(
                index_knuckle_coord - pinky_knuckle_coord
            )
        else:
            cross_product = normalize_vector(
                pinky_knuckle_coord - index_knuckle_coord
            )
        
        return [origin_coord, cross_product, palm_normal, palm_direction]
    
    def transform_keypoints(self, hand_coords, hand_name):
        """
        키포인트 좌표계 변환
        (webcam_teleoperator.py와 동일)
        
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
    
    def _calculate_motor_positions(self, keypoints):
        """
        키포인트에서 모터 위치 계산 (실제 제어 시 사용되는 값)
        
        Returns:
            dict: 각 손가락별 모터 위치 값
        """
        fingertips = calculate_fingertips(keypoints)
        joint_angles = calculate_joint_angles(keypoints)
        
        # 실제 컨트롤러가 사용하는 입력 형태
        return {
            "fingertips": fingertips,
            "joint_angles": joint_angles,
            "raw_keypoints": keypoints,
        }
    
    # =========================================================================
    # 손 포즈 검출 메서드
    # =========================================================================
    
    def detect_hand_pose(self, img):
        """
        현재 손 포즈 검출
        
        Returns:
            dict: 손 정보 (fingers_up, hand_type, landmarks 등)
        """
        hands, img_with_landmarks = self.detector.findHands(img, draw=True, flipType=True)
        
        result = {
            "detected": False,
            "img": img_with_landmarks,
            "hands": [],
        }
        
        if hands:
            for hand in hands:
                hand_info = {
                    "type": hand["type"].lower(),
                    "lmList": hand["lmList"],
                    "bbox": hand["bbox"],
                    "center": hand["center"],
                    "fingers_up": self.detector.fingersUp(hand),
                }
                
                # 펴진 손가락 개수
                if len(hand_info["fingers_up"]) >= 5:
                    hand_info["fingers_count"] = sum(hand_info["fingers_up"][:5])
                    hand_info["thumb_rotation"] = hand_info["fingers_up"][5] if len(hand_info["fingers_up"]) > 5 else 0
                else:
                    hand_info["fingers_count"] = 0
                    hand_info["thumb_rotation"] = 0
                
                result["hands"].append(hand_info)
            
            result["detected"] = True
        
        return result
    
    def is_hand_open(self, hand_info):
        """손이 완전히 펴졌는지 판단"""
        if not hand_info or "fingers_up" not in hand_info:
            return False
        
        fingers = hand_info["fingers_up"][:5]
        # 5개 손가락 모두 펴짐 (또는 4개 이상)
        return sum(fingers) >= 4
    
    def is_hand_closed(self, hand_info):
        """손이 완전히 접혔는지 판단 (주먹)"""
        if not hand_info or "fingers_up" not in hand_info:
            return False
        
        fingers = hand_info["fingers_up"][:5]
        # 모든 손가락 접힘 (1개 이하)
        return sum(fingers) <= 1
    
    def classify_fist_type(self, hand_info, lmList):
        """
        주먹 타입 분류 (엄지 위치 기반)
        
        Type A: 엄지가 다른 4손가락 안쪽 (손바닥 쪽)
        Type B: 엄지가 다른 4손가락 바깥쪽
        
        Returns:
            str: "type_a", "type_b", or "unknown"
        """
        if not lmList or len(lmList) < 21:
            return "unknown"
        
        # 엄지 끝 (4) 위치
        thumb_tip = np.array(lmList[4][:2])
        
        # 검지 MCP (5) 위치 - 주먹 기준점
        index_mcp = np.array(lmList[5][:2])
        
        # 검지 끝 (8) 위치
        index_tip = np.array(lmList[8][:2])
        
        # 중지 MCP (9) 위치
        middle_mcp = np.array(lmList[9][:2])
        
        # 손바닥 중심 계산
        palm_center = (index_mcp + middle_mcp) / 2
        
        # 엄지-손바닥 중심 거리
        thumb_to_palm = np.linalg.norm(thumb_tip - palm_center)
        
        # 검지끝-손바닥 중심 거리
        index_to_palm = np.linalg.norm(index_tip - palm_center)
        
        # Type A: 엄지가 손바닥 안쪽 (거리가 짧음)
        # Type B: 엄지가 바깥쪽 (거리가 김)
        if thumb_to_palm < index_to_palm * 1.2:
            return "type_a"
        else:
            return "type_b"
    
    def update_thumb_curl_speed(self, thumb_curl_value):
        """엄지 굽힘 속도 업데이트 (Type A/B 구분용)"""
        current_time = time.time()
        self.thumb_curl_history.append(thumb_curl_value)
        self.thumb_curl_timestamps.append(current_time)
        
        # 최근 1초 데이터만 유지
        while self.thumb_curl_timestamps and (current_time - self.thumb_curl_timestamps[0] > 1.0):
            self.thumb_curl_history.pop(0)
            self.thumb_curl_timestamps.pop(0)
        
        # 굽힘 속도 계산
        if len(self.thumb_curl_history) >= 2:
            curl_change = self.thumb_curl_history[-1] - self.thumb_curl_history[0]
            time_change = self.thumb_curl_timestamps[-1] - self.thumb_curl_timestamps[0]
            if time_change > 0:
                return curl_change / time_change
        return 0
    
    # =========================================================================
    # 검증 메서드
    # =========================================================================
    
    def validate_open_hand(self):
        """
        Open Hand (손 펴기) 검증
        
        3회 반복하여 모터 값 수집 후 tension_limits와 비교
        """
        print("\n" + "=" * 70)
        print("Open Hand (손 펴기) 검증 시작")
        print("=" * 70)
        print(f"\n목표: 손을 완전히 펴서 tension_limits와 비교")
        print(f"반복 횟수: {VALIDATION_REPEAT}회")
        
        collected_motor_values = []
        
        for trial in range(VALIDATION_REPEAT):
            print(f"\n{'─' * 50}")
            print(f"[시도 {trial + 1}/{VALIDATION_REPEAT}]")
            print("손을 완전히 펴세요... (5손가락 모두)")
            
            # 포즈 감지 대기
            detected = False
            while not detected and not self.quit_flag:
                success, img = self.cap.read()
                if not success:
                    continue
                
                pose_result = self.detect_hand_pose(img)
                
                # 상태 표시
                status_text = "손을 펴세요..."
                if pose_result["detected"]:
                    for hand in pose_result["hands"]:
                        if hand["type"] == self.hand_type:
                            finger_count = hand.get("fingers_count", 0)
                            status_text = f"펴진 손가락: {finger_count}/5"
                            
                            if self.is_hand_open(hand):
                                status_text = "✓ 손 펴짐 감지됨!"
                                detected = True
                
                # 화면 표시
                cv2.putText(img, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Trial {trial+1}/{VALIDATION_REPEAT}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img, "Press 'q' to quit, 'p' to pause", (10, img.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Open Hand Validation", pose_result["img"])
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.quit_flag = True
                    break
                elif key == ord('p'):
                    self._pause_menu()
            
            if self.quit_flag:
                break
            
            # 사용자 확인
            print("\n손 펴짐 감지됨! 확인하시겠습니까?")
            user_input = input("계속하려면 'y' 입력 (n: 다시 시도): ").strip().lower()
            
            if user_input != 'y':
                print("다시 시도합니다...")
                continue
            
            # 대기 후 값 수집
            print(f"\n{WAIT_SECONDS}초간 포즈를 유지하세요...")
            start_time = time.time()
            motor_values_samples = []
            
            while time.time() - start_time < WAIT_SECONDS:
                success, img = self.cap.read()
                if not success:
                    continue
                
                pose_result = self.detect_hand_pose(img)
                
                if pose_result["detected"]:
                    for hand in pose_result["hands"]:
                        if hand["type"] == self.hand_type:
                            # ✅ webcam_teleoperator 방식으로 변환
                            finger_keypoints = self._mediapipe_to_finger_keypoints(hand["lmList"])
                            finger_keypoints = finger_keypoints / 1000.0  # 픽셀 → 미터
                            transformed_coords, _ = self.transform_keypoints(finger_keypoints, self.hand_type)
                            motor_data = self._calculate_motor_positions(transformed_coords)
                            motor_values_samples.append(motor_data)
                
                # 남은 시간 표시
                remaining = WAIT_SECONDS - (time.time() - start_time)
                cv2.putText(img, f"Hold pose: {remaining:.1f}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Open Hand Validation", img)
                cv2.waitKey(1)
            
            if motor_values_samples:
                # 평균값 계산
                avg_joint_angles = np.mean([m["joint_angles"] for m in motor_values_samples], axis=0)
                avg_fingertips = np.mean([m["fingertips"] for m in motor_values_samples], axis=0)
                
                collected_motor_values.append({
                    "trial": trial + 1,
                    "joint_angles": avg_joint_angles,
                    "fingertips": avg_fingertips,
                    "samples": len(motor_values_samples),
                })
                
                print(f"✓ 데이터 수집 완료 ({len(motor_values_samples)} 샘플)")
        
        # 결과 분석
        if collected_motor_values:
            self._analyze_open_hand_results(collected_motor_values)
        
        cv2.destroyWindow("Open Hand Validation")
    
    def _analyze_open_hand_results(self, collected_values):
        """Open Hand 검증 결과 분석"""
        print("\n" + "=" * 70)
        print("Open Hand 검증 결과 분석")
        print("=" * 70)
        
        # 평균 값 계산
        all_joint_angles = np.array([v["joint_angles"] for v in collected_values])
        avg_angles = np.mean(all_joint_angles, axis=0)
        std_angles = np.std(all_joint_angles, axis=0)
        
        print(f"\n수집된 데이터: {len(collected_values)}회")
        print(f"\n각 손가락별 관절 각도 (평균 ± 표준편차):")
        
        for i, name in enumerate(FINGER_NAMES):
            if i < len(avg_angles):
                angles = avg_angles[i]
                stds = std_angles[i]
                print(f"  {name}:")
                print(f"    MCP: {angles[0]:.1f}° ± {stds[0]:.1f}°")
                print(f"    PIP: {angles[1]:.1f}° ± {stds[1]:.1f}°")
                print(f"    DIP: {angles[2]:.1f}° ± {stds[2]:.1f}°")
        
        # Tension Limits 비교
        print(f"\n[Tension Limits 비교]")
        print(f"  Tension Limits: {self.tension_limits}")
        print(f"  (손이 펴졌을 때 모터가 이 값 근처에 있어야 함)")
        
        # 결과 저장
        self.validation_results["open_hand"] = {
            "collected_values": collected_values,
            "avg_angles": avg_angles,
            "std_angles": std_angles,
            "tension_limits": self.tension_limits,
        }
    
    def validate_closed_hand(self):
        """
        Closed Hand (주먹 쥐기) 검증
        
        Type A: 엄지가 안쪽 (느린 굽힘)
        Type B: 엄지가 바깥쪽 (빠른 굽힘)
        """
        print("\n" + "=" * 70)
        print("Closed Hand (주먹 쥐기) 검증 시작")
        print("=" * 70)
        
        for fist_type, description in [
            ("type_a", "엄지가 4손가락 안쪽 (천천히 쥐기)"),
            ("type_b", "엄지가 4손가락 바깥쪽 (빠르게 쥐기)")
        ]:
            print(f"\n{'─' * 50}")
            print(f"[{fist_type.upper()}] {description}")
            print(f"반복 횟수: {VALIDATION_REPEAT}회")
            
            collected_motor_values = []
            
            for trial in range(VALIDATION_REPEAT):
                print(f"\n[시도 {trial + 1}/{VALIDATION_REPEAT}]")
                print(f"주먹을 쥐세요... ({description})")
                
                # 포즈 감지 대기
                detected = False
                detected_type = None
                
                while not detected and not self.quit_flag:
                    success, img = self.cap.read()
                    if not success:
                        continue
                    
                    pose_result = self.detect_hand_pose(img)
                    
                    status_text = "주먹을 쥐세요..."
                    if pose_result["detected"]:
                        for hand in pose_result["hands"]:
                            if hand["type"] == self.hand_type:
                                if self.is_hand_closed(hand):
                                    # 주먹 타입 분류
                                    detected_type = self.classify_fist_type(hand, hand["lmList"])
                                    status_text = f"✓ 주먹 감지! (타입: {detected_type})"
                                    
                                    if detected_type == fist_type or fist_type == "any":
                                        detected = True
                                else:
                                    finger_count = hand.get("fingers_count", 5)
                                    status_text = f"더 쥐세요... (펴진 손가락: {finger_count})"
                    
                    # 화면 표시
                    cv2.putText(img, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Target: {fist_type}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Closed Hand Validation", pose_result["img"])
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.quit_flag = True
                        break
                    elif key == ord('p'):
                        self._pause_menu()
                
                if self.quit_flag:
                    break
                
                # 사용자 확인
                print(f"\n주먹 감지됨! (타입: {detected_type})")
                print("확인하시겠습니까?")
                user_input = input("계속하려면 'y' 입력 (n: 다시 시도): ").strip().lower()
                
                if user_input != 'y':
                    print("다시 시도합니다...")
                    continue
                
                # 대기 후 값 수집
                print(f"\n{WAIT_SECONDS}초간 포즈를 유지하세요...")
                start_time = time.time()
                motor_values_samples = []
                
                while time.time() - start_time < WAIT_SECONDS:
                    success, img = self.cap.read()
                    if not success:
                        continue
                    
                    pose_result = self.detect_hand_pose(img)
                    
                    if pose_result["detected"]:
                        for hand in pose_result["hands"]:
                            if hand["type"] == self.hand_type:
                                # ✅ webcam_teleoperator 방식으로 변환
                                finger_keypoints = self._mediapipe_to_finger_keypoints(hand["lmList"])
                                finger_keypoints = finger_keypoints / 1000.0
                                transformed_coords, _ = self.transform_keypoints(finger_keypoints, self.hand_type)
                                motor_data = self._calculate_motor_positions(transformed_coords)
                                motor_data["fist_type"] = detected_type
                                motor_values_samples.append(motor_data)
                    
                    remaining = WAIT_SECONDS - (time.time() - start_time)
                    cv2.putText(img, f"Hold pose: {remaining:.1f}s", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Closed Hand Validation", img)
                    cv2.waitKey(1)
                
                if motor_values_samples:
                    avg_joint_angles = np.mean([m["joint_angles"] for m in motor_values_samples], axis=0)
                    avg_fingertips = np.mean([m["fingertips"] for m in motor_values_samples], axis=0)
                    
                    collected_motor_values.append({
                        "trial": trial + 1,
                        "fist_type": detected_type,
                        "joint_angles": avg_joint_angles,
                        "fingertips": avg_fingertips,
                        "samples": len(motor_values_samples),
                    })
                    
                    print(f"✓ 데이터 수집 완료 ({len(motor_values_samples)} 샘플)")
            
            if collected_motor_values:
                key = f"closed_hand_{fist_type}"
                self._analyze_closed_hand_results(collected_motor_values, fist_type)
            
            if self.quit_flag:
                break
        
        cv2.destroyWindow("Closed Hand Validation")
    
    def _analyze_closed_hand_results(self, collected_values, fist_type):
        """Closed Hand 검증 결과 분석"""
        print(f"\n{'─' * 50}")
        print(f"[{fist_type.upper()}] 검증 결과 분석")
        
        all_joint_angles = np.array([v["joint_angles"] for v in collected_values])
        avg_angles = np.mean(all_joint_angles, axis=0)
        std_angles = np.std(all_joint_angles, axis=0)
        
        print(f"\n수집된 데이터: {len(collected_values)}회")
        print(f"\n각 손가락별 관절 각도 (평균 ± 표준편차):")
        
        for i, name in enumerate(FINGER_NAMES):
            if i < len(avg_angles):
                angles = avg_angles[i]
                stds = std_angles[i]
                print(f"  {name}:")
                print(f"    MCP: {angles[0]:.1f}° ± {stds[0]:.1f}°")
                print(f"    PIP: {angles[1]:.1f}° ± {stds[1]:.1f}°")
                print(f"    DIP: {angles[2]:.1f}° ± {stds[2]:.1f}°")
        
        # Curl Limits 비교
        print(f"\n[Curl Limits 비교]")
        print(f"  Curl Limits: {self.curl_limits}")
        print(f"  (주먹을 쥐었을 때 모터가 이 값 근처에 있어야 함)")
        
        # Tendon 특성 분석
        print(f"\n[Tendon Driven 특성 분석]")
        print(f"  - DIP 모터 (4, 6, 9, 11): curl_limits에 가까워야 함")
        print(f"  - MCP/PIP 모터 (5, 7, 8, 10): tension_limits에 가까워야 함")
        
        # 결과 저장
        key = f"closed_hand_{fist_type}"
        self.validation_results[key] = {
            "collected_values": collected_values,
            "avg_angles": avg_angles,
            "std_angles": std_angles,
            "curl_limits": self.curl_limits,
        }
    
    def interactive_validation(self):
        """
        대화형 검증 모드
        
        실시간으로 손 추적하면서 키보드 입력으로
        현재 모터 값을 확인하고 검증
        """
        print("\n" + "=" * 70)
        print("대화형 검증 모드 (Interactive Validation)")
        print("=" * 70)
        print("\n조작 방법:")
        print("  'SPACE': 현재 상태 캡처 및 분석")
        print("  'r': RUKA 로봇 실행 (현재 포즈 적용)")
        print("  'o': Open Hand 기준값과 비교")
        print("  'c': Closed Hand 기준값과 비교")
        print("  's': 현재 결과 저장")
        print("  'q': 종료")
        
        captured_data = []
        
        while not self.quit_flag:
            self.timer.start_loop()
            
            success, img = self.cap.read()
            if not success:
                continue
            
            pose_result = self.detect_hand_pose(img)
            
            # 현재 상태 표시
            current_data = None
            transformed_coords = None
            if pose_result["detected"]:
                for hand in pose_result["hands"]:
                    if hand["type"] == self.hand_type:
                        # ✅ webcam_teleoperator 방식으로 변환
                        finger_keypoints = self._mediapipe_to_finger_keypoints(hand["lmList"])
                        finger_keypoints = finger_keypoints / 1000.0
                        transformed_coords, _ = self.transform_keypoints(finger_keypoints, self.hand_type)
                        motor_data = self._calculate_motor_positions(transformed_coords)
                        current_data = motor_data
                        
                        # 손가락 상태 표시
                        fingers = hand["fingers_up"][:5] if len(hand["fingers_up"]) >= 5 else [0]*5
                        status = f"Fingers: {sum(fingers)}/5"
                        
                        # 관절 각도 표시 (간략)
                        if motor_data and "joint_angles" in motor_data:
                            angles = motor_data["joint_angles"]
                            if len(angles) > 0:
                                mcp_avg = np.mean([a[0] for a in angles])
                                status += f" | MCP avg: {mcp_avg:.1f}°"
                        
                        cv2.putText(img, status, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 조작 안내
            cv2.putText(img, "SPACE: Capture | r: Robot | o/c: Compare | q: Quit", 
                       (10, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Interactive Validation", pose_result["img"])
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.quit_flag = True
                break
            
            elif key == ord(' '):  # SPACE - 캡처
                if current_data:
                    captured_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "data": current_data,
                    })
                    print(f"\n[캡처 #{len(captured_data)}]")
                    print(f"  관절 각도:")
                    for i, name in enumerate(FINGER_NAMES):
                        if i < len(current_data["joint_angles"]):
                            angles = current_data["joint_angles"][i]
                            print(f"    {name}: MCP={angles[0]:.1f}°, PIP={angles[1]:.1f}°, DIP={angles[2]:.1f}°")
                else:
                    print("\n[WARNING] 손이 감지되지 않음")
            
            elif key == ord('r'):  # Robot 실행
                if current_data and self.robot and transformed_coords is not None:
                    print("\n[INFO] RUKA에 현재 포즈 적용 중...")
                    try:
                        self.robot.step(transformed_coords)
                        print("  ✓ 적용 완료")
                    except Exception as e:
                        print(f"  ✗ 오류: {e}")
                elif not self.robot:
                    print("\n[WARNING] 로봇이 연결되지 않음 (시뮬레이션 모드)")
            
            elif key == ord('o'):  # Open Hand 비교
                if current_data:
                    self._compare_with_limits(current_data, "open")
            
            elif key == ord('c'):  # Closed Hand 비교
                if current_data:
                    self._compare_with_limits(current_data, "closed")
            
            elif key == ord('s'):  # 저장
                if captured_data:
                    self._save_captured_data(captured_data)
            
            self.timer.end_loop()
        
        cv2.destroyWindow("Interactive Validation")
        
        if captured_data:
            self.validation_results["interactive"] = captured_data
    
    def _compare_with_limits(self, current_data, pose_type):
        """현재 데이터를 limit 값과 비교"""
        print(f"\n[{pose_type.upper()} 기준값 비교]")
        
        if pose_type == "open":
            reference = self.tension_limits
            desc = "Tension Limits"
        else:
            reference = self.curl_limits
            desc = "Curl Limits"
        
        print(f"  {desc}: {reference}")
        print(f"  현재 관절 각도:")
        
        for i, name in enumerate(FINGER_NAMES):
            if i < len(current_data["joint_angles"]):
                angles = current_data["joint_angles"][i]
                print(f"    {name}: MCP={angles[0]:.1f}°, PIP={angles[1]:.1f}°, DIP={angles[2]:.1f}°")
    
    def _save_captured_data(self, captured_data):
        """캡처된 데이터 저장"""
        filename = f"validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        try:
            np.save(filename, captured_data)
            print(f"\n✓ 데이터 저장됨: {filename}")
        except Exception as e:
            print(f"\n✗ 저장 실패: {e}")
    
    def _pause_menu(self):
        """일시정지 메뉴"""
        print("\n[일시정지]")
        print("  r: 계속")
        print("  q: 종료")
        
        while True:
            key = input("선택: ").strip().lower()
            if key == 'r':
                break
            elif key == 'q':
                self.quit_flag = True
                break
    
    # =========================================================================
    # 메인 실행
    # =========================================================================
    
    def run_full_validation(self):
        """전체 검증 실행"""
        print("\n" + "=" * 70)
        print("전체 검증 프로세스 시작")
        print("=" * 70)
        
        menu = """
검증 메뉴:
  1. Open Hand (손 펴기) 검증
  2. Closed Hand (주먹 쥐기) 검증
  3. 대화형 검증 모드
  4. 전체 검증 (1→2→3)
  5. 결과 리포트 생성
  q. 종료
"""
        
        while not self.quit_flag:
            print(menu)
            choice = input("선택: ").strip()
            
            if choice == '1':
                self.validate_open_hand()
            elif choice == '2':
                self.validate_closed_hand()
            elif choice == '3':
                self.interactive_validation()
            elif choice == '4':
                self.validate_open_hand()
                if not self.quit_flag:
                    self.validate_closed_hand()
                if not self.quit_flag:
                    self.interactive_validation()
            elif choice == '5':
                self._generate_report()
            elif choice.lower() == 'q':
                break
        
        self.cleanup()
    
    def _generate_report(self):
        """검증 결과 리포트 생성"""
        print("\n" + "=" * 70)
        print("검증 결과 리포트")
        print("=" * 70)
        print(f"생성 시간: {datetime.now().isoformat()}")
        print(f"대상 손: {self.hand_type}")
        
        for key, data in self.validation_results.items():
            if data:
                print(f"\n[{key}]")
                if isinstance(data, dict):
                    if "avg_angles" in data:
                        print(f"  평균 관절 각도 수집됨")
                        print(f"  샘플 수: {len(data.get('collected_values', []))}")
                elif isinstance(data, list):
                    print(f"  캡처 횟수: {len(data)}")
        
        # 파일로 저장
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("RUKA Gesture-Motor Mapping Validation Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Hand Type: {self.hand_type}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                f.write("Calibration Limits:\n")
                f.write(f"  Tension: {self.tension_limits.tolist()}\n")
                f.write(f"  Curl: {self.curl_limits.tolist()}\n\n")
                
                for key, data in self.validation_results.items():
                    f.write(f"\n[{key}]\n")
                    if isinstance(data, dict) and "avg_angles" in data:
                        f.write(f"  Average angles: {data['avg_angles'].tolist()}\n")
            
            print(f"\n✓ 리포트 저장됨: {report_file}")
        except Exception as e:
            print(f"\n✗ 리포트 저장 실패: {e}")
    
    def cleanup(self):
        """리소스 정리"""
        print("\n[INFO] 리소스 정리 중...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("  ✓ 정리 완료")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RUKA Gesture-Motor Mapping Validator")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("-ht", "--hand_type", type=str, default="right", 
                       choices=["left", "right"], help="Hand type")
    parser.add_argument("-s", "--simulation", action="store_true", 
                       help="Run in simulation mode (no robot)")
    parser.add_argument("-f", "--frequency", type=int, default=30, help="Control frequency")
    
    args = parser.parse_args()
    
    validator = GestureMappingValidator(
        camera_id=args.camera,
        hand_type=args.hand_type,
        frequency=args.frequency,
        simulation_mode=args.simulation,
    )
    
    validator.run_full_validation()


if __name__ == "__main__":
    main()