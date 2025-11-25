#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebCam Teleoperator - MediaPipe 기반 RUKA 로봇 손 원격조정 (최적화 버전)

이 모듈은 웹캠과 MediaPipe를 사용하여 RUKA 로봇 손을 실시간으로 제어합니다.

주요 최적화:
1. MediaPipe Lite 모델 사용 (modelComplexity=0)
2. 낮은 해상도로 빠른 처리 (640x480)
3. 개선된 타이머 (sleep 기반)
4. 프레임 스킵 지원
5. 성능 모니터링

작성자: 이동준
버전: 2.0 (최적화)
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

from copy import deepcopy as copy
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
from collections import deque

from HandTrackingModule import HandDetector
from ruka_hand.control.operator import RUKAOperator
from ruka_hand.utils.constants import *
from ruka_hand.utils.vectorops import *

# =============================================================================
# 최적화된 타이머 클래스
# =============================================================================

class OptimizedTimer:
    """CPU 효율적인 주파수 타이머"""
    
    def __init__(self, frequency):
        self.target_period = 1.0 / frequency
        self.start_time = 0
        self._fps_history = deque(maxlen=30)
    
    def start_loop(self):
        self.start_time = time.perf_counter()
    
    def end_loop(self):
        elapsed = time.perf_counter() - self.start_time
        self._fps_history.append(1.0 / elapsed if elapsed > 0 else 0)
        
        sleep_time = self.target_period - elapsed
        
        if sleep_time > 0.001:  # 1ms 이상일 때만 sleep
            time.sleep(sleep_time * 0.9)  # 90% sleep
            # 나머지 정밀 대기
            while time.perf_counter() - self.start_time < self.target_period:
                pass
    
    @property
    def actual_fps(self):
        """실제 FPS 반환"""
        if self._fps_history:
            return sum(self._fps_history) / len(self._fps_history)
        return 0
    
    @property
    def loop_time_ms(self):
        """루프 시간 (ms)"""
        return (time.perf_counter() - self.start_time) * 1000

# =============================================================================
# MediaPipe 랜드마크 매핑
# =============================================================================

MEDIAPIPE_FINGER_INDICES = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

# =============================================================================
# WebCamTeleoperator 클래스 (최적화)
# =============================================================================

class WebCamTeleoperator:
    """
    웹캠 기반 RUKA 로봇 손 원격조정 클래스 (최적화 버전)
    
    최적화 포인트:
    - MediaPipe Lite 모델 (modelComplexity=0)
    - 640x480 해상도
    - 프레임 버퍼 최소화
    - CPU 효율적 타이머
    - 프레임 스킵 지원
    """

    def __init__(
        self,
        camera_id=0,
        frequency=20,                   # 🔧 30→20 (현실적 목표)
        moving_average_limit=5,         # 🔧 10→5 (지연 감소)
        hands=["left", "right"],
        detection_confidence=0.5,       # 🔧 0.7→0.5 (속도 향상)
        tracking_confidence=0.5,        # 🔧 0.7→0.5
        debug=False,
        # 🆕 최적화 파라미터
        resolution=(640, 480),          # 🔧 저해상도
        model_complexity=0,             # 🔧 Lite 모델
        skip_frames=0,                  # 🆕 프레임 스킵 (0=없음)
    ):
        """
        WebCamTeleoperator 초기화 (최적화 버전)
        
        Parameters:
        -----------
        camera_id : int
            웹캠 ID (기본값: 0)
        frequency : int
            제어 주파수 (Hz) - 20Hz 권장
        moving_average_limit : int
            이동평균 필터 크기 - 5 권장
        hands : list
            제어할 손 ["left", "right"]
        detection_confidence : float
            손 검출 신뢰도 - 0.5 권장 (속도↑)
        tracking_confidence : float
            손 추적 신뢰도 - 0.5 권장 (속도↑)
        debug : bool
            디버그 모드 (성능 저하 주의!)
        resolution : tuple
            웹캠 해상도 - (640, 480) 권장
        model_complexity : int
            MediaPipe 모델 복잡도 (0=Lite, 1=Full)
        skip_frames : int
            프레임 스킵 수 (0=스킵 없음)
        """
        
        self.debug = debug
        self.frequency = frequency
        self.skip_frames = skip_frames
        self.frame_counter = 0
        
        # 🔧 최적화된 타이머
        self.timer = OptimizedTimer(frequency)
        
        # 🔧 웹캠 최적화 설정
        print(f"[INFO] 웹캠 초기화 중... (카메라 ID: {camera_id})")
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"웹캠 ID {camera_id}를 열 수 없습니다!")
        
        # 🔧 해상도 설정 (낮을수록 빠름)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        # 🆕 버퍼 최소화 (지연 감소)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 실제 적용된 설정 확인
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"  ✓ 웹캠 해상도: {actual_width}x{actual_height}")
        print(f"  ✓ 웹캠 FPS: {actual_fps}")
        
        # 🔧 MediaPipe Lite 모델 사용
        self.detector = HandDetector(
            staticMode=False,
            maxHands=2,
            modelComplexity=model_complexity,  # 🔧 0=Lite (빠름!)
            detectionCon=detection_confidence,
            minTrackCon=tracking_confidence
        )
        print(f"  ✓ MediaPipe (complexity={model_complexity}) 초기화 완료")
        
        # 이동평균 필터
        self.moving_average_limit = moving_average_limit
        self.coord_moving_average_queues = {"left": [], "right": []}
        
        # 제어할 손
        self.hand_names = hands
        self.hands = {}
        
        # 성능 모니터링
        self._timing_stats = {
            'webcam': deque(maxlen=30),
            'mediapipe': deque(maxlen=30),
            'transform': deque(maxlen=30),
            'robot': deque(maxlen=30),
        }
        
        print("=" * 60)
        print("WebCam Teleoperator 초기화 완료 (최적화 버전)")
        print(f"  - 해상도: {resolution[0]}x{resolution[1]}")
        print(f"  - 모델: {'Lite' if model_complexity == 0 else 'Full'}")
        print(f"  - 목표 주파수: {frequency} Hz")
        print(f"  - 이동평균: {moving_average_limit}")
        print(f"  - 프레임 스킵: {skip_frames}")
        print("=" * 60)

    def _init_hands(self):
        """RUKAOperator 초기화"""
        print("\n[INFO] 로봇 손 초기화 중...")
        
        for hand_name in self.hand_names:
            try:
                self.hands[hand_name] = RUKAOperator(
                    hand_type=hand_name,
                    moving_average_limit=3,  # 🔧 5→3 (반응 속도↑)
                )
                print(f"  ✓ {hand_name.upper()} 로봇 손 초기화 완료")
            except Exception as e:
                print(f"  ✗ {hand_name.upper()} 초기화 실패: {e}")
        
        print("=" * 60)

    def _mediapipe_to_finger_keypoints(self, lmList):
        """MediaPipe 21개 랜드마크를 (5, 5, 3) 형태로 변환"""
        keypoints = np.zeros((5, 5, 3))
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        for finger_idx, finger_name in enumerate(finger_names):
            indices = MEDIAPIPE_FINGER_INDICES[finger_name]
            for joint_idx, mp_idx in enumerate(indices):
                if mp_idx < len(lmList):
                    keypoints[finger_idx, joint_idx] = lmList[mp_idx][:3]
        
        return keypoints

    def _translate_coords(self, hand_coords):
        """손목을 원점으로 하는 상대 좌표계로 변환"""
        wrist = hand_coords[0, 0]
        translated = copy(hand_coords)
        for finger_idx in range(5):
            translated[finger_idx] = translated[finger_idx] - wrist
        return translated

    def _get_hand_dir_frame(self, origin_coord, index_knuckle_coord, pinky_knuckle_coord, hand_name):
        """손 방향 프레임 계산"""
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
        """키포인트 좌표계 변환"""
        translated_coords = self._translate_coords(hand_coords)
        
        wrist_pos = hand_coords[0, 0]
        index_knuckle = translated_coords[1, 1]
        pinky_knuckle = translated_coords[4, 1]
        
        hand_dir_frame = self._get_hand_dir_frame(
            wrist_pos, index_knuckle, pinky_knuckle, hand_name
        )
        
        transformation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rotation_matrix = np.array(hand_dir_frame[1:])
        transformed_rotation_matrix = transformation_matrix @ rotation_matrix
        
        projected_coords = np.zeros_like(translated_coords)
        for finger_idx in range(5):
            projected_coords[finger_idx] = (
                translated_coords[finger_idx] @ transformed_rotation_matrix.T
            )
        
        projected_coords = projected_coords * 100.0
        
        return projected_coords, hand_dir_frame

    def _operate_hand(self, hand_name, transformed_hand_coords):
        """로봇 손 제어"""
        if hand_name not in self.hands:
            return
        
        try:
            t_start = time.perf_counter()
            
            # 이동평균 필터 적용
            transformed_hand_coords = moving_average(
                transformed_hand_coords,
                self.coord_moving_average_queues[hand_name],
                self.moving_average_limit,
            )
            
            # 로봇 제어 명령
            self.hands[hand_name].step(transformed_hand_coords)
            
            # 타이밍 기록
            self._timing_stats['robot'].append(
                (time.perf_counter() - t_start) * 1000
            )
            
        except Exception as e:
            if self.debug:
                print(f"[WARNING] {hand_name} 손 처리 실패: {e}")

    def _process_frame(self, img):
        """프레임 처리 및 손 검출"""
        t_start = time.perf_counter()
        
        # MediaPipe로 손 검출
        hands, img = self.detector.findHands(img, draw=True, flipType=True)
        
        # MediaPipe 타이밍 기록
        self._timing_stats['mediapipe'].append(
            (time.perf_counter() - t_start) * 1000
        )
        
        hand_data = {}
        
        if hands:
            for hand in hands:
                mp_hand_type = hand["type"].lower()
                
                # 거울 모드 보정
                if mp_hand_type == "left":
                    hand_type = "right"
                else:
                    hand_type = "left"
                
                lmList = hand["lmList"]
                finger_keypoints = self._mediapipe_to_finger_keypoints(lmList)
                finger_keypoints = finger_keypoints / 1000.0
                
                hand_data[hand_type] = finger_keypoints
        
        return hand_data, img

    def _run_robots(self):
        """메인 제어 루프"""
        # 웹캠 프레임 읽기
        t_webcam = time.perf_counter()
        success, img = self.cap.read()
        self._timing_stats['webcam'].append(
            (time.perf_counter() - t_webcam) * 1000
        )
        
        if not success:
            return None
        
        # 🆕 프레임 스킵 처리
        self.frame_counter += 1
        if self.skip_frames > 0 and self.frame_counter % (self.skip_frames + 1) != 0:
            return img  # 처리 없이 화면만 반환
        
        # 손 검출 및 처리
        hand_data, img = self._process_frame(img)
        
        # 좌표 변환 및 로봇 제어
        t_transform = time.perf_counter()
        
        for hand_name in self.hand_names:
            if hand_name in hand_data:
                transformed_hand_coords, _ = self.transform_keypoints(
                    hand_data[hand_name], hand_name
                )
                self._operate_hand(hand_name, transformed_hand_coords)
        
        self._timing_stats['transform'].append(
            (time.perf_counter() - t_transform) * 1000
        )
        
        return img

    def _draw_stats(self, img):
        """성능 통계 화면 표시"""
        # FPS 표시
        fps = self.timer.actual_fps
        cv2.putText(
            img, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        
        # 루프 시간 표시
        loop_ms = self.timer.loop_time_ms
        cv2.putText(
            img, f"Loop: {loop_ms:.1f}ms", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        # 제어 대상 표시
        status_text = "Hands: " + ", ".join(
            [h.upper() for h in self.hand_names if h in self.hands]
        )
        cv2.putText(
            img, status_text, (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        
        # 상세 타이밍 (디버그 모드)
        if self.debug:
            y_offset = 120
            for name, times in self._timing_stats.items():
                if times:
                    avg_time = sum(times) / len(times)
                    cv2.putText(
                        img, f"{name}: {avg_time:.1f}ms",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                    )
                    y_offset += 25
        
        return img

    def run(self):
        """메인 실행 루프"""
        self._init_hands()
        
        print("\n[INFO] 텔레오퍼레이션 시작")
        print("[INFO] 종료: 'q' 키")
        print("=" * 60)
        
        try:
            while True:
                self.timer.start_loop()
                
                # 로봇 제어
                img = self._run_robots()
                
                if img is not None:
                    # 통계 표시
                    img = self._draw_stats(img)
                    cv2.imshow("WebCam Teleoperator - RUKA Hand (Optimized)", img)
                
                self.timer.end_loop()
                
                # 종료 키
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] 사용자 종료 요청")
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C로 종료")
        
        finally:
            self._cleanup()

    def _cleanup(self):
        """리소스 정리"""
        print("\n[INFO] 리소스 정리 중...")
        
        # 로봇 토크 비활성화
        for hand_name, hand_op in self.hands.items():
            try:
                print(f"  → {hand_name.upper()} 토크 비활성화...")
                if hasattr(hand_op, 'controller') and hasattr(hand_op.controller, 'hand'):
                    hand_op.controller.hand.disable_torque()
                    print(f"    ✓ {hand_name.upper()} 토크 비활성화 완료")
            except Exception as e:
                print(f"    ✗ {hand_name.upper()} 토크 비활성화 실패: {e}")
        
        # 웹캠 해제
        if self.cap.isOpened():
            self.cap.release()
            print("  ✓ 웹캠 해제 완료")
        
        # OpenCV 창 닫기
        cv2.destroyAllWindows()
        print("  ✓ OpenCV 창 닫기 완료")
        
        # 성능 요약 출력
        print("\n[성능 요약]")
        for name, times in self._timing_stats.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"  - {name}: 평균 {avg_time:.1f}ms")
        
        print(f"  - 평균 FPS: {self.timer.actual_fps:.1f}")
        print("\n[INFO] 리소스 정리 완료")
        print("=" * 60)


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """WebCam Teleoperator 실행 (최적화 버전)"""
    
    teleoperator = WebCamTeleoperator(
        camera_id=0,
        frequency=20,                   # 20Hz (현실적 목표)
        moving_average_limit=5,         # 짧은 필터
        hands=["right"],                # 단일 손 (성능↑)
        detection_confidence=0.5,       # 낮은 임계값 (속도↑)
        tracking_confidence=0.5,
        debug=False,                    # 디버그 OFF
        resolution=(640, 480),          # 저해상도
        model_complexity=0,             # Lite 모델
        skip_frames=0,                  # 프레임 스킵 없음
    )
    
    teleoperator.run()


if __name__ == "__main__":
    main()