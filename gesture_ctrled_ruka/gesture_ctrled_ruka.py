#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUKA Hand - WebCAM Gesture Control (Direct Port from OYMotion)

OYMotion ROHand의 gesture_ctrled_hand.py를 RUKA Hand용으로 직접 포팅한 코드입니다.
HandTrackingModule.py는 그대로 사용하고, 제어 부분만 RUKA Hand에 맞게 수정했습니다.

주요 변경사항:
1. ModBus RTU → Dynamixel Protocol 2.0
2. ROHand 6개 모터 → RUKA 11개 모터 매핑
3. 모터 값 범위: 0~65535 → 0~4000
4. 제스처 → RUKA 모터 위치 직접 매핑

작성일: 2024-11-18
기반 코드: OYMotion gesture_ctrled_hand.py
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import os
import sys
import cv2
import time
import queue
import threading
import numpy as np

# HandTrackingModule (그대로 사용)
from HandTrackingModule import HandDetector

# RUKA Hand 모듈
from ruka_hand.control.hand import Hand
from ruka_hand.utils.constants import (
    FINGER_NAMES_TO_MOTOR_IDS,
    USB_PORTS,
)

# =============================================================================
# 전역 변수 및 설정
# =============================================================================

# 현재 스크립트 경로 (제스처 이미지 로드용)
file_path = os.path.abspath(os.path.dirname(__file__))

# RUKA Hand 설정
NUM_FINGERS = 5         # 제어할 손가락 개수 (엄지~새끼)
HAND_TYPE = "right"     # "right" 또는 "left"

# =============================================================================
# 스레드 간 통신용 큐
# =============================================================================
gesture_queue = queue.Queue(maxsize=NUM_FINGERS)
image_queue = queue.Queue(maxsize=1)

# =============================================================================
# 카메라 초기화
# =============================================================================
video = cv2.VideoCapture(0)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"카메라 해상도: {width} x {height}")

# =============================================================================
# 손 추적기 초기화
# =============================================================================
detector = HandDetector(maxHands=1, detectionCon=0.8)

# =============================================================================
# OpenCV 윈도우 설정
# =============================================================================
cv2.namedWindow("RUKA Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RUKA Gesture Control", width, height)

# =============================================================================
# 제스처 → RUKA 모터 위치 매핑 함수
# =============================================================================

def gesture_to_motor_positions(finger_states, thumb_rotation):
    """
    MediaPipe 손가락 상태를 RUKA 모터 위치로 변환
    
    Args:
        finger_states: [엄지, 검지, 중지, 약지, 새끼] (0=접힘, 1=펴짐)
        thumb_rotation: 엄지 회전 각도 (0~1)
    
    Returns:
        motor_positions: (11,) numpy array - 11개 모터의 목표 위치
    
    매핑 전략:
    - ROHand: 0(펴짐) ~ 65535(접힘)
    - RUKA: 0(펴짐) ~ 4000(접힘)
    - 변환: ruka_pos = rohand_pos * (4000 / 65535)
    
    모터 배치:
    - 엄지: motors[0, 1] (IP, MCP)
    - 검지: motors[2, 3, 4] (DIP, MCP, PIP)
    - 중지: motors[5, 6] (DIP, MCP)
    - 약지: motors[7, 8] (DIP, MCP)
    - 새끼: motors[9, 10] (DIP, MCP)
    """
    
    # 초기값: 모두 접힌 상태
    motor_positions = np.full(11, 4000, dtype=np.int32)
    
    # ROHand 기준 위치 (0~65535)
    ROHAND_MAX = 65535
    ROHAND_THUMB_MIDDLE = 45000  # 엄지 중간 위치
    
    # RUKA 변환 비율
    SCALE = 4000.0 / ROHAND_MAX
    
    # =========================================================================
    # 엄지 (Thumb) - Motor ID: 1, 2
    # =========================================================================
    thumb_state = finger_states[0]
    
    if thumb_state == 1:  # 펴짐
        # 엄지 회전 각도에 따라 위치 조정
        thumb_pos = int(ROHAND_THUMB_MIDDLE * thumb_rotation * SCALE)
        motor_positions[0] = thumb_pos  # IP
        motor_positions[1] = thumb_pos  # MCP
    else:  # 접힘
        thumb_pos = int(ROHAND_MAX * SCALE)
        motor_positions[0] = thumb_pos
        motor_positions[1] = thumb_pos
    
    # =========================================================================
    # 검지 (Index) - Motor ID: 3, 4, 5
    # =========================================================================
    index_state = finger_states[1]
    
    if index_state == 1:  # 펴짐
        motor_positions[2] = 0  # DIP
        motor_positions[3] = 0  # MCP
        motor_positions[4] = 0  # PIP
    else:  # 접힘
        motor_positions[2] = 4000
        motor_positions[3] = 4000
        motor_positions[4] = 4000
    
    # =========================================================================
    # 중지 (Middle) - Motor ID: 6, 7
    # =========================================================================
    middle_state = finger_states[2]
    
    if middle_state == 1:  # 펴짐
        motor_positions[5] = 0  # DIP
        motor_positions[6] = 0  # MCP
    else:  # 접힘
        motor_positions[5] = 4000
        motor_positions[6] = 4000
    
    # =========================================================================
    # 약지 (Ring) - Motor ID: 8, 9
    # =========================================================================
    ring_state = finger_states[3]
    
    if ring_state == 1:  # 펴짐
        motor_positions[7] = 0  # DIP
        motor_positions[8] = 0  # MCP
    else:  # 접힘
        motor_positions[7] = 4000
        motor_positions[8] = 4000
    
    # =========================================================================
    # 새끼 (Pinky) - Motor ID: 10, 11
    # =========================================================================
    pinky_state = finger_states[4]
    
    if pinky_state == 1:  # 펴짐
        motor_positions[9] = 0   # DIP
        motor_positions[10] = 0  # MCP
    else:  # 접힘
        motor_positions[9] = 4000
        motor_positions[10] = 4000
    
    return motor_positions


# =============================================================================
# 카메라 스레드 함수
# =============================================================================

def camera_thread():
    """
    카메라 영상 처리 및 손 추적 스레드
    
    OYMotion의 camera_thread()와 동일한 구조를 유지하되,
    제스처 → 모터 위치 변환 로직만 RUKA용으로 수정
    """
    
    # 엄지 간섭 방지 변수
    timer = 0
    interval = 10
    original_thumb_state = 0
    prev_finger_states = [0, 0, 0, 0, 0]
    
    while True:
        # 프레임 읽기 및 좌우 반전
        _, img = video.read()
        img = cv2.flip(img, 1)
        
        # 손 검출
        hands = detector.findHands(img, draw=True)
        
        # 제스처 아이콘 로드
        gesture_pic = cv2.imread(file_path + "/gestures/unknown.png")
        
        # 기본 제스처 (모두 접힘)
        finger_states = [0, 0, 0, 0, 0]
        thumb_rotation = 0.0
        
        # 손 감지된 경우
        if hands:
            hand = hands[0]
            
            if hand and hand[0]:
                try:
                    # 손가락 상태 추출
                    finger_up = detector.fingersUp(hand[0])
                    
                    # finger_up = [엄지, 검지, 중지, 약지, 새끼, 엄지회전]
                    finger_states = finger_up[:5]
                    thumb_rotation = finger_up[5]
                    
                    # 제스처 패턴 인식 (OYMotion과 동일)
                    if finger_states == [0, 0, 0, 0, 0]:
                        gesture_pic = cv2.imread(file_path + "/gestures/0.png")
                    elif finger_states == [0, 1, 0, 0, 0]:
                        gesture_pic = cv2.imread(file_path + "/gestures/1.png")
                    elif finger_states == [0, 1, 1, 0, 0]:
                        gesture_pic = cv2.imread(file_path + "/gestures/2.png")
                    elif finger_states == [0, 1, 1, 1, 0]:
                        gesture_pic = cv2.imread(file_path + "/gestures/3.png")
                    elif finger_states == [0, 1, 1, 1, 1]:
                        gesture_pic = cv2.imread(file_path + "/gestures/4.png")
                    elif finger_states == [1, 1, 1, 1, 1]:
                        gesture_pic = cv2.imread(file_path + "/gestures/5.png")
                
                except Exception as e:
                    print(f"손 추적 오류: {e}")
        
        else:
            # 손이 없으면 모두 펴짐
            finger_states = [1, 1, 1, 1, 1]
            thumb_rotation = 1.0
        
        # 제스처 아이콘 오버레이
        if gesture_pic is not None and gesture_pic.size > 0:
            gesture_pic = cv2.resize(gesture_pic, (161, 203))
            img[0:203, 0:161] = gesture_pic
        
        # =====================================================================
        # 엄지 간섭 방지 알고리즘 (OYMotion과 동일)
        # =====================================================================
        if finger_states[0] == 0 and thumb_rotation > 0:
            # 다른 손가락 상태가 변경되었는지 확인
            if prev_finger_states[1:] != finger_states[1:]:
                timer = 0
                prev_finger_states = finger_states.copy()
            
            if timer == 0:
                original_thumb_state = finger_states[0]
            
            timer += 1
            
            if timer <= interval:
                finger_states[0] = 1  # 엄지 임시로 펴기
            else:
                finger_states[0] = original_thumb_state
        else:
            if timer > 0:
                finger_states[0] = original_thumb_state
            timer = 0
        
        # =====================================================================
        # 제스처 → RUKA 모터 위치 변환
        # =====================================================================
        motor_positions = gesture_to_motor_positions(finger_states, thumb_rotation)
        
        # 큐에 데이터 전달
        if not gesture_queue.full():
            gesture_queue.put(motor_positions)
        
        if not image_queue.full():
            image_queue.put(img)


# =============================================================================
# 메인 함수
# =============================================================================

def main():
    """
    메인 제어 함수
    
    OYMotion의 main()과 동일한 구조를 유지하되,
    ModBus → Dynamixel Protocol로 변경
    """
    
    print("\n" + "="*70)
    print("RUKA Hand - WebCAM Gesture Control")
    print("="*70)
    
    # =========================================================================
    # 1단계: RUKA Hand 연결
    # =========================================================================
    print(f"\n[단계 1] RUKA Hand 연결 중...")
    print(f"  손 타입: {HAND_TYPE}")
    
    try:
        hand = Hand(hand_type=HAND_TYPE)
        print("✓ RUKA Hand 연결 성공!")
    except Exception as e:
        print(f"\n✗ RUKA Hand 연결 실패: {e}")
        print("  가능한 원인:")
        print("    1. USB 케이블 연결 확인")
        print("    2. 로봇 손 전원 확인")
        print("    3. USB 포트 권한 확인: sudo usermod -aG dialout $USER")
        return
    
    # =========================================================================
    # 2단계: 변수 초기화
    # =========================================================================
    prev_motor_positions = np.zeros(11, dtype=np.int32)
    last_time = time.time()
    
    # =========================================================================
    # 3단계: 카메라 스레드 시작
    # =========================================================================
    print(f"\n[단계 2] 카메라 스레드 시작...")
    threading.Thread(target=camera_thread, daemon=True).start()
    print("✓ 카메라 스레드 실행 중")
    
    print("\n" + "="*70)
    print("손동작으로 RUKA Hand를 제어하세요!")
    print("종료: 'q' 키를 누르세요")
    print("="*70 + "\n")
    
    # =========================================================================
    # 4단계: 메인 루프
    # =========================================================================
    try:
        while True:
            # 큐에서 모터 위치 수신
            motor_positions = gesture_queue.get()
            
            # 이미지 수신 (논블로킹)
            if not image_queue.empty():
                img = image_queue.get()
                
                # 안내 텍스트
                cv2.putText(
                    img,
                    "Control RUKA with gestures",
                    (16, 272),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3
                )
                
                # 현재 제스처 상태 표시
                status_text = f"Motors: {motor_positions[0]}, {motor_positions[2]}, {motor_positions[5]}, ..."
                cv2.putText(
                    img,
                    status_text[:50],  # 화면에 맞게 자르기
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                cv2.imshow("RUKA Gesture Control", img)
            
            # 제스처 변경 감지
            if not np.array_equal(prev_motor_positions, motor_positions):
                # 엄지 지연 제어 (OYMotion과 동일)
                current_time = time.time()
                
                if current_time - last_time < 0.7:
                    # 0.7초 이내 → 엄지 펴진 상태로
                    motor_positions[0] = 0
                    motor_positions[1] = 0
                else:
                    last_time = current_time
                
                # RUKA Hand에 명령 전송
                try:
                    hand.set_pos(motor_positions)
                    print(f"✓ 모터 위치 업데이트: {motor_positions[:5]}...")
                except Exception as e:
                    print(f"✗ 모터 제어 실패: {e}")
                
                prev_motor_positions = motor_positions.copy()
            
            # 키보드 입력 확인
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' 키가 입력되었습니다. 종료합니다...")
                break
    
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    
    finally:
        # =====================================================================
        # 5단계: 정리 및 종료
        # =====================================================================
        print("\n[종료 처리]")
        
        # 카메라 종료
        print("  - 카메라 종료...")
        video.release()
        
        # OpenCV 윈도우 닫기
        print("  - 윈도우 닫기...")
        cv2.destroyAllWindows()
        
        # RUKA Hand 안전 종료
        print("  - RUKA Hand 연결 종료...")
        try:
            # 펴진 상태로 리셋
            reset_pos = np.zeros(11, dtype=np.int32)
            hand.set_pos(reset_pos)
            time.sleep(0.5)
            hand.close()
        except:
            pass
        
        print("\n" + "="*70)
        print("프로그램이 정상적으로 종료되었습니다.")
        print("="*70 + "\n")


# =============================================================================
# 프로그램 진입점
# =============================================================================

if __name__ == "__main__":
    """
    프로그램 시작점
    
    실행 순서:
    1. main() 함수 호출
    2. RUKA Hand 연결
    3. 카메라 스레드 시작
    4. 제스처 인식 및 제어 루프
    5. 'q' 키로 종료
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            video.release()
            cv2.destroyAllWindows()
        except:
            pass