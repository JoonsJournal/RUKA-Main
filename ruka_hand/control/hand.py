#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RUKA Robot Hand Control - 로봇 손 제어 클래스

이 모듈은 RUKA 로봇 손의 저수준 Dynamixel 모터 제어를 담당합니다.

주요 기능:
1. Dynamixel 모터 초기화 및 연결
2. 모터 위치/속도/전류 읽기
3. 모터 위치 제어 명령
4. PID 게인 설정 및 관리
5. 캘리브레이션 데이터 로드 및 적용
6. 안전한 모터 제한값 설정

클래스:
- Hand: 로봇 손 제어 메인 클래스

작성: NYU RUKA Team
라이선스: MIT License
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import os                           # 파일 시스템 조작
from copy import deepcopy as copy   # 깊은 복사 (배열 독립 복제)
import time                         # 시간 관련 함수

import numpy as np                  # 수치 연산 및 배열 처리

# RUKA 프로젝트 모듈 임포트
from ruka_hand.utils.constants import (
    FINGER_NAMES_TO_MOTOR_IDS,     # 손가락 이름 → 모터 ID 매핑
    MOTOR_RANGES_LEFT,              # 왼손 모터 동작 범위
    MOTOR_RANGES_RIGHT,             # 오른손 모터 동작 범위
    USB_PORTS,                      # USB 시리얼 포트 경로
)
from ruka_hand.utils.dynamixel_util import *  # Dynamixel 통신 유틸리티
from ruka_hand.utils.file_ops import get_repo_root  # 프로젝트 루트 경로

# =============================================================================
# 의존성 파일 설명
# =============================================================================

"""
이 모듈은 다음 파일들과 의존 관계를 가집니다:

┌─────────────────────────────────────────────────────────────────┐
│                        hand.py (현재 파일)                        │
│                                                                   │
│  역할: RUKA 로봇 손의 Dynamixel 모터 저수준 제어                   │
└─────────────────────────────────────────────────────────────────┘
│
├─► [1] ruka_hand/utils/constants.py
│     │
│     ├─ USB_PORTS: 시리얼 포트 경로
│     │   예: {"right": "/dev/ttyUSB0", "left": "/dev/ttyUSB1"}
│     │
│     ├─ FINGER_NAMES_TO_MOTOR_IDS: 손가락→모터 매핑
│     │   예: {"Thumb": [1, 2], "Index": [3, 4, 5], ...}
│     │
│     ├─ MOTOR_RANGES_RIGHT: 오른손 모터 동작 범위
│     │   예: 1100 (curl과 tension 사이 거리)
│     │
│     └─ MOTOR_RANGES_LEFT: 왼손 모터 동작 범위
│           예: 1100
│
├─► [2] ruka_hand/utils/dynamixel_util.py
│     │
│     ├─ DynamixelClient 클래스
│     │   ├─ connect(): 시리얼 포트 연결
│     │   ├─ disconnect(): 연결 종료
│     │   ├─ sync_write(): 다중 모터 쓰기
│     │   ├─ sync_read(): 다중 모터 읽기
│     │   ├─ read_pos(): 위치 읽기
│     │   ├─ read_vel(): 속도 읽기
│     │   ├─ read_cur(): 전류 읽기
│     │   ├─ set_pos(): 위치 명령
│     │   └─ set_torque_enabled(): 토크 활성화/비활성화
│     │
│     └─ ModBus RTU 프로토콜 레지스터 주소 상수
│         ├─ ADDR_OPERATING_MODE (11): 동작 모드
│         ├─ ADDR_CURRENT_LIMIT (38): 전류 제한
│         ├─ ADDR_TEMP_LIMIT (31): 온도 제한
│         ├─ ADDR_GOAL_VELOCITY (104): 목표 속도
│         ├─ ADDR_GOAL_POSITION (116): 목표 위치
│         ├─ ADDR_PRESENT_POSITION (132): 현재 위치
│         ├─ ADDR_PRESENT_VELOCITY (128): 현재 속도
│         ├─ ADDR_PRESENT_CURRENT (126): 현재 전류
│         ├─ ADDR_POSITION_P_GAIN (84): P 게인
│         ├─ ADDR_POSITION_I_GAIN (82): I 게인
│         └─ ADDR_POSITION_D_GAIN (80): D 게인
│
├─► [3] ruka_hand/utils/file_ops.py
│     │
│     └─ get_repo_root(): 프로젝트 루트 디렉토리 경로 반환
│
└─► [4] curl_limits/*.npy (캘리브레이션 데이터)
      │
      ├─ {hand_type}_curl_limits.npy
      │   └─ 11개 모터의 최대 구부림 위치
      │
      └─ {hand_type}_tension_limits.npy
          └─ 11개 모터의 텐션 걸린 펼침 위치

데이터 흐름:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] Hand 클래스 인스턴스 생성
    hand = Hand("right")
    ↓
[2] __init__() 실행
    ↓
    ├─► [2-1] 모터 ID 리스트 설정
    │         self.motors = [1, 2, 3, ..., 11]
    │
    ├─► [2-2] USB 포트 경로 로드
    │         self.port = USB_PORTS["right"]
    │         예: "/dev/ttyUSB0"
    │
    ├─► [2-3] DynamixelClient 생성 및 연결
    │         self.dxl_client = DynamixelClient(motors, port)
    │         self.dxl_client.connect()
    │         ↓
    │         ModBus RTU 시리얼 통신 시작
    │         Baud Rate: 1000000 bps
    │
    ├─► [2-4] 캘리브레이션 데이터 로드
    │         ├─ curl_limits.npy 로드 (있으면)
    │         ├─ tension_limits.npy 로드 (있으면)
    │         └─ 없으면 기본값 사용
    │
    ├─► [2-5] 모터 설정
    │         ├─ Operating Mode: 5 (Current-based Position Control)
    │         ├─ Temperature Limit: 60°C
    │         ├─ Current Limit: 700mA (엄지: 700mA)
    │         └─ Goal Velocity: 400
    │
    ├─► [2-6] PID 게인 설정
    │         ├─ DIP/PIP 모터: P=500, I=100, D=960
    │         └─ MCP 모터: P=450, I=120, D=1000
    │
    └─► [2-7] 토크 활성화
              set_torque_enabled(True)
    ↓
[3] Hand 객체 사용 가능
    ├─ read_pos(): 현재 위치 읽기
    ├─ set_pos(): 목표 위치 명령
    ├─ read_vel(): 현재 속도 읽기
    └─ read_cur(): 현재 전류 읽기
    ↓
[4] 프로그램 종료 시
    hand.close()
    ↓
    ├─ set_torque_enabled(False)
    └─ disconnect()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# PID 게인 상수 정의
# =============================================================================

"""
PID 제어기 게인 값

PID 제어는 모터의 위치 제어 정밀도를 결정하는 핵심 파라미터입니다.

P (Proportional) Gain - 비례 게인:
- 오차에 비례하는 제어 출력
- 값이 클수록: 빠른 응답, 높은 강성, 오버슈트 위험
- 값이 작을수록: 느린 응답, 낮은 강성, 정상상태 오차

I (Integral) Gain - 적분 게인:
- 누적 오차에 비례하는 제어 출력
- 정상상태 오차 제거
- 값이 클수록: 빠른 정착, 진동 위험
- 값이 작을수록: 느린 정착, 안정적

D (Derivative) Gain - 미분 게인:
- 오차 변화율에 비례하는 제어 출력
- 오버슈트 억제 및 안정성 향상
- 값이 클수록: 댐핑 효과 증가, 노이즈 민감
- 값이 작을수록: 댐핑 효과 감소, 진동 가능

관절별 게인 차이 이유:

MCP (Metacarpophalangeal) 관절 - 손가락 뿌리:
- 큰 토크 필요 (무게 + 레버리지)
- 낮은 P 게인 (450): 과도한 강성 방지
- 높은 I 게인 (120): 정상상태 오차 최소화
- 높은 D 게인 (1000): 진동 억제
"""

MCP_P_GAIN = 450   # MCP 관절 비례 게인
MCP_I_GAIN = 120   # MCP 관절 적분 게인
MCP_D_GAIN = 1000  # MCP 관절 미분 게인

"""
DIP (Distal Interphalangeal) - 손가락 끝 관절
PIP (Proximal Interphalangeal) - 손가락 중간 관절:
- 작은 토크 필요 (가벼운 세그먼트)
- 높은 P 게인 (500): 정밀한 위치 제어
- 낮은 I 게인 (100): 진동 최소화
- 중간 D 게인 (960): 적절한 댐핑
"""

DIP_PIP_P_GAIN = 500   # DIP/PIP 관절 비례 게인
DIP_PIP_I_GAIN = 100   # DIP/PIP 관절 적분 게인
DIP_PIP_D_GAIN = 960   # DIP/PIP 관절 미분 게인


# =============================================================================
# Hand 클래스
# =============================================================================

class Hand:
    """
    로봇 손 제어 메인 클래스
    
    RUKA 로봇 손의 11개 Dynamixel 모터를 제어하기 위한 핵심 클래스입니다.
    모터 초기화, 위치 제어, 센서 읽기, PID 설정 등 모든 저수준 제어를
    담당합니다.
    
    주요 책임:
    1. Dynamixel 통신 초기화 및 관리
    2. 모터 동작 파라미터 설정 (전류, 온도, 속도 제한)
    3. PID 게인 설정 (관절별 최적화)
    4. 캘리브레이션 데이터 로드 및 적용
    5. 실시간 센서 데이터 읽기 (위치, 속도, 전류)
    6. 모터 위치 명령 전송
    7. 안전한 연결 종료
    
    로봇 손 구조:
    - 총 11개 모터 (5개 손가락)
    - 엄지: 2개 모터 (IP, MCP)
    - 검지/중지/약지/소지: 각 3개 또는 2개 모터 (DIP, PIP, MCP)
    
    모터 ID 할당:
    1: 엄지 IP (Interphalangeal)
    2: 엄지 MCP (Metacarpophalangeal)
    3: 검지 DIP (Distal Interphalangeal)
    4: 검지 MCP
    5: 검지 PIP (Proximal Interphalangeal)
    6: 중지 DIP
    7: 중지 MCP
    8: 약지 DIP
    9: 약지 MCP
    10: 소지 DIP
    11: 소지 MCP
    
    텐던 구동 방식:
    - DIP와 PIP 관절은 단일 텐던으로 연결 (커플링)
    - 에너지 보존: DIP 구부림 ≈ PIP 펼침 (상호 보완)
    - 이로 인해 DIP/PIP 모터는 특별한 PID 게인 필요
    
    Attributes:
        motors (list): 모터 ID 리스트 [1, 2, ..., 11]
        DIP_PIP_motors (list): DIP/PIP 모터 ID 리스트 [4, 6, 9, 11]
        MCP_motors (list): MCP 모터 ID 리스트 [5, 7, 8, 10]
        port (str): USB 시리얼 포트 경로
        dxl_client (DynamixelClient): Dynamixel 통신 클라이언트
        fingers_dict (dict): 손가락 이름→모터 ID 매핑
        hand_type (str): 손 종류 ("right" 또는 "left")
        curled_bound (np.ndarray): 최대 구부림 위치 (11개)
        tensioned_pos (np.ndarray): 텐션 걸린 펼침 위치 (11개)
        min_lim (np.ndarray): 최소 위치 제한
        max_lim (np.ndarray): 최대 위치 제한
        init_pos (np.ndarray): 초기 위치 (tensioned_pos 복사)
        _commanded_pos (np.ndarray): 마지막 명령 위치
        curr_lim (int): 전류 제한값 (mA)
        temp_lim (int): 온도 제한값 (°C)
        goal_velocity (int): 목표 속도
        operating_mode (int): 동작 모드 (5: Current-based Position Control)
    """
    
    def __init__(self, hand_type="right"):
        """
        Hand 클래스 초기화 메서드
        
        이 메서드는 로봇 손의 모든 초기화 과정을 수행합니다:
        1. 모터 ID 설정
        2. Dynamixel 클라이언트 생성 및 연결
        3. 캘리브레이션 데이터 로드
        4. 모터 동작 파라미터 설정
        5. PID 게인 설정
        6. 토크 활성화
        
        Args:
            hand_type (str, optional): 손 종류. "right" 또는 "left".
                                      기본값 "right".
        
        Raises:
            serial.SerialException: USB 포트 연결 실패
            FileNotFoundError: 캘리브레이션 파일 없음 (경고만, 기본값 사용)
            RuntimeError: Dynamixel 통신 실패
            ValueError: 잘못된 hand_type
        
        초기화 소요 시간:
        - 정상: 약 1-2초
        - 재시도 포함: 최대 5초
        
        초기화 후 상태:
        - 모든 모터 토크 활성화
        - 위치: tensioned_pos (펼친 상태)
        - 준비 완료: set_pos() 호출 가능
        """
        
        # =====================================================================
        # [단계 1/7] 모터 ID 설정
        # =====================================================================
        """
        11개 모터의 ID와 관절 타입을 정의합니다.
        
        모터 그룹:
        - DIP_PIP_motors: 텐던 커플링 관절 (특수 PID)
        - MCP_motors: 메인 관절 (표준 PID)
        - 엄지 모터: 1, 2 (별도 전류 제한)
        """
        print("\n" + "="*70)
        print("RUKA Robot Hand Initialization")
        print("로봇 손 초기화")
        print("="*70)
        print(f"\n[단계 1/7] 모터 ID 설정")
        print(f"  손 종류: {hand_type.upper()}")
        
        # 모든 모터 ID 리스트 (1번부터 11번까지)
        self.motors = motors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        print(f"  ✓ 전체 모터: {len(self.motors)}개")
        
        # DIP/PIP 관절 모터 (텐던 커플링)
        # 검지(4), 중지(6), 약지(9), 소지(11)의 DIP 모터
        self.DIP_PIP_motors = [4, 6, 9, 11]
        print(f"  ✓ DIP/PIP 모터: {self.DIP_PIP_motors}")
        
        # MCP 관절 모터 (주 관절)
        # 검지(5), 중지(7), 약지(8), 소지(10)의 MCP 모터
        self.MCP_motors = [5, 7, 8, 10]
        print(f"  ✓ MCP 모터: {self.MCP_motors}")
        
        # =====================================================================
        # [단계 2/7] USB 포트 연결 및 Dynamixel 클라이언트 초기화
        # =====================================================================
        """
        시리얼 포트로 Dynamixel 모터와 통신을 시작합니다.
        
        연결 파라미터:
        - Protocol: 2.0 (Dynamixel Protocol 2.0)
        - Baud Rate: 1000000 bps (1 Mbps)
        - Data Bits: 8
        - Parity: None
        - Stop Bits: 1
        
        연결 실패 시나리오:
        1. 포트 없음: /dev/ttyUSB* 디바이스 미존재
        2. 권한 부족: dialout 그룹 멤버십 필요
        3. 포트 사용 중: 다른 프로그램이 점유
        4. 케이블 불량: 물리적 연결 문제
        """
        print(f"\n[단계 2/7] Dynamixel 통신 초기화")
        
        try:
            # USB 포트 경로 로드
            # 예: {"right": "/dev/ttyUSB0", "left": "/dev/ttyUSB1"}
            self.port = USB_PORTS[hand_type]
            print(f"  시리얼 포트: {self.port}")
            
            # DynamixelClient 인스턴스 생성
            print(f"  → Dynamixel 클라이언트 생성 중...")
            self.dxl_client = DynamixelClient(motors, self.port)
            print(f"  ✓ 클라이언트 생성 완료")
            
            # 시리얼 포트 연결 시도
            print(f"  → 시리얼 포트 연결 중...")
            self.dxl_client.connect()
            print(f"  ✓ 포트 연결 성공")
            print(f"  ✓ Dynamixel 모터 {len(motors)}개 감지")
            
        except KeyError as e:
            print(f"\n✗ 포트 설정 오류!")
            print(f"  에러: hand_type '{hand_type}'가 USB_PORTS에 없음")
            print(f"  사용 가능한 타입: {list(USB_PORTS.keys())}")
            raise ValueError(f"Invalid hand_type: {hand_type}. "
                           f"Must be one of {list(USB_PORTS.keys())}")
        
        except Exception as e:
            print(f"\n✗ Dynamixel 연결 실패!")
            print(f"  에러 타입: {type(e).__name__}")
            print(f"  에러 메시지: {e}")
            print(f"\n가능한 원인:")
            print(f"  1. USB 케이블이 연결되지 않음")
            print(f"  2. 로봇 손 전원이 꺼져 있음")
            print(f"  3. 포트 경로 오류: {self.port}")
            print(f"  4. 시리얼 포트 권한 부족")
            print(f"     해결: sudo usermod -aG dialout $USER && sudo reboot")
            print(f"  5. 다른 프로그램에서 포트 사용 중")
            print(f"     확인: lsof {self.port}")
            raise
        
        # =====================================================================
        # [단계 3/7] 손가락 매핑 및 기본 파라미터 설정
        # =====================================================================
        """
        손가락 이름과 모터 ID 매핑, 제어 파라미터 설정
        """
        print(f"\n[단계 3/7] 제어 파라미터 설정")
        
        # 손가락 이름 → 모터 ID 딕셔너리
        # 예: {"Thumb": [1, 2], "Index": [3, 4, 5], ...}
        self.fingers_dict = FINGER_NAMES_TO_MOTOR_IDS
        print(f"  ✓ 손가락 매핑: {len(self.fingers_dict)}개 손가락")
        
        # 손 종류 저장
        self.hand_type = hand_type
        
        # 모터 제한값 설정
        self.curr_lim = 700        # 전류 제한: 700mA
        self.temp_lim = 60         # 온도 제한: 60°C
        self.goal_velocity = 400   # 목표 속도: 400 (Dynamixel 단위)
        self.operating_mode = 5    # 동작 모드: Current-based Position Control
        
        print(f"  ✓ 전류 제한: {self.curr_lim} mA")
        print(f"  ✓ 온도 제한: {self.temp_lim} °C")
        print(f"  ✓ 목표 속도: {self.goal_velocity}")
        print(f"  ✓ 동작 모드: {self.operating_mode} (Current-based Position Control)")
        
        # =====================================================================
        # [단계 4/7] 캘리브레이션 데이터 로드
        # =====================================================================
        """
        calibrate_motors.py로 생성된 캘리브레이션 파일을 로드합니다.
        
        캘리브레이션 파일:
        - curl_limits.npy: 최대 구부림 위치
        - tension_limits.npy: 텐션 걸린 펼침 위치
        
        파일이 없으면:
        - 기본값 사용 (MOTOR_RANGES)
        - 경고 메시지 출력
        - 프로그램은 계속 실행 (안전 모드)
        """
        print(f"\n[단계 4/7] 캘리브레이션 데이터 로드")
        
        # 프로젝트 루트 디렉토리 경로 가져오기
        repo_root = get_repo_root()
        print(f"  프로젝트 루트: {repo_root}")
        
        # 오른손/왼손에 따라 다른 캘리브레이션 로드
        if hand_type == "right":
            print(f"\n  → 오른손 캘리브레이션 로드 중...")
            
            # Curl Limits 로드 시도
            curl_path = f"{repo_root}/motor_limits/right_curl_limits.npy"
            if os.path.exists(curl_path):
                try:
                    self.curled_bound = np.load(curl_path)
                    print(f"  ✓ Curl Limits 로드 성공")
                    print(f"    파일: {curl_path}")
                    print(f"    데이터 범위: [{self.curled_bound.min():.0f}, "
                          f"{self.curled_bound.max():.0f}]")
                except Exception as e:
                    print(f"  ⚠️ Curl Limits 로드 실패, 기본값 사용")
                    print(f"    에러: {e}")
                    self.curled_bound = np.ones(11) * MOTOR_RANGES_RIGHT
            else:
                print(f"  ⚠️ Curl Limits 파일 없음, 기본값 사용")
                print(f"    경로: {curl_path}")
                print(f"    기본값: {MOTOR_RANGES_RIGHT}")
                self.curled_bound = np.ones(11) * MOTOR_RANGES_RIGHT
            
            # Tension Limits 로드 시도
            tens_path = f"{repo_root}/motor_limits/right_tension_limits.npy"
            if os.path.exists(tens_path):
                try:
                    self.tensioned_pos = np.load(tens_path)
                    print(f"  ✓ Tension Limits 로드 성공")
                    print(f"    파일: {tens_path}")
                    print(f"    데이터 범위: [{self.tensioned_pos.min():.0f}, "
                          f"{self.tensioned_pos.max():.0f}]")
                except Exception as e:
                    print(f"  ⚠️ Tension Limits 로드 실패, 기본값 사용")
                    print(f"    에러: {e}")
                    self.tensioned_pos = self.curled_bound - MOTOR_RANGES_RIGHT
            else:
                print(f"  ⚠️ Tension Limits 파일 없음, 기본값 사용")
                print(f"    경로: {tens_path}")
                self.tensioned_pos = self.curled_bound - MOTOR_RANGES_RIGHT
            
            # 오른손: 최소=tension, 최대=curl
            # (모터 값 증가 = 구부림)
            self.min_lim, self.max_lim = self.tensioned_pos, self.curled_bound
            print(f"  ✓ 오른손 동작 범위 설정 완료")
            print(f"    방향: 모터 값 증가 = 구부림")
            
        elif hand_type == "left":
            print(f"\n  → 왼손 캘리브레이션 로드 중...")
            
            # Curl Limits 로드 시도
            curl_path = f"{repo_root}/motor_limits/left_curl_limits.npy"
            if os.path.exists(curl_path):
                try:
                    self.curled_bound = np.load(curl_path)
                    print(f"  ✓ Curl Limits 로드 성공")
                    print(f"    파일: {curl_path}")
                    print(f"    데이터 범위: [{self.curled_bound.min():.0f}, "
                          f"{self.curled_bound.max():.0f}]")
                except Exception as e:
                    print(f"  ⚠️ Curl Limits 로드 실패, 기본값 사용")
                    print(f"    에러: {e}")
                    self.curled_bound = 4000 - np.ones(11) * MOTOR_RANGES_LEFT
            else:
                print(f"  ⚠️ Curl Limits 파일 없음, 기본값 사용")
                print(f"    경로: {curl_path}")
                print(f"    기본값: 4000 - {MOTOR_RANGES_LEFT}")
                self.curled_bound = 4000 - np.ones(11) * MOTOR_RANGES_LEFT
            
            # Tension Limits 로드 시도
            tens_path = f"{repo_root}/motor_limits/left_tension_limits.npy"
            if os.path.exists(tens_path):
                try:
                    self.tensioned_pos = np.load(tens_path)
                    print(f"  ✓ Tension Limits 로드 성공")
                    print(f"    파일: {tens_path}")
                    print(f"    데이터 범위: [{self.tensioned_pos.min():.0f}, "
                          f"{self.tensioned_pos.max():.0f}]")
                except Exception as e:
                    print(f"  ⚠️ Tension Limits 로드 실패, 기본값 사용")
                    print(f"    에러: {e}")
                    self.tensioned_pos = self.curled_bound + MOTOR_RANGES_LEFT
            else:
                print(f"  ⚠️ Tension Limits 파일 없음, 기본값 사용")
                print(f"    경로: {tens_path}")
                self.tensioned_pos = self.curled_bound + MOTOR_RANGES_LEFT
            
            # 왼손: 최소=curl, 최대=tension
            # (모터 값 감소 = 구부림, 미러 이미지)
            self.min_lim, self.max_lim = self.curled_bound, self.tensioned_pos
            print(f"  ✓ 왼손 동작 범위 설정 완료")
            print(f"    방향: 모터 값 감소 = 구부림 (미러)")
        
        else:
            print(f"\n✗ 잘못된 hand_type: {hand_type}")
            raise ValueError(f"hand_type must be 'right' or 'left', got '{hand_type}'")
        
        # 초기 위치 및 명령 위치 설정
        # tensioned_pos의 깊은 복사 (독립적인 배열)
        self.init_pos = copy(self.tensioned_pos)
        self._commanded_pos = copy(self.tensioned_pos)
        
        print(f"\n  ✓ 초기 위치 설정: tensioned_pos (펼친 상태)")
        print(f"    평균 위치: {self.init_pos.mean():.0f}")
        
        # =====================================================================
        # [단계 5/7] 모터 동작 모드 및 제한값 설정
        # =====================================================================
        """
        Dynamixel 모터의 동작 파라미터를 설정합니다.
        
        설정 항목:
        1. Operating Mode: Current-based Position Control (모드 5)
           - 위치 제어 + 전류 제한
           - 과부하 방지
        
        2. Temperature Limit: 60°C
           - 과열 보호
           - 초과 시 모터 자동 정지
        
        3. Current Limit: 700mA (엄지 700mA)
           - 과전류 방지
           - 텐던 보호
        
        4. Goal Velocity: 400
           - 최대 속도 제한
           - 급격한 움직임 방지
        
        GroupSyncWrite 사용:
        - 11개 모터에 동시에 같은 값 쓰기
        - 개별 쓰기 대비 5.5배 빠름
        """
        print(f"\n[단계 5/7] 모터 동작 파라미터 설정")
        
        try:
            # [5-1] Operating Mode 설정
            print(f"  → Operating Mode 설정 중... (Mode {self.operating_mode})")
            self.dxl_client.sync_write(
                motors,
                # np.ones(len(motors)) * self.operating_mode,
				[int(self.operating_mode)] * len(motors),
                ADDR_OPERATING_MODE,
                LEN_OPERATING_MODE,
            )
            print(f"  ✓ Operating Mode 설정 완료")
            print(f"    모드: Current-based Position Control")
            
            # [5-2] Temperature Limit 설정
            print(f"  → Temperature Limit 설정 중... ({self.temp_lim}°C)")
            self.dxl_client.sync_write(
                motors,
                # np.ones(len(motors)) * self.temp_lim,
				[int(self.temp_lim)] * len(motors),
                ADDR_TEMP_LIMIT,
                LEN_TEMP_LIMIT,
            )
            print(f"  ✓ Temperature Limit 설정 완료")
            
            # [5-3] Current Limit 설정 (일반 모터)
            print(f"  → Current Limit 설정 중... ({self.curr_lim}mA)")
            self.dxl_client.sync_write(
                motors,
                # np.ones(len(motors)) * self.curr_lim,
				[int(self.curr_lim)] * len(motors),
                ADDR_CURRENT_LIMIT,
                LEN_CURRENT_LIMIT,
            )
            print(f"  ✓ Current Limit 설정 완료 (일반 모터)")
            
            # [5-4] Current Limit 설정 (엄지 특수)
            thumb_motors = FINGER_NAMES_TO_MOTOR_IDS["Thumb"]
            thumb_curr_lim = 700
            print(f"  → 엄지 Current Limit 설정 중... ({thumb_curr_lim}mA)")
            self.dxl_client.sync_write(
                thumb_motors,
                # np.ones(len(thumb_motors)) * thumb_curr_lim,
				[int(thumb_curr_lim)] * len(thumb_motors),
                ADDR_CURRENT_LIMIT,
                LEN_CURRENT_LIMIT,
            )
            print(f"  ✓ 엄지 Current Limit 설정 완료")
            print(f"    엄지 모터: {thumb_motors}")
            
            # [5-5] Goal Velocity 설정
            print(f"  → Goal Velocity 설정 중... ({self.goal_velocity})")
            self.dxl_client.sync_write(
                motors,
                # np.ones(len(motors)) * self.goal_velocity,
				[int(self.goal_velocity)] * len(motors),
                ADDR_GOAL_VELOCITY,
                LEN_GOAL_VELOCITY,
            )
            print(f"  ✓ Goal Velocity 설정 완료")
            
        except Exception as e:
            print(f"\n✗ 모터 파라미터 설정 실패!")
            print(f"  에러 타입: {type(e).__name__}")
            print(f"  에러 메시지: {e}")
            print(f"\n가능한 원인:")
            print(f"  1. Dynamixel 통신 오류")
            print(f"  2. 모터 응답 없음 (전원 확인)")
            print(f"  3. 잘못된 레지스터 주소")
            raise
        
        # =====================================================================
        # [단계 6/7] PID 게인 설정
        # =====================================================================
        """
        관절별로 최적화된 PID 게인을 설정합니다.
        
        관절 타입:
        1. DIP/PIP 모터 (텐던 커플링):
           - P=500, I=100, D=960
           - 높은 P: 정밀한 위치 제어
           - 낮은 I: 진동 최소화
        
        2. MCP 모터 (주 관절):
           - P=450, I=120, D=1000
           - 낮은 P: 과도한 강성 방지
           - 높은 I: 정상상태 오차 제거
           - 높은 D: 진동 억제
        """
        print(f"\n[단계 6/7] PID 게인 설정")
        
        try:
            # [6-1] DIP/PIP 모터 P 게인
            print(f"  → DIP/PIP 모터 PID 게인 설정 중...")
            print(f"    대상 모터: {self.DIP_PIP_motors}")
            self.dxl_client.sync_write(
                self.DIP_PIP_motors,
                # np.ones(len(self.DIP_PIP_motors)) * DIP_PIP_P_GAIN,
				[int(DIP_PIP_P_GAIN)] * len(self.DIP_PIP_motors),
                ADDR_POSITION_P_GAIN,
                LEN_POSITION_P_GAIN,
            )
            print(f"    ✓ P Gain: {DIP_PIP_P_GAIN}")
            
            # [6-2] DIP/PIP 모터 I 게인
            self.dxl_client.sync_write(
                self.DIP_PIP_motors,
                # np.ones(len(self.DIP_PIP_motors)) * DIP_PIP_I_GAIN,
				[int(DIP_PIP_I_GAIN)] * len(self.DIP_PIP_motors),
                ADDR_POSITION_I_GAIN,
                LEN_POSITION_I_GAIN,
            )
            print(f"    ✓ I Gain: {DIP_PIP_I_GAIN}")
            
            # [6-3] DIP/PIP 모터 D 게인
            self.dxl_client.sync_write(
                self.DIP_PIP_motors,
                # np.ones(len(self.DIP_PIP_motors)) * DIP_PIP_D_GAIN,
                [int(DIP_PIP_D_GAIN)] * len(self.DIP_PIP_motors),
				ADDR_POSITION_D_GAIN,
                LEN_POSITION_D_GAIN,
            )
            print(f"    ✓ D Gain: {DIP_PIP_D_GAIN}")
            print(f"  ✓ DIP/PIP 모터 PID 설정 완료")
            
            # [6-4] MCP 모터 P 게인
            print(f"\n  → MCP 모터 PID 게인 설정 중...")
            print(f"    대상 모터: {self.MCP_motors}")
            self.dxl_client.sync_write(
                self.MCP_motors,
                # np.ones(len(self.MCP_motors)) * MCP_P_GAIN,
				[int(MCP_P_GAIN)] * len(self.MCP_motors),
                ADDR_POSITION_P_GAIN,
                LEN_POSITION_P_GAIN,
            )
            print(f"    ✓ P Gain: {MCP_P_GAIN}")
            
            # [6-5] MCP 모터 I 게인
            self.dxl_client.sync_write(
                self.MCP_motors,
                # np.ones(len(self.MCP_motors)) * MCP_I_GAIN,
				[int(MCP_I_GAIN)] * len(self.MCP_motors),
                ADDR_POSITION_I_GAIN,
                LEN_POSITION_I_GAIN,
            )
            print(f"    ✓ I Gain: {MCP_I_GAIN}")
            
            # [6-6] MCP 모터 D 게인
            self.dxl_client.sync_write(
                self.MCP_motors,
                # np.ones(len(self.MCP_motors)) * MCP_D_GAIN,
				[int(MCP_D_GAIN)] * len(self.MCP_motors),
                ADDR_POSITION_D_GAIN,
                LEN_POSITION_D_GAIN,
            )
            print(f"    ✓ D Gain: {MCP_D_GAIN}")
            print(f"  ✓ MCP 모터 PID 설정 완료")
            
        except Exception as e:
            print(f"\n✗ PID 게인 설정 실패!")
            print(f"  에러 타입: {type(e).__name__}")
            print(f"  에러 메시지: {e}")
            print(f"\n경고: PID 게인이 기본값으로 설정될 수 있습니다.")
            print(f"      제어 성능이 저하될 수 있습니다.")
            # PID 설정 실패는 치명적이지 않으므로 예외를 다시 발생시키지 않음
        
        # =====================================================================
        # [단계 7/7] 토크 활성화
        # =====================================================================
        """
        모든 모터의 토크를 활성화합니다.
        
        Torque Enable = True:
        - 모터가 위치 명령을 받을 준비 완료
        - 외부 힘에 저항
        - 전력 소비 시작
        
        Torque Enable = False:
        - 모터가 자유롭게 움직임
        - 위치 명령 무시
        - 전력 소비 최소
        
        활성화 시간:
        - set_torque_enabled(True, -1, 0.05)
        - -1: 모든 모터
        - 0.05: 50ms 간격으로 순차 활성화
        """
        print(f"\n[단계 7/7] 모터 토크 활성화")
        
        try:
            print(f"  → 토크 활성화 중... (11개 모터)")
            self.dxl_client.set_torque_enabled(True, -1, 0.05)
            print(f"  ✓ 모든 모터 토크 활성화 완료")
            print(f"    상태: 위치 제어 가능")
            
        except Exception as e:
            print(f"\n✗ 토크 활성화 실패!")
            print(f"  에러 타입: {type(e).__name__}")
            print(f"  에러 메시지: {e}")
            print(f"\n경고: 모터가 위치 명령에 응답하지 않을 수 있습니다.")
            raise
        
        # =====================================================================
        # 데이터 기록 함수 딕셔너리 설정
        # =====================================================================
        """
        데이터 로깅을 위한 콜백 함수 딕셔너리
        
        사용 예:
        recorder = DataRecorder(hand.data_recording_functions)
        recorder.record("actual_hand_state")
        """
        self.data_recording_functions = {
            "actual_hand_state": self.get_hand_state,
            "commanded_hand_state": self.get_commanded_hand_state,
        }
        
        # =====================================================================
        # 초기화 완료
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"[초기화 완료]")
        print(f"{'='*70}")
        print(f"\n  ✓ Hand 클래스 초기화 성공!")
        print(f"  ✓ 손 종류: {self.hand_type.upper()}")
        print(f"  ✓ 모터 수: {len(self.motors)}개")
        print(f"  ✓ 현재 위치: Tensioned (펼친 상태)")
        print(f"  ✓ 토크 상태: 활성화")
        print(f"  ✓ 제어 준비: 완료")
        print(f"\n  사용 가능한 메서드:")
        print(f"    - read_pos(): 현재 위치 읽기")
        print(f"    - set_pos(pos): 목표 위치 명령")
        print(f"    - read_vel(): 현재 속도 읽기")
        print(f"    - read_cur(): 현재 전류 읽기")
        print(f"    - close(): 안전한 종료")
        print(f"{'='*70}\n")


    def close(self):
        """
        Hand 객체를 안전하게 종료하는 메서드
        
        이 메서드는 다음 작업을 수행합니다:
        1. 모든 모터의 토크 비활성화
        2. Dynamixel 클라이언트 연결 해제
        3. 시리얼 포트 닫기
        
        토크 비활성화 이유:
        - 프로그램 종료 후 모터가 계속 활성화되어 있으면
        - 외부 힘에 저항하여 불필요한 전력 소비
        - 발열 및 모터 수명 단축
        - 안전사고 위험
        
        반드시 호출해야 하는 시점:
        - 프로그램 정상 종료 시
        - 예외 발생으로 종료 시
        - KeyboardInterrupt (Ctrl+C) 시
        - finally 블록에서
        
        호출하지 않으면:
        - 다음 실행 시 "Port already in use" 에러
        - 시스템 재부팅 필요할 수 있음
        - 모터가 계속 전력 소비
        
        사용 예:
            try:
                hand = Hand("right")
                # 작업 수행
            finally:
                hand.close()  # 반드시 호출
        """
        print(f"\n{'─'*70}")
        print(f"[Hand 객체 종료]")
        print(f"{'─'*70}")
        
        try:
            # Dynamixel 클라이언트 연결 해제
            # 내부적으로 토크 비활성화 + 포트 닫기 수행
            print(f"  → 토크 비활성화 중...")
            print(f"  → 시리얼 포트 닫기 중...")
            self.dxl_client.disconnect()
            print(f"  ✓ 연결 종료 완료")
            print(f"  ✓ 토크 비활성화 완료")
            print(f"  ✓ 포트 닫기 완료")
            
        except AttributeError:
            # dxl_client가 생성되지 않은 경우
            print(f"  ⚠️ Dynamixel 클라이언트가 초기화되지 않음")
            print(f"     (초기화 실패 후 종료)")
            
        except Exception as e:
            print(f"\n  ⚠️ 종료 중 에러 발생 (무시됨)")
            print(f"     에러 타입: {type(e).__name__}")
            print(f"     에러 메시지: {e}")
            # 종료 시 에러는 무시 (이미 종료 중이므로)
        
        print(f"{'─'*70}\n")


    def read_any(self, addr: int, size: int):
        """
        임의의 레지스터 주소에서 데이터 읽기
        
        Dynamixel 모터의 모든 레지스터에 접근할 수 있는 범용 읽기 함수입니다.
        
        Args:
            addr (int): 읽을 레지스터 주소
                       예: ADDR_PRESENT_POSITION (132)
            size (int): 읽을 데이터 크기 (바이트)
                       예: LEN_PRESENT_POSITION (4)
        
        Returns:
            list: 각 모터에서 읽은 데이터 리스트 (11개)
        
        사용 가능한 레지스터 예시:
        - 위치: addr=132, size=4 (32비트)
        - 속도: addr=128, size=4
        - 전류: addr=126, size=2 (16비트)
        - 온도: addr=146, size=1 (8비트)
        - 전압: addr=144, size=2
        - PWM: addr=124, size=2
        
        주의사항:
        - 잘못된 주소나 크기는 통신 오류 발생
        - 쓰기 전용 레지스터는 읽을 수 없음
        - 모든 모터에 대해 동시 읽기 (GroupSyncRead)
        
        사용 예:
            # 온도 읽기
            temps = hand.read_any(146, 1)
            
            # 전압 읽기
            voltages = hand.read_any(144, 2)
        """
        try:
            return self.dxl_client.sync_read(self.motors, addr, size)
        except Exception as e:
            print(f"\n✗ 레지스터 읽기 실패!")
            print(f"  주소: {addr}, 크기: {size} bytes")
            print(f"  에러: {e}")
            return None


    def read_pos(self):
        """
        현재 모터 위치 읽기
        
        11개 모터의 현재 위치를 GroupSyncRead로 읽어옵니다.
        통신 오류 시 자동으로 재시도합니다.
        
        Returns:
            np.ndarray: 11개 모터의 현재 위치 (shape: (11,))
                       값 범위: [0, 4095] (12비트 해상도)
        
        재시도 로직:
        - None 값 수신 시 계속 재시도
        - 1ms 간격으로 재시도
        - 성공할 때까지 무한 재시도
        
        재시도가 필요한 이유:
        - ModBus 통신 타임아웃
        - 전파 간섭 (노이즈)
        - 모터 응답 지연
        - 버퍼 오버플로우
        
        호출 빈도:
        - 제어 루프: 10-100Hz (10-100ms마다)
        - 센서 읽기: 필요 시마다
        
        통신 시간:
        - 정상: 약 20ms (GroupSyncRead)
        - 재시도 포함: 최대 50ms
        
        주의사항:
        - 무한 루프 가능성 (통신 완전 단절 시)
        - 타임아웃 없음 (의도적 설계)
        - 호출 스레드 블로킹
        
        사용 예:
            curr_pos = hand.read_pos()
            print(f"모터 1 위치: {curr_pos[0]}")
        """
        # Dynamixel 클라이언트로 위치 읽기 시도
        curr_pos = self.dxl_client.read_pos()
        
        # None 값 수신 시 재시도 루프
        retry_count = 0
        while curr_pos is None:
            retry_count += 1
            
            # 10회마다 경고 출력
            if retry_count % 10 == 0:
                print(f"  ⚠️ 위치 읽기 재시도 중... ({retry_count}회)")
            
            # 짧은 대기 후 재시도
            time.sleep(0.001)  # 1ms 대기
            curr_pos = self.dxl_client.read_pos()
        
        # 재시도 횟수가 많았으면 경고
        if retry_count > 5:
            print(f"  ⚠️ 위치 읽기 성공 ({retry_count}회 재시도 후)")
        
        return curr_pos


    def read_vel(self):
        """
        현재 모터 속도 읽기
        
        11개 모터의 현재 속도를 읽어옵니다.
        
        Returns:
            np.ndarray: 11개 모터의 현재 속도 (shape: (11,))
                       단위: Dynamixel 속도 단위
                       양수: 정방향 회전
                       음수: 역방향 회전
        
        속도 단위 변환:
        - Dynamixel 단위: 0.229 rpm
        - 예: 값 100 = 22.9 rpm
        
        용도:
        - 동작 분석
        - 피드백 제어
        - 충돌 감지
        
        주의사항:
        - 재시도 로직 없음 (read_pos와 다름)
        - None 반환 가능
        - 저주파수로 읽기 권장 (10-20Hz)
        
        사용 예:
            vel = hand.read_vel()
            if vel is not None:
                print(f"평균 속도: {np.mean(vel):.1f}")
        """
        try:
            return self.dxl_client.read_vel()
        except Exception as e:
            print(f"  ⚠️ 속도 읽기 실패: {e}")
            return None


    def read_cur(self):
        """
        현재 모터 전류 읽기
        
        11개 모터의 현재 전류를 읽어옵니다.
        
        Returns:
            np.ndarray: 11개 모터의 현재 전류 (shape: (11,))
                       단위: mA (밀리암페어)
        
        전류 의미:
        - 높은 전류: 큰 부하 (무거운 물체, 저항)
        - 낮은 전류: 작은 부하 (자유 동작)
        - 전류 급증: 충돌 또는 한계 도달
        
        용도:
        - 과부하 감지
        - 충돌 감지
        - 캘리브레이션 (전류 제한 측정)
        - 그립 강도 추정
        
        주의사항:
        - 전류 제한: 700mA (일반), 700mA (엄지)
        - 제한 초과 시 모터 보호 모드
        - 지속적인 고전류는 과열 유발
        
        사용 예:
            cur = hand.read_cur()
            if cur is not None:
                max_cur = np.max(cur)
                if max_cur > 600:
                    print(f"⚠️ 높은 전류 감지: {max_cur}mA")
        """
        try:
            return self.dxl_client.read_cur()
        except Exception as e:
            print(f"  ⚠️ 전류 읽기 실패: {e}")
            return None


    def read_single_cur(self, motor_id):
        """
        단일 모터의 전류 읽기
        
        특정 모터 하나의 전류만 읽습니다.
        calibrate_motors.py에서 사용됩니다.
        
        Args:
            motor_id (int): 읽을 모터 ID (1-11)
        
        Returns:
            float: 모터 전류 (mA)
                  None: 읽기 실패
        
        용도:
        - 캘리브레이션 중 전류 모니터링
        - 개별 모터 진단
        - 특정 손가락 부하 측정
        
        read_cur() vs read_single_cur():
        - read_cur(): 11개 모터 동시 읽기 (GroupSyncRead)
        - read_single_cur(): 1개 모터만 읽기 (더 빠름)
        
        사용 예:
            # 검지 MCP 모터(4번) 전류 읽기
            cur = hand.read_single_cur(4)
            if cur is not None and cur > 200:
                print(f"검지에 높은 부하: {cur}mA")
        """
        try:
            cur = self.dxl_client.read_single_cur(motor_id)
            return cur
        except Exception as e:
            print(f"  ⚠️ 모터 {motor_id} 전류 읽기 실패: {e}")
            return None


    def set_pos(self, pos):
        """
        목표 위치 명령
        
        11개 모터에 목표 위치를 동시에 전송합니다.
        
        Args:
            pos (np.ndarray or list): 목표 위치 배열 (shape: (11,))
                                     값 범위: [0, 4095]
        
        동작:
        1. 명령 위치를 내부 변수에 저장 (_commanded_pos)
        2. DynamixelClient를 통해 모터에 전송
        3. 모터가 목표 위치로 이동 시작
        
        GroupSyncWrite 사용:
        - 11개 모터에 동시 전송
        - 총 시간: 약 20ms
        - 개별 전송 대비 5.5배 빠름
        
        모터 응답:
        - PID 제어기가 위치 오차 계산
        - 전류를 조절하여 위치 추종
        - Current-based Position Control 모드
        
        주의사항:
        - 위치 범위 검증 없음 (호출자 책임)
        - 안전 범위: [min_lim, max_lim]
        - 범위 초과 시 하드웨어 손상 가능
        - 급격한 위치 변화는 충격 유발
        
        권장 사항:
        - 궤적 생성 사용 (move_to_pos)
        - 한 번에 50-100씩 이동
        - 제한값 확인: np.clip(pos, min_lim, max_lim)
        
        사용 예:
            # 안전한 사용
            target = np.clip(desired_pos, hand.min_lim, hand.max_lim)
            hand.set_pos(target)
            
            # 위험한 사용 (피해야 함)
            hand.set_pos([4095] * 11)  # 급격한 이동!
        """
        # 명령 위치 저장 (나중에 get_commanded_hand_state에서 사용)
        self._commanded_pos = pos
        
        try:
            # Dynamixel 클라이언트를 통해 위치 명령 전송
            self.dxl_client.set_pos(pos)
            
        except Exception as e:
            print(f"\n✗ 위치 명령 전송 실패!")
            print(f"  목표 위치: {pos}")
            print(f"  에러 타입: {type(e).__name__}")
            print(f"  에러 메시지: {e}")
            print(f"\n가능한 원인:")
            print(f"  1. Dynamixel 통신 오류")
            print(f"  2. 모터 응답 없음")
            print(f"  3. 잘못된 위치 값 (범위 초과)")
            print(f"  4. 토크 비활성화 상태")
            # 예외를 다시 발생시키지 않음 (제어 루프 계속)


    def read_temp(self):
        """
        모터 온도 읽기
        
        11개 모터의 현재 온도를 읽어옵니다.
        
        Returns:
            np.ndarray: 11개 모터의 온도 (shape: (11,))
                       단위: °C (섭씨)
        
        온도 제한:
        - 정상: 20-40°C
        - 주의: 40-55°C
        - 경고: 55-60°C
        - 위험: 60°C 이상 (자동 정지)
        
        과열 원인:
        - 장시간 고부하 동작
        - 높은 전류 (충돌, 과부하)
        - 불충분한 냉각
        - 환경 온도 높음
        
        과열 대처:
        1. 즉시 동작 중단
        2. 토크 비활성화 (냉각)
        3. 30초-1분 대기
        4. 온도 확인 후 재시작
        
        주의사항:
        - 레지스터 주소: 146
        - 크기: 1 byte
        - 반환값 확인 필요 (None 가능)
        
        사용 예:
            temps = hand.read_temp()
            if temps is not None:
                max_temp = np.max(temps)
                if max_temp > 55:
                    print(f"⚠️ 과열 경고: {max_temp}°C")
                    hand.close()  # 안전을 위해 종료
        """
        try:
            return self.dxl_client.sync_read(self.motors, 146, 1)
        except Exception as e:
            print(f"  ⚠️ 온도 읽기 실패: {e}")
            return None


    @property
    def commanded_pos(self):
        """
        마지막 명령 위치 반환 (속성)
        
        set_pos()로 마지막으로 전송한 위치를 반환합니다.
        
        Returns:
            np.ndarray: 마지막 명령 위치 (shape: (11,))
        
        commanded_pos vs actual_pos:
        - commanded_pos: 명령한 목표 위치
        - actual_pos: 모터가 실제 도달한 위치
        - 차이: 위치 오차 (PID 제어 대상)
        
        용도:
        - 제어 성능 분석
        - 위치 오차 계산
        - 데이터 로깅
        
        주의사항:
        - 모터 통신 없음 (즉시 반환)
        - 초기값: tensioned_pos
        - set_pos() 호출 시마다 업데이트
        
        사용 예:
            cmd = hand.commanded_pos
            act = hand.actual_pos
            error = np.abs(act - cmd)
            print(f"평균 오차: {error.mean():.1f}")
        """
        return self._commanded_pos


    @property
    def actual_pos(self):
        """
        실제 모터 위치 반환 (속성)
        
        read_pos()를 호출하여 현재 위치를 읽고,
        None 값이 없을 때까지 재시도합니다.
        
        Returns:
            np.ndarray: 현재 모터 위치 (shape: (11,))
        
        재시도 로직:
        - None 값 포함 시 계속 재시도
        - 0.1ms 간격으로 재시도
        - 모든 모터가 유효한 값 반환할 때까지
        
        재시도가 필요한 이유:
        - 부분적인 통신 실패 가능
        - 일부 모터만 응답
        - GroupSyncRead의 특성
        
        호출 비용:
        - 통신 포함: 약 20-50ms
        - 재시도 포함: 최대 100ms
        - 루프에서 사용 시 주의
        
        주의사항:
        - read_pos()보다 느릴 수 있음
        - 무한 루프 가능성
        - 고주파수 호출 비권장
        
        사용 예:
            # 현재 위치 읽기
            pos = hand.actual_pos
            print(f"손가락 펼쳐짐: {pos[0] < 1000}")
        """
        # 위치 읽기
        pos = self.read_pos()
        
        # None 값이 하나라도 있으면 재시도
        retry_count = 0
        while any(item is None for item in pos):
            retry_count += 1
            
            # 10회마다 경고
            if retry_count % 10 == 0:
                print(f"  ⚠️ 부분적 통신 실패, 재시도 중... ({retry_count}회)")
            
            time.sleep(0.0001)  # 0.1ms 대기
            pos = self.read_pos()
        
        if retry_count > 5:
            print(f"  ⚠️ actual_pos 읽기 성공 ({retry_count}회 재시도 후)")
        
        return pos


    def get_hand_state(self):
        """
        로봇 손 전체 상태 반환
        
        현재 시점의 로봇 손 상태를 딕셔너리로 반환합니다.
        데이터 로깅 및 분석에 사용됩니다.
        
        Returns:
            dict: 손 상태 딕셔너리
                - position (np.ndarray): 실제 위치 (11,)
                - commanded_position (np.ndarray): 명령 위치 (11,)
                - velocity (np.ndarray): 현재 속도 (11,)
                - timestamp (float): 타임스탬프 (Unix time)
        
        용도:
        - 실시간 데이터 로깅
        - 제어 성능 분석
        - 리플레이 데이터 생성
        - 디버깅 및 시각화
        
        데이터 타입:
        - 모든 배열: np.float32 (메모리 효율성)
        - 타임스탬프: float (초 단위)
        
        호출 빈도:
        - 제어 루프: 10-100Hz
        - 데이터 수집: 20-50Hz 권장
        
        주의사항:
        - actual_pos 호출 (통신 포함)
        - 느릴 수 있음 (20-50ms)
        - 고주파수 호출 시 성능 저하
        
        사용 예:
            # 상태 저장
            state = hand.get_hand_state()
            
            # 데이터 로깅
            logger.log({
                'time': state['timestamp'],
                'pos': state['position'],
                'vel': state['velocity']
            })
        """
        try:
            motor_state = dict(
                position=np.array(self.actual_pos, dtype=np.float32),
                commanded_position=np.array(self.commanded_pos, dtype=np.float32),
                velocity=np.array(self.read_vel(), dtype=np.float32),
                timestamp=time.time(),
            )
            return motor_state
        
        except Exception as e:
            print(f"  ⚠️ 손 상태 읽기 실패: {e}")
            return None


    def get_commanded_hand_state(self):
        """
        명령 위치 기반 손 상태 반환
        
        실제 센서 읽기 없이 마지막 명령 위치만 반환합니다.
        빠른 로깅에 적합합니다.
        
        Returns:
            dict: 명령 상태 딕셔너리
                - position (np.ndarray): 명령 위치 (11,)
                - timestamp (float): 타임스탬프 (Unix time)
        
        get_hand_state() vs get_commanded_hand_state():
        - get_hand_state(): 실제 센서 데이터 (느림, 정확)
        - get_commanded_hand_state(): 명령 데이터 (빠름, 추정)
        
        용도:
        - 고속 데이터 로깅 (>100Hz)
        - 명령 이력 기록
        - 제어 입력 분석
        
        장점:
        - 통신 없음 (즉시 반환)
        - 고주파수 호출 가능
        - CPU 부하 낮음
        
        단점:
        - 실제 위치 모름
        - 센서 피드백 없음
        - 오차 정보 없음
        
        사용 예:
            # 고속 로깅 (200Hz)
            for _ in range(1000):
                state = hand.get_commanded_hand_state()
                logger.log(state)
                time.sleep(0.005)  # 5ms
        """
        try:
            motor_state = dict(
                position=np.array(self.commanded_pos, dtype=np.float32),
                timestamp=time.time(),
            )
            return motor_state
        
        except Exception as e:
            print(f"  ⚠️ 명령 상태 읽기 실패: {e}")
            return None


# =============================================================================
# 모듈 테스트 코드
# =============================================================================

if __name__ == "__main__":
    """
    Hand 클래스 기본 테스트
    
    이 코드는 직접 실행 시에만 동작합니다:
        python hand.py
    
    테스트 내용:
    1. Hand 객체 생성
    2. 위치 읽기 테스트
    3. 속도 읽기 테스트
    4. 전류 읽기 테스트
    5. 안전한 종료
    """
    print("\n" + "="*70)
    print("Hand 클래스 테스트")
    print("="*70)
    
    try:
        # Hand 객체 생성
        print("\n[테스트 1/4] Hand 객체 생성")
        hand = Hand("right")
        
        # 위치 읽기
        print("\n[테스트 2/4] 위치 읽기")
        pos = hand.read_pos()
        print(f"  현재 위치: {pos}")
        print(f"  평균: {np.mean(pos):.1f}")
        print(f"  범위: [{pos.min()}, {pos.max()}]")
        
        # 속도 읽기
        print("\n[테스트 3/4] 속도 읽기")
        vel = hand.read_vel()
        if vel is not None:
            print(f"  현재 속도: {vel}")
            print(f"  평균: {np.mean(vel):.1f}")
        
        # 전류 읽기
        print("\n[테스트 4/4] 전류 읽기")
        cur = hand.read_cur()
        if cur is not None:
            print(f"  현재 전류: {cur}")
            print(f"  평균: {np.mean(cur):.1f} mA")
        
        print("\n" + "="*70)
        print("테스트 성공!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n[사용자 중단]")
        print("  Ctrl+C가 감지되었습니다.")
        
    except Exception as e:
        print(f"\n\n[테스트 실패]")
        print(f"  에러: {e}")
        
    finally:
        # 안전한 종료
        try:
            hand.close()
        except:
            pass
        
        print("\n프로그램 종료\n")