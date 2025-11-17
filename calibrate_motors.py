#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RUKA Robot Hand Motor Calibration - 로봇 손 모터 캘리브레이션 프로그램 (Windows 버전)

이 프로그램은 RUKA(로봇 손)의 모터 동작 범위를 측정하고 저장하는
캘리브레이션 프로그램입니다.

주요 기능:
1. Curl Limits 자동 측정 - 손가락이 완전히 구부러진 위치 자동 측정
2. Tension Limits 대화형 조정 - 텐던 장력이 유지된 펼친 위치 수동 미세 조정
3. 개별 모터 최적화 - 11개 모터 각각의 고유한 동작 범위 저장
4. 양손 지원 - 오른손/왼손 각각의 미러 방향 처리

캘리브레이션 목적:
- 텐던 구동 방식의 특성상 케이블 장력이 제작마다 다름
- 모터별로 동작 범위가 상이함
- 정확한 제어를 위해 개별 캘리브레이션 필수
- 안전한 동작 범위 설정으로 하드웨어 보호

캘리브레이션 결과물:
motor_limits/
├── {hand_type}_curl_limits.npy      # 11개 모터의 구부린 위치
└── {hand_type}_tension_limits.npy   # 11개 모터의 펼친 위치

사용 방법:
python calibrate_motors.py --hand-type right --mode both
python calibrate_motors.py -ht left -m curl
python calibrate_motors.py -ht right -m tension

시스템 요구사항:
- Windows 10/11 (msvcrt 모듈 사용)
- Python 3.8 이상
- NumPy

작성: NYU RUKA Team
라이선스: MIT License
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import os        # 파일 경로 조작, 디렉토리 생성
import sys       # 시스템 입출력 제어
import time      # 시간 지연 및 타이밍 제어

import numpy as np  # 배열 연산 및 데이터 저장

# RUKA 프로젝트 모듈 임포트
from ruka_hand.control.hand import *            # Hand 클래스: 로봇 손 제어
from ruka_hand.utils.file_ops import get_repo_root  # 프로젝트 루트 경로

# Windows 키보드 입력용 (get_key() 함수 내부에서 import)
# import msvcrt

# =============================================================================
# 의존성 파일 설명 (Windows 버전)
# =============================================================================

"""
이 스크립트는 다음 파일들과 의존 관계를 가집니다:

┌─────────────────────────────────────────────────────────────────┐
│         calibrate_motors.py (현재 파일 - Windows 버전)           │
│                                                                   │
│  역할: RUKA 로봇 손의 모터 동작 범위 측정 및 저장                  │
└─────────────────────────────────────────────────────────────────┘
│
├─► [1] ruka_hand/control/hand.py
│     │
│     ├─ Hand 클래스 제공
│     ├─ set_pos(): 모터 위치 명령
│     ├─ read_pos(): 현재 위치 읽기
│     ├─ read_single_cur(): 단일 모터 전류 읽기
│     ├─ hand_type 속성 (left/right)
│     └─ Dynamixel 모터 저수준 제어
│
├─► [2] ruka_hand/utils/file_ops.py
│     │
│     ├─ get_repo_root(): 프로젝트 루트 경로 반환
│     └─ 파일 시스템 유틸리티
│
└─► [간접 의존성]
    │
    ├─► ruka_hand/utils/constants.py
    │     ├─ USB_PORTS: 시리얼 포트 정의
    │     ├─ FINGER_NAMES_TO_MOTOR_IDS: 손가락→모터 매핑
    │     ├─ MOTOR_RANGES: 모터 동작 범위
    │     └─ 기타 시스템 상수
    │
    ├─► ruka_hand/utils/dynamixel_util.py
    │     ├─ DynamixelClient 클래스
    │     ├─ ModBus RTU 프로토콜 통신
    │     ├─ GroupSyncWrite/Read
    │     └─ 저수준 하드웨어 인터페이스
    │
    └─► 표준 라이브러리
          ├─ os: 파일 시스템 조작
          ├─ sys: 시스템 입출력
          ├─ time: 타이밍 제어
          ├─ msvcrt: Windows 키보드 입력 (Windows 전용)
          ├─ numpy: 수치 연산 및 데이터 저장
          └─ argparse: 명령줄 인자 파싱

데이터 흐름:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] calibrate_motors.py 시작
    ↓
[2] argparse로 명령줄 인자 파싱
    ├─ hand_type: "right" 또는 "left"
    ├─ mode: "curl", "tension", "both"
    ├─ curr_lim: 전류 제한값 (기본 50mA)
    └─ testing: 디버그 출력 활성화
    ↓
[3] HandCalibrator 클래스 인스턴스 생성
    ↓
    ├─► Hand 클래스 초기화 (hand.py)
    │     ├─ DynamixelClient 연결 (시리얼 포트)
    │     ├─ 모터 ID 설정 (1~11번)
    │     ├─ Operating Mode 설정 (Mode 5)
    │     ├─ PID 게인 설정
    │     ├─ Current/Temp/Velocity Limit 설정
    │     └─ Torque Enable
    │
    ├─ save_dir 경로 설정 (curl_limits/)
    ├─ curled_path 설정 ({hand_type}_curl_limits.npy)
    └─ tension_path 설정 ({hand_type}_tension_limits.npy)
    ↓
[4A] mode가 "curl" 또는 "both"인 경우
    ↓
    save_curled_limits() 실행
    ↓
    ├─ find_curled() 호출
    │   ↓
    │   └─ 11개 모터 순회
    │         ├─► find_bound(motor_id) 호출
    │         │     ├─ 이진 탐색으로 전류 제한 위치 찾기
    │         │     ├─ hand.set_pos() 반복 호출
    │         │     ├─ hand.read_single_cur() 전류 측정
    │         │     ├─ hand.read_pos() 위치 확인
    │         │     └─ 수렴할 때까지 반복
    │         └─ 각 모터의 curl 위치 저장
    ↓
    └─ np.save(curled_path, curled) 파일 저장
    ↓
[4B] mode가 "tension" 또는 "both"인 경우
    ↓
    save_tensioned_limits() 실행
    ↓
    ├─ curled_limits.npy 로드 (없으면 자동 측정)
    │   ↓
    ├─ estimate_tensioned_from_curled() 호출
    │   ├─ 오른손: tensioned = curled - 1100
    │   └─ 왼손: tensioned = curled + 1100
    │   ↓
    ├─ interactive_refine_tensioned() 호출
    │   ↓
    │   └─ 11개 모터 순회
    │         ├─ 현재 모터를 tensioned 위치로 이동
    │         ├─ 사용자 키 입력 대기 (msvcrt.getch())
    │         │   ├─ ↑/→: +10씩 증가
    │         │   ├─ ↓/←: -10씩 감소
    │         │   ├─ Enter: 현재 값 저장 및 다음 모터
    │         │   └─ q: 현재 값 유지하고 스킵
    │         └─ 각 모터의 tension 위치 저장
    ↓
    └─ np.save(tension_path, tensioned) 파일 저장
    ↓
[5] hand.close() 호출
    ↓
    ├─ torque_enabled = False
    ├─ DynamixelClient.disconnect()
    └─ 시리얼 포트 닫기
    ↓
[6] 프로그램 종료
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# 키보드 입력 캡처 함수 (Windows 전용)
# =============================================================================

def get_key():
    """
    단일 키 입력을 캡처하는 함수 (Windows 전용)
    
    msvcrt 모듈을 사용하여 키 입력을 즉시 읽습니다.
    화살표 키는 2바이트 시퀀스로 전송되며, Unix 스타일로 변환합니다.
    
    동작 원리:
    1. msvcrt.getch()로 첫 번째 바이트 읽기
    2. 특수 키(0x00, 0xe0)인 경우 다음 바이트도 읽기
    3. 화살표 키 코드를 Unix 스타일 이스케이프 시퀀스로 변환
    4. 일반 키는 UTF-8 디코딩하여 반환
    
    Windows 키 코드:
    - Up Arrow:    0xe0 + 0x48 → \x1b[A
    - Down Arrow:  0xe0 + 0x50 → \x1b[B
    - Right Arrow: 0xe0 + 0x4d → \x1b[C
    - Left Arrow:  0xe0 + 0x4b → \x1b[D
    - Enter:       \r
    
    Returns:
        str: 입력된 키 문자열 (화살표 키는 3문자 시퀀스)
    
    예시:
        key = get_key()
        if key == "\x1b[A":     # Up Arrow
            print("위쪽 화살표 입력됨")
        elif key == "\r":        # Enter
            print("엔터 입력됨")
    
    주의사항:
    - Windows 전용 (msvcrt 모듈 사용)
    - Python 3.x 필수
    """
    import msvcrt
    
    # 첫 번째 바이트 읽기
    ch = msvcrt.getch()
    
    # 특수 키인 경우 (0x00 또는 0xe0)
    if ch in (b'\x00', b'\xe0'):
        # 다음 바이트 읽기 (실제 키 코드)
        ch2 = msvcrt.getch()
        
        # Windows 키 코드를 Unix 스타일 이스케이프 시퀀스로 변환
        key_map = {
            b'H': '\x1b[A',  # ↑ Up Arrow
            b'P': '\x1b[B',  # ↓ Down Arrow
            b'M': '\x1b[C',  # → Right Arrow
            b'K': '\x1b[D',  # ← Left Arrow
        }
        return key_map.get(ch2, ch2.decode('utf-8', errors='ignore'))
    
    # 일반 키 - UTF-8 디코딩
    return ch.decode('utf-8', errors='ignore')


# =============================================================================
# HandCalibrator 클래스
# =============================================================================

class HandCalibrator:
    """
    로봇 손 모터 캘리브레이션 클래스
    
    RUKA 로봇 손의 11개 모터 각각의 동작 범위를 측정하고 저장합니다.
    
    주요 기능:
    1. Curl Limits 자동 측정
       - 이진 탐색 알고리즘 사용
       - 전류 제한값 감지로 최대 구부림 위치 찾기
       - 모터 과부하 방지
    
    2. Tension Limits 대화형 조정
       - Curl 위치에서 일정 거리 떨어진 초기값 계산
       - 사용자가 화살표 키로 미세 조정
       - 각 모터별 개별 최적화
    
    3. 양손 지원
       - 오른손: 모터 값 증가 = 손가락 구부림
       - 왼손: 모터 값 감소 = 손가락 구부림 (미러)
    
    캘리브레이션 필요성:
    - 텐던 장력이 제작마다 다름
    - 모터 위치와 실제 관절 각도 비선형 관계
    - 안전한 동작 범위 설정으로 하드웨어 보호
    
    Attributes:
        hand (Hand): 로봇 손 제어 객체
        curr_lim (int): 전류 제한값 (mA 단위)
        testing (bool): 디버그 출력 활성화 플래그
        motor_ids (list): 캘리브레이션할 모터 ID 리스트
        data_save_dir (str): 캘리브레이션 데이터 저장 디렉토리
        curled_path (str): curl_limits.npy 파일 경로
        tension_path (str): tension_limits.npy 파일 경로
    """
    
    def __init__(
        self,
        data_save_dir,
        hand_type,
        curr_lim=50,
        testing=False,
        motor_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        """
        HandCalibrator 초기화 메서드
        
        Args:
            data_save_dir (str): 캘리브레이션 데이터를 저장할 디렉토리 경로
            hand_type (str): 핸드 타입 ("right" 또는 "left")
            curr_lim (int, optional): 전류 제한값 (mA). 기본값 50mA.
                                     이 값에 도달하면 최대 구부림으로 판단
            testing (bool, optional): 디버그 출력 활성화. 기본값 False.
                                     True시 상세한 진행 정보 출력
            motor_ids (list, optional): 캘리브레이션할 모터 ID 리스트.
                                       기본값 [1, 2, ..., 11]
        
        초기화 과정:
        1. Hand 객체 생성
           - Dynamixel 모터 연결
           - 모터 설정 (Operating Mode, PID, Limits)
           - Torque Enable
        
        2. 캘리브레이션 파라미터 설정
           - 전류 제한값 (curr_lim)
           - 테스트 모드 플래그 (testing)
           - 대상 모터 ID 리스트 (motor_ids)
        
        3. 파일 경로 설정
           - 저장 디렉토리 (data_save_dir)
           - curl_limits.npy 경로
           - tension_limits.npy 경로
        
        특수 모터 전류 제한:
        - 모터 4번 (검지 MCP): 250mA (더 큰 힘 필요)
        - 모터 5번 (검지 PIP): 200mA
        - 기타 모터: 50mA (기본값)
        
        예외 처리:
        - Hand 초기화 실패 시 프로그램 종료
        - 시리얼 포트 연결 오류
        - 캘리브레이션 파일 없음 (자동 생성)
        """
        print("\n" + "="*70)
        print("RUKA Robot Hand Motor Calibration")
        print("로봇 손 모터 캘리브레이션 프로그램")
        print("="*70)
        print(f"\n[초기화 시작]")
        print(f"  손 종류: {hand_type.upper()}")
        print(f"  전류 제한: {curr_lim} mA")
        print(f"  테스트 모드: {'활성화' if testing else '비활성화'}")
        print(f"  대상 모터: {len(motor_ids)}개 (ID: {motor_ids})")
        
        try:
            # Hand 클래스 인스턴스 생성
            # hand.py의 __init__() 메서드 실행
            print(f"\n  → Hand 클래스 초기화 중...")
            self.hand = Hand(hand_type)
            print(f"  ✓ Hand 클래스 초기화 완료")
            print(f"  ✓ Dynamixel 모터 연결 완료 (11개 모터)")
            print(f"  ✓ 모터 토크 활성화 완료")
            
        except Exception as e:
            print(f"\n✗ Hand 클래스 초기화 실패!")
            print(f"  에러: {e}")
            print(f"\n가능한 원인:")
            print(f"  1. USB 케이블이 연결되지 않음")
            print(f"  2. 로봇 손 전원이 꺼져 있음")
            print(f"  3. 시리얼 포트 권한 부족")
            print(f"  4. 다른 프로그램에서 포트 사용 중")
            raise
        
        # 캘리브레이션 파라미터 저장
        self.curr_lim = curr_lim
        self.testing = testing
        self.motor_ids = motor_ids
        self.data_save_dir = data_save_dir
        
        # 저장 경로 설정
        # 예: curl_limits/right_curl_limits.npy
        self.curled_path = os.path.join(
            self.data_save_dir, f"{hand_type}_curl_limits.npy"
        )
        # 예: curl_limits/right_tension_limits.npy
        self.tension_path = os.path.join(
            self.data_save_dir, f"{hand_type}_tension_limits.npy"
        )
        
        print(f"\n  저장 디렉토리: {data_save_dir}")
        print(f"  Curl 파일: {os.path.basename(self.curled_path)}")
        print(f"  Tension 파일: {os.path.basename(self.tension_path)}")
        print(f"\n[초기화 완료]")

    
    def _safe_read_pos(self):
        """
        안전하게 모터 위치를 읽고 int32 NumPy 배열로 반환
        
        hand.read_pos()가 반환하는 값의 타입이 불확실하므로
        안전하게 int32로 변환합니다.
        
        Returns:
            np.ndarray: int32 타입의 위치 배열
        """
        pos = self.hand.read_pos()
        
        # 리스트나 배열을 int32로 변환
        if isinstance(pos, (list, tuple)):
            return np.array([int(x) for x in pos], dtype=np.int32)
        elif isinstance(pos, np.ndarray):
            return pos.astype(np.int32)
        else:
            # 단일 값인 경우
            return np.array([int(pos)], dtype=np.int32)
    
    
    def _safe_set_pos(self, pos):
        """
        안전하게 모터 위치를 설정
        
        NumPy 배열을 int 리스트로 변환하여 전달합니다.
        
        Args:
            pos: 위치 배열 또는 리스트
        """
        if isinstance(pos, np.ndarray):
            # NumPy 배열을 Python int 리스트로 변환
            pos_list = [int(x) for x in pos]
            self.hand.set_pos(pos_list)
        else:
            self.hand.set_pos(pos)


    def find_bound(self, motor_id):
        """
        이진 탐색으로 단일 모터의 최대 구부림 위치(Curl Limit) 찾기
        
        이 함수는 모터를 점진적으로 구부리면서 전류 제한값에 도달하는
        위치를 이진 탐색 알고리즘으로 찾습니다.
        
        Args:
            motor_id (int): 캘리브레이션할 모터 ID (1~11)
        
        Returns:
            int: 모터의 최대 구부림 위치 (0~4095)
        
        알고리즘 동작 원리:
        
        1. 초기 설정
           - 탐색 범위: [100, 4000] (안전 마진 포함)
           - 시작 위치: 오른손 100, 왼손 4000
           - 방향 계수: 오른손 +1, 왼손 -1
        
        2. 이진 탐색 루프
           while (탐색 범위 > 10) or (전류 < 제한값):
               a) 중간 위치 계산: com_pos = (상한 + 하한) / 2
               b) 모터를 com_pos로 이동
               c) 안정화 대기 (2~5초)
               d) 현재 전류 측정
               e) 실제 도달 위치 읽기
               
               if 전류 < 제한값:  # 아직 더 구부릴 수 있음
                   오른손: 하한 = 현재위치 + 1
                   왼손: 상한 = 현재위치 - 1
               else:  # 전류 제한 도달, 너무 구부렸음
                   오른손: 상한 = 현재위치 + 1
                   왼손: 하한 = 현재위치 - 1
        
        3. 수렴 조건
           - 탐색 범위가 10 이하로 좁아지고
           - 동시에 전류가 제한값 이하
        
        특수 모터 처리:
        - 모터 4번 (검지 MCP): curr_lim = 250mA, 대기시간 5초
        - 모터 5번 (검지 PIP): curr_lim = 200mA, 대기시간 5초
        - 기타 모터: curr_lim = 50mA, 대기시간 2초
        
        예시:
            오른손 모터 1번 캘리브레이션
            
            초기: l_bound=100, u_bound=4000
            
            반복 1: com_pos=2050 → cur=30mA < 50mA
                   → 더 구부릴 수 있음 → l_bound=2051
            
            반복 2: com_pos=3025 → cur=35mA < 50mA
                   → 더 구부릴 수 있음 → l_bound=3026
            
            반복 3: com_pos=3513 → cur=48mA < 50mA
                   → 더 구부릴 수 있음 → l_bound=3514
            
            반복 4: com_pos=3757 → cur=52mA > 50mA
                   → 너무 구부림 → u_bound=3758
            
            반복 5: com_pos=3635 → cur=49mA < 50mA
                   → l_bound=3636, 범위=122
            
            ...
            
            최종: pres_pos=3642 (전류 50mA 직전 위치)
        
        주의사항:
        - 모터가 물리적 한계에 도달하면 전류가 급증
        - 너무 높은 전류는 모터 손상 위험
        - 안정화 대기 시간이 중요 (관성 및 텐던 장력 안정화)
        - 모터마다 기계적 특성이 다를 수 있음
        """
        # 특수 모터의 경우 전류 제한값과 대기 시간 조정
        t = 2  # 기본 대기 시간 2초
        if motor_id in [4, 5]:
            # 모터 4번과 5번은 검지 손가락 (더 큰 힘 필요)
            if motor_id == 4:
                self.curr_lim = 250  # MCP 관절: 250mA
            else:
                self.curr_lim = 200  # PIP 관절: 200mA
            t = 5  # 더 긴 안정화 시간 필요
        
        # 진행 상황 출력
        print(f"\n{'─'*70}")
        print(f"[모터 {motor_id} 캘리브레이션 시작]")
        print(f"  전류 제한: {self.curr_lim} mA")
        print(f"  안정화 시간: {t}초")
        
        if self.testing:
            print(f"\n{'='*70}")
            print(f"MOTOR {motor_id} - 상세 디버그 정보")
            print(f"{'='*70}")
        
        # 오른손/왼손에 따른 초기 설정
        if self.hand.hand_type == "right":
            start_pos = 100      # 오른손 시작 위치 (펼친 상태)
            f = 1                # 방향 계수: +1 (증가 = 구부림)
            print(f"  손 방향: 오른손 (모터 값 증가 = 구부림)")
        elif self.hand.hand_type == "left":
            start_pos = 4000     # 왼손 시작 위치 (펼친 상태)
            f = -1               # 방향 계수: -1 (감소 = 구부림)
            print(f"  손 방향: 왼손 (모터 값 감소 = 구부림)")
        
        # 이진 탐색 범위 초기화
        l_bound = 100    # 하한 (안전 마진)
        u_bound = 4000   # 상한 (12비트 해상도의 최대값 4095에서 마진)
        
        # 모든 모터를 시작 위치로 초기화
        pos = np.array([start_pos] * 11, dtype=np.int32)
        
        # 초기 전류값 (큰 값으로 설정하여 루프 진입 보장)
        cur = 1000000
        
        # 이진 탐색 카운터
        iteration = 0
        
        print(f"  초기 탐색 범위: [{l_bound}, {u_bound}]")
        print(f"\n  이진 탐색 시작...")
        
        # 이진 탐색 메인 루프
        # 조건: 탐색 범위가 10보다 크거나, 전류가 제한값을 초과하는 동안
        while abs(u_bound - l_bound) > 10 or f * cur > self.curr_lim:
            iteration += 1
            
            # 중간 위치 계산
            com_pos = int((u_bound + l_bound) // 2 - 1)
            
            # 대상 모터만 com_pos로 설정 (나머지는 start_pos 유지)
            pos[motor_id - 1] = com_pos
            
            # 모터 이동 명령
            self._safe_set_pos(pos)
            
            # 안정화 대기 (모터 이동 완료 + 텐던 장력 안정화)
            print(f"    반복 {iteration}: 위치 {com_pos}로 이동 중... ", end='', flush=True)
            time.sleep(t)
            
            # 현재 전류 측정
            cur = self.hand.read_single_cur(motor_id)
            
            # 실제 도달 위치 읽기 (명령 위치와 다를 수 있음)
            pres_pos = int(self._safe_read_pos()[motor_id - 1])
            
            print(f"완료 (전류: {cur:.1f}mA, 실제 위치: {pres_pos})")
            
            # 디버그 모드: 상세 정보 출력
            if self.testing:
                print(f"  상한: {u_bound:4d}  |  하한: {l_bound:4d}  |  "
                      f"명령: {com_pos:4d}  |  실제: {pres_pos:4d}  |  "
                      f"전류: {cur:.1f}mA")
            
            # 이진 탐색 범위 업데이트
            if f * cur < self.curr_lim:
                # 전류가 제한값보다 작음 → 아직 더 구부릴 수 있음
                if self.hand.hand_type == "right":
                    # 오른손: 하한을 현재 위치로 올림
                    l_bound = pres_pos + 1
                    u_bound -= 1
                else:
                    # 왼손: 상한을 현재 위치로 내림
                    u_bound = pres_pos - 1
                    l_bound += 1
                
                if self.testing:
                    print(f"    → 전류 부족 ({f * cur:.1f} < {self.curr_lim}), "
                          f"더 구부림 가능")
            else:
                # 전류가 제한값 이상 → 너무 구부렸음
                if self.hand.hand_type == "right":
                    # 오른손: 상한을 현재 위치로 내림
                    u_bound = pres_pos + 1
                else:
                    # 왼손: 하한을 현재 위치로 올림
                    l_bound = pres_pos - 1
                
                if self.testing:
                    print(f"    → 전류 초과 ({f * cur:.1f} > {self.curr_lim}), "
                          f"후퇴 필요")
            
            # 현재 탐색 범위 출력
            range_size = abs(u_bound - l_bound)
            if self.testing:
                print(f"    → 새 범위: [{l_bound}, {u_bound}] (크기: {range_size})")
        
        # 이진 탐색 완료
        print(f"\n  ✓ 모터 {motor_id} 캘리브레이션 완료")
        print(f"    최종 위치: {pres_pos}")
        print(f"    최종 전류: {cur:.1f} mA")
        print(f"    반복 횟수: {iteration}회")
        
        return pres_pos


    def find_curled(self):
        """
        모든 모터의 Curl Limits를 순차적으로 측정
        
        11개 모터 각각에 대해 find_bound()를 호출하여
        최대 구부림 위치를 측정하고 배열로 반환합니다.
        
        Returns:
            np.ndarray: 11개 모터의 curl 위치 배열 (shape: (11,), dtype: int)
        
        측정 순서:
        1. 모터 1번 (엄지 IP)
        2. 모터 2번 (엄지 MCP)
        3. 모터 3번 (검지 DIP)
        4. 모터 4번 (검지 MCP)
        5. 모터 5번 (검지 PIP)
        6. 모터 6번 (중지 DIP)
        7. 모터 7번 (중지 MCP)
        8. 모터 8번 (약지 DIP)
        9. 모터 9번 (약지 MCP)
        10. 모터 10번 (소지 DIP)
        11. 모터 11번 (소지 MCP)
        
        각 모터는 독립적으로 측정되며, 다른 모터는 안전한
        펼친 위치에 유지됩니다.
        
        소요 시간:
        - 모터 1-3, 6-11: 약 20-40초/모터 (이진 탐색 10-20회)
        - 모터 4-5: 약 50-100초/모터 (더 긴 안정화 시간)
        - 총 예상 시간: 약 5-10분
        
        주의사항:
        - 측정 중 손가락이 움직이므로 주변 장애물 제거 필요
        - 모터가 발열할 수 있으므로 과열 모니터링 권장
        - 비정상적인 소리나 움직임 발견 시 즉시 중단
        """
        print("\n" + "="*70)
        print("[Curl Limits 자동 측정 시작]")
        print("="*70)
        print(f"\n  총 모터 수: {len(self.motor_ids)}개")
        print(f"  예상 소요 시간: 약 5-10분")
        print(f"\n  주의사항:")
        print(f"  - 로봇 손 주변에 장애물이 없는지 확인하세요")
        print(f"  - 손가락이 점진적으로 구부러집니다")
        print(f"  - 비정상적인 소리 발생 시 Ctrl+C로 중단하세요")
        
        # 결과 저장 배열 초기화
        curled = np.zeros(len(self.motor_ids), dtype=np.int32)
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 모든 모터 순회
        for i, mid in enumerate(self.motor_ids):
            print(f"\n진행: [{i+1}/{len(self.motor_ids)}] 모터 {mid}번 측정 중...")
            
            try:
                # 이진 탐색으로 curl 위치 찾기
                curled[i] = int(self.find_bound(mid))
                
                # 중간 결과 출력
                print(f"  모터 {mid}: {curled[i]} (✓)")
                
            except KeyboardInterrupt:
                print(f"\n\n{'='*70}")
                print(f"[사용자 중단]")
                print(f"  Ctrl+C가 감지되었습니다.")
                print(f"  현재까지 측정된 데이터:")
                for j in range(i):
                    print(f"    모터 {self.motor_ids[j]}: {curled[j]}")
                print(f"\n  나머지 {len(self.motor_ids)-i}개 모터는 측정되지 않았습니다.")
                raise
            
            except Exception as e:
                print(f"\n✗ 모터 {mid} 측정 실패!")
                print(f"  에러: {e}")
                print(f"  현재까지 측정된 데이터:")
                for j in range(i):
                    print(f"    모터 {self.motor_ids[j]}: {curled[j]}")
                raise
        
        # 완료 시간 계산
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        # 최종 결과 출력
        print(f"\n{'='*70}")
        print(f"[Curl Limits 측정 완료]")
        print(f"{'='*70}")
        print(f"\n  소요 시간: {minutes}분 {seconds}초")
        print(f"\n  측정 결과:")
        for i, mid in enumerate(self.motor_ids):
            print(f"    모터 {mid:2d}: {curled[i]:4d}")
        
        return curled


    def estimate_tensioned_from_curled(self, curled):
        """
        Curl 위치로부터 Tension 위치의 초기 추정값 계산
        
        Curl 위치(완전히 구부린 상태)로부터 일정 거리(1100)만큼
        떨어진 위치를 Tension 위치의 초기값으로 사용합니다.
        이 값은 사용자가 대화형으로 미세 조정할 수 있습니다.
        
        Args:
            curled (np.ndarray): 11개 모터의 curl 위치 배열
        
        Returns:
            np.ndarray: 11개 모터의 tension 초기 추정값 (shape: (11,), dtype: int)
        
        계산 방식:
        - 오른손: tensioned = curled - 1100
          (모터 값 감소 = 손가락 펼침)
        
        - 왼손: tensioned = curled + 1100
          (모터 값 증가 = 손가락 펼침, 미러)
        
        오프셋 1100의 의미:
        - 경험적으로 결정된 값
        - 손가락이 완전히 펼쳐지되 텐던이 느슨하지 않은 상태
        - 로봇마다 조정 필요 (재질, 텐던 장력 차이)
        
        예시:
            오른손 모터 1번
            curled = 3642
            tensioned = 3642 - 1100 = 2542
            
            왼손 모터 1번
            curled = 458
            tensioned = 458 + 1100 = 1558
        
        주의사항:
        - 이 값은 초기 추정값일 뿐
        - 반드시 대화형 조정(interactive_refine_tensioned) 필요
        - 텐던이 너무 느슨하거나 팽팽하지 않도록 조정
        - 각 모터마다 최적값이 다를 수 있음
        """
        print(f"\n{'─'*70}")
        print(f"[Tension Limits 초기 추정]")
        print(f"{'─'*70}")
        
        # 방향 계수: 오른손 +1, 왼손 -1
        f = 1 if self.hand.hand_type == "right" else -1
        
        # 추정 계산
        tensioned = np.array([int(x - f * 1100) for x in curled], dtype=np.int32)
        
        print(f"  방향: {self.hand.hand_type.upper()}")
        print(f"  오프셋: {f * 1100} ({'감소' if f == 1 else '증가'})")
        print(f"\n  초기 추정값:")
        
        for i, mid in enumerate(self.motor_ids):
            direction = "←" if f == 1 else "→"
            print(f"    모터 {mid:2d}: {curled[i]:4d} {direction} {tensioned[i]:4d} "
                  f"(차이: {abs(curled[i] - tensioned[i])})")
        
        print(f"\n  ※ 이 값은 초기 추정값입니다.")
        print(f"     대화형 조정으로 미세 조정하세요.")
        
        return tensioned


    def interactive_refine_tensioned(self, tensioned_init, step=10):
        """
        대화형 인터페이스로 Tension Limits 미세 조정
        
        각 모터를 순회하면서 사용자가 화살표 키로 위치를 미세 조정할 수 있습니다.
        실시간으로 모터가 움직이므로 시각적으로 최적 위치를 찾을 수 있습니다.
        
        Args:
            tensioned_init (np.ndarray): 초기 tension 위치 배열 (11개)
            step (int, optional): 한 번 키 입력당 이동 거리. 기본값 10.
        
        Returns:
            np.ndarray: 사용자가 조정한 최종 tension 위치 (shape: (11,), dtype: int)
        
        키 조작법:
        - ↑ (Up) / → (Right): 현재 위치에서 +step*f 이동
        - ↓ (Down) / ← (Left): 현재 위치에서 -step*f 이동
        - Enter: 현재 값을 저장하고 다음 모터로 이동
        - q: 현재 모터 스킵 (초기값 유지)
        - Ctrl+C: 전체 캘리브레이션 중단
        
        방향 계수 (f):
        - 오른손: f = +1
          - 위/오른쪽 화살표: 모터 값 증가 (손가락 구부림)
          - 아래/왼쪽 화살표: 모터 값 감소 (손가락 펼침)
        
        - 왼손: f = -1
          - 위/오른쪽 화살표: 모터 값 감소 (손가락 구부림)
          - 아래/왼쪽 화살표: 모터 값 증가 (손가락 펼침)
        
        조정 순서:
        1. 대상 모터를 초기 tension 위치로 이동
        2. 나머지 모터는 현재 위치 유지
        3. 사용자 키 입력 대기
        4. 화살표 키: 위치 조정 (step 단위)
        5. Enter: 값 저장 및 다음 모터
        6. q: 스킵
        
        조정 팁:
        - 손가락이 완전히 펼쳐져야 함
        - 텐던이 느슨하지 않아야 함 (약간의 장력 유지)
        - 손가락을 가볍게 눌러봤을 때 저항이 느껴져야 함
        - 너무 팽팽하면 손가락 동작 범위 감소
        - 너무 느슨하면 제어 정밀도 저하
        
        위치 제한:
        - 최소: 10 (안전 마진)
        - 최대: 4090 (12비트 해상도 4095에서 마진)
        
        예시 세션:
            [모터 1] Current candidate: 2542
            Adjust with arrows, Enter to save, 'q' to abort this motor.
            
            (사용자가 ↑ 입력)
            [모터 1] Current candidate: 2552
            
            (사용자가 ↑ 입력)
            [모터 1] Current candidate: 2562
            
            (사용자가 ↓ 입력)
            [모터 1] Current candidate: 2552
            
            (사용자가 Enter 입력)
            Saved Motor 1 tensioned = 2552
            
            [모터 2] Current candidate: 2634
            ...
        """
        print("\n" + "="*70)
        print("[Tension Limits 대화형 조정]")
        print("="*70)
        
        print(f"\n  조작법:")
        print(f"    ↑ 또는 → : +{step}씩 이동 (손가락 구부림 방향)")
        print(f"    ↓ 또는 ← : -{step}씩 이동 (손가락 펼침 방향)")
        print(f"    Enter     : 현재 값 저장 및 다음 모터")
        print(f"    q         : 현재 모터 스킵 (초기값 유지)")
        print(f"    Ctrl+C    : 전체 중단")
        
        print(f"\n  조정 목표:")
        print(f"    - 손가락이 완전히 펼쳐진 상태")
        print(f"    - 텐던에 약간의 장력 유지 (느슨하지 않게)")
        print(f"    - 손가락을 눌러봤을 때 저항감 있어야 함")
        
        print(f"\n  현재 모터를 조정 중일 때 다른 모터는 안전 위치에 유지됩니다.")
        print(f"  실시간으로 모터가 움직이므로 시각적 확인 가능합니다.\n")
        
        # 현재 모터 위치 읽기 (안전 위치로 사용)
        current_pos = self._safe_read_pos()
        
        # 조정 결과 저장 배열 (초기값으로 시작)
        tensioned = tensioned_init.copy()
        
        # 방향 계수: 오른손 +1, 왼손 -1
        f = 1 if self.hand.hand_type == "right" else -1
        
        # 모든 모터 순회
        for motor_idx, mid in enumerate(self.motor_ids):
            idx = mid - 1  # 배열 인덱스 (0-based)
            
            print(f"\n{'─'*70}")
            print(f"[모터 {mid} 조정] - [{motor_idx + 1}/{len(self.motor_ids)}]")
            print(f"{'─'*70}")
            print(f"  초기 추정값: {tensioned[idx]}")
            
            # 모든 모터를 안전 위치로, 현재 모터만 tension 위치로
            pos = current_pos.copy()
            pos[idx] = tensioned[idx]
            self._safe_set_pos(pos)
            
            # 모터 이동 완료 대기
            print(f"  → 모터 이동 중... ", end='', flush=True)
            time.sleep(0.2)
            print(f"완료")
            
            # 실제 도달 위치 확인
            actual_pos = int(self._safe_read_pos()[idx])
            if abs(actual_pos - pos[idx]) > 50:
                print(f"  ⚠️ 경고: 목표 위치({pos[idx]})와 실제 위치({actual_pos}) 차이 큼")
            
            # 대화형 조정 루프
            adjustment_count = 0
            
            while True:
                # 현재 상태 출력
                print(f"\n  [모터 {mid}] 현재 후보: {pos[idx]:4d}")
                print(f"  화살표로 조정, Enter로 저장, 'q'로 스킵: ", end='', flush=True)
                
                # 키 입력 대기
                k = get_key()
                
                if k in ("\r", "\n"):
                    # Enter: 현재 값 저장
                    tensioned[idx] = int(pos[idx])
                    print(f"\n  ✓ 모터 {mid} 저장: {tensioned[idx]}")
                    print(f"    조정 횟수: {adjustment_count}회")
                    break
                
                elif k in ("\x1b[A", "\x1b[C"):
                    # Up Arrow 또는 Right Arrow: +step*f
                    old_pos = int(pos[idx])
                    pos[idx] = int(max(min(int(pos[idx]) + step * f, 4090), 10))
                    
                    if pos[idx] != old_pos:
                        self._safe_set_pos(pos)
                        adjustment_count += 1
                        direction = "구부림" if f == 1 else "펼침"
                        print(f"\n    → {old_pos} → {pos[idx]} (+{step*f}, {direction})")
                    else:
                        print(f"\n    (위치 제한 도달)")
                
                elif k in ("\x1b[B", "\x1b[D"):
                    # Down Arrow 또는 Left Arrow: -step*f
                    old_pos = int(pos[idx])
                    pos[idx] = int(max(min(int(pos[idx]) - step * f, 4090), 10))
                    
                    if pos[idx] != old_pos:
                        self._safe_set_pos(pos)
                        adjustment_count += 1
                        direction = "펼침" if f == 1 else "구부림"
                        print(f"\n    → {old_pos} → {pos[idx]} ({-step*f}, {direction})")
                    else:
                        print(f"\n    (위치 제한 도달)")
                
                elif k.lower() == "q":
                    # q: 스킵
                    print(f"\n  ⊗ 모터 {mid} 스킵 (초기값 {tensioned[idx]} 유지)")
                    break
                
                else:
                    # 기타 키 무시
                    print(f"\n    (알 수 없는 키: 화살표/Enter/q만 사용)")
        
        # 최종 결과 출력
        print(f"\n{'='*70}")
        print(f"[Tension Limits 조정 완료]")
        print(f"{'='*70}")
        print(f"\n  최종 tension 배열:")
        
        for i, mid in enumerate(self.motor_ids):
            diff = tensioned[i] - tensioned_init[i]
            sign = "+" if diff > 0 else ""
            status = "변경됨" if diff != 0 else "유지"
            print(f"    모터 {mid:2d}: {tensioned[i]:4d} "
                  f"(초기값 {tensioned_init[i]:4d} {sign}{diff:+4d}) - {status}")
        
        return tensioned.astype(int)


    def save_curled_limits(self):
        """
        Curl Limits 측정 및 저장
        
        find_curled()를 호출하여 11개 모터의 최대 구부림 위치를 측정하고
        numpy 파일(.npy)로 저장합니다.
        
        저장 경로:
            curl_limits/{hand_type}_curl_limits.npy
        
        파일 형식:
            - Numpy binary format (.npy)
            - Shape: (11,)
            - Dtype: int
            - 값 범위: [100, 4000]
        
        예시:
            오른손: curl_limits/right_curl_limits.npy
            [3642, 3521, 2894, 3743, 3156, 2987, 3621, 3104, 3589, 3012, 3598]
            
            왼손: curl_limits/left_curl_limits.npy
            [458, 579, 1206, 357, 944, 1113, 479, 996, 511, 1088, 502]
        
        주의사항:
        - 기존 파일이 있으면 덮어쓰기
        - 측정 실패 시 예외 발생하고 파일 미생성
        - 측정 후 반드시 육안으로 확인 권장
        """
        print("\n" + "="*70)
        print("[Curl Limits 저장]")
        print("="*70)
        
        try:
            # Curl Limits 측정
            print(f"\n  측정 시작...")
            curled = self.find_curled()
            
            # 파일 저장
            print(f"\n  파일 저장 중...")
            np.save(self.curled_path, curled)
            
            # 완료 메시지
            print(f"\n  ✓ Curl Limits 저장 완료!")
            print(f"    파일: {self.curled_path}")
            print(f"    크기: {os.path.getsize(self.curled_path)} bytes")
            
            # 저장된 데이터 확인
            print(f"\n  저장된 데이터 확인:")
            loaded = np.load(self.curled_path)
            print(f"    Shape: {loaded.shape}")
            print(f"    Dtype: {loaded.dtype}")
            print(f"    Min: {loaded.min()}, Max: {loaded.max()}")
            print(f"    Mean: {loaded.mean():.1f}, Std: {loaded.std():.1f}")
            
        except Exception as e:
            print(f"\n  ✗ Curl Limits 저장 실패!")
            print(f"    에러: {e}")
            raise


    def save_tensioned_limits(self):
        """
        Tension Limits 대화형 조정 및 저장
        
        이 함수는 다음 순서로 실행됩니다:
        1. Curl Limits 로드 (없으면 자동 측정)
        2. 초기 Tension 위치 추정
        3. 대화형 미세 조정
        4. 결과 저장
        
        저장 경로:
            curl_limits/{hand_type}_tension_limits.npy
        
        파일 형식:
            - Numpy binary format (.npy)
            - Shape: (11,)
            - Dtype: int
            - 값 범위: [10, 4090]
        
        Curl Limits 의존성:
        - Tension 측정은 Curl 측정 후에만 가능
        - Curl 파일이 없으면 자동으로 측정 후 저장
        - Curl → Tension 순서 필수
        
        예시:
            오른손: curl_limits/right_tension_limits.npy
            [2542, 2421, 1794, 2643, 2056, 1887, 2521, 2004, 2489, 1912, 2498]
            
            왼손: curl_limits/left_tension_limits.npy
            [1558, 1679, 2306, 1457, 2044, 2213, 1579, 2096, 1611, 2188, 1602]
        
        주의사항:
        - 대화형 조정은 시간이 걸림 (모터당 30초~2분)
        - 각 모터를 신중하게 조정 필요
        - 잘못 조정하면 reset_motors.py에서 비정상 동작
        - 조정 후 reset_motors.py로 검증 권장
        """
        print("\n" + "="*70)
        print("[Tension Limits 저장]")
        print("="*70)
        
        # Curl Limits 로드 (없으면 자동 측정)
        print(f"\n  Curl Limits 확인 중...")
        
        if os.path.exists(self.curled_path):
            print(f"  ✓ 기존 Curl 파일 발견: {self.curled_path}")
            curled = np.load(self.curled_path)
            print(f"    데이터: {curled}")
        else:
            print(f"  ✗ Curl 파일 없음. 자동 측정을 시작합니다...")
            print(f"\n  ⚠️ 주의: Curl 측정은 5-10분 소요됩니다.")
            
            # Curl Limits 자동 측정
            curled = self.find_curled()
            
            # Curl Limits 저장
            np.save(self.curled_path, curled)
            print(f"\n  ✓ Curl Limits 자동 저장 완료")
            print(f"    파일: {self.curled_path}")
        
        try:
            # 초기 Tension 위치 추정
            t_init = self.estimate_tensioned_from_curled(curled)
            
            # 대화형 미세 조정
            t_refined = self.interactive_refine_tensioned(t_init, step=10)
            
            # 파일 저장
            print(f"\n{'─'*70}")
            print(f"[파일 저장]")
            print(f"  파일 경로: {self.tension_path}")
            np.save(self.tension_path, t_refined)
            
            # 완료 메시지
            print(f"\n  ✓ Tension Limits 저장 완료!")
            print(f"    파일: {self.tension_path}")
            print(f"    크기: {os.path.getsize(self.tension_path)} bytes")
            
            # 저장된 데이터 확인
            print(f"\n  저장된 데이터 확인:")
            loaded = np.load(self.tension_path)
            print(f"    Shape: {loaded.shape}")
            print(f"    Dtype: {loaded.dtype}")
            print(f"    Min: {loaded.min()}, Max: {loaded.max()}")
            print(f"    Mean: {loaded.mean():.1f}, Std: {loaded.std():.1f}")
            
            # Curl과 Tension 차이 분석
            print(f"\n  Curl vs Tension 차이 분석:")
            diff = np.abs(curled - t_refined)
            print(f"    평균 차이: {diff.mean():.1f}")
            print(f"    최소 차이: {diff.min()}")
            print(f"    최대 차이: {diff.max()}")
            print(f"    표준편차: {diff.std():.1f}")
            
        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"[사용자 중단]")
            print(f"  Ctrl+C가 감지되었습니다.")
            print(f"  Tension Limits가 저장되지 않았습니다.")
            print(f"{'='*70}")
            raise
        
        except Exception as e:
            print(f"\n  ✗ Tension Limits 저장 실패!")
            print(f"    에러: {e}")
            raise


# =============================================================================
# 명령줄 인자 파싱 함수
# =============================================================================

def parse_args():
    """
    명령줄 인자를 파싱하는 함수
    
    사용 가능한 인자:
    
    -ht, --hand-type:
        - 로봇 손 종류 선택
        - 선택: "right" 또는 "left"
        - 기본값: "right"
        - 예: --hand-type left
    
    --testing:
        - 디버그 모드 활성화
        - True/False
        - 기본값: True
        - 예: --testing True
    
    --curr-lim:
        - 전류 제한값 (mA)
        - 정수
        - 기본값: 50
        - 예: --curr-lim 100
    
    -m, --mode:
        - 캘리브레이션 모드 선택
        - 선택: "curl", "tension", "both"
        - 기본값: "both"
        - curl: Curl Limits만 측정
        - tension: Tension Limits만 조정
        - both: 둘 다 수행
        - 예: --mode curl
    
    사용 예시:
        # 오른손 전체 캘리브레이션
        python calibrate_motors.py --hand-type right --mode both
        
        # 왼손 Curl만 측정
        python calibrate_motors.py -ht left -m curl
        
        # 오른손 Tension만 조정 (Curl은 기존 파일 사용)
        python calibrate_motors.py -ht right -m tension
        
        # 높은 전류 제한으로 측정
        python calibrate_motors.py --curr-lim 100
        
        # 디버그 모드 비활성화
        python calibrate_motors.py --testing False
    
    Returns:
        argparse.Namespace: 파싱된 인자 객체
    """
    import argparse
    
    # ArgumentParser 생성
    parser = argparse.ArgumentParser(
        description="RUKA Robot Hand Motor Calibration - 로봇 손 모터 캘리브레이션",
        epilog="""
사용 예시:
  python calibrate_motors.py --hand-type right --mode both
  python calibrate_motors.py -ht left -m curl
  python calibrate_motors.py -ht right -m tension --curr-lim 100

캘리브레이션 모드:
  curl    : Curl Limits만 자동 측정 (5-10분 소요)
  tension : Tension Limits만 대화형 조정 (Curl 파일 필요)
  both    : 둘 다 수행 (기본값, 10-20분 소요)

주의사항:
  - 로봇 손이 USB로 연결되어 있어야 합니다
  - 로봇 손 전원이 켜져 있어야 합니다
  - 주변에 장애물이 없어야 합니다
  - Tension 조정 시 신중하게 진행하세요
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # hand-type 인자
    parser.add_argument(
        "-ht",
        "--hand-type",
        type=str,
        default="right",
        choices=["right", "left"],
        help="로봇 손 종류 ('right' 또는 'left'). 기본값: right",
    )
    
    # testing 인자
    parser.add_argument(
        "--testing",
        type=bool,
        default=True,
        help="디버그 출력 활성화 (True/False). 기본값: True",
    )
    
    # curr-lim 인자
    parser.add_argument(
        "--curr-lim",
        type=int,
        default=50,
        help="전류 제한값 (mA 단위). 기본값: 50",
    )
    
    # mode 인자
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["curl", "tension", "both"],
        default="both",
        help="캘리브레이션 모드 ('curl', 'tension', 'both'). 기본값: both",
    )
    
    return parser.parse_args()


# =============================================================================
# 메인 실행 블록
# =============================================================================

if __name__ == "__main__":
    """
    프로그램 엔트리 포인트
    
    실행 흐름:
    1. 명령줄 인자 파싱
    2. 프로젝트 루트 경로 찾기
    3. 저장 디렉토리 생성
    4. HandCalibrator 인스턴스 생성
    5. mode에 따라 캘리브레이션 수행
       - "curl": save_curled_limits()만 실행
       - "tension": save_tensioned_limits()만 실행
       - "both": 둘 다 실행
    6. 안전한 종료
    
    예외 처리:
    - KeyboardInterrupt: Ctrl+C 입력 시
    - 하드웨어 연결 오류
    - 파일 시스템 오류
    - 측정 실패
    """
    print("\n" + "="*70)
    print("RUKA Robot Hand Motor Calibration")
    print("로봇 손 모터 캘리브레이션 프로그램")
    print("="*70)
    print("\nCopyright (c) NYU RUKA Team")
    print("License: MIT License\n")
    
    try:
        # 명령줄 인자 파싱
        print("[단계 1/5] 명령줄 인자 파싱 중...")
        args = parse_args()
        print(f"  ✓ 인자 파싱 완료")
        print(f"    손 종류: {args.hand_type.upper()}")
        print(f"    모드: {args.mode.upper()}")
        print(f"    전류 제한: {args.curr_lim} mA")
        print(f"    디버그: {'ON' if args.testing else 'OFF'}")
        
        # 프로젝트 루트 경로 찾기
        print(f"\n[단계 2/5] 프로젝트 경로 확인 중...")
        repo_root = get_repo_root()
        print(f"  ✓ 프로젝트 루트: {repo_root}")
        
        # 저장 디렉토리 설정 및 생성
        print(f"\n[단계 3/5] 저장 디렉토리 설정 중...")
        save_dir = f"{repo_root}/curl_limits"
        
        if not os.path.exists(save_dir):
            print(f"  → 디렉토리 생성: {save_dir}")
            os.makedirs(save_dir)
        else:
            print(f"  ✓ 기존 디렉토리 사용: {save_dir}")
        
        # HandCalibrator 인스턴스 생성
        print(f"\n[단계 4/5] HandCalibrator 초기화 중...")
        calibrator = HandCalibrator(
            data_save_dir=save_dir,
            hand_type=args.hand_type,
            curr_lim=args.curr_lim,
            testing=args.testing,
        )
        
        # 캘리브레이션 실행
        print(f"\n[단계 5/5] 캘리브레이션 실행 중...")
        print(f"  모드: {args.mode.upper()}")
        
        # mode에 따라 실행
        if args.mode in ("curl", "both"):
            print(f"\n  → Curl Limits 측정 시작...")
            calibrator.save_curled_limits()
        
        if args.mode in ("tension", "both"):
            print(f"\n  → Tension Limits 조정 시작...")
            calibrator.save_tensioned_limits()
        
        # 완료 메시지
        print(f"\n{'='*70}")
        print(f"[캘리브레이션 완료]")
        print(f"{'='*70}")
        print(f"\n  모든 캘리브레이션이 성공적으로 완료되었습니다!")
        print(f"\n  생성된 파일:")
        
        if args.mode in ("curl", "both"):
            if os.path.exists(calibrator.curled_path):
                size = os.path.getsize(calibrator.curled_path)
                print(f"    ✓ {calibrator.curled_path} ({size} bytes)")
        
        if args.mode in ("tension", "both"):
            if os.path.exists(calibrator.tension_path):
                size = os.path.getsize(calibrator.tension_path)
                print(f"    ✓ {calibrator.tension_path} ({size} bytes)")
        
        print(f"\n  다음 단계:")
        print(f"    1. reset_motors.py로 동작 테스트")
        print(f"    2. 손가락이 부드럽게 움직이는지 확인")
        print(f"    3. 이상 발견 시 캘리브레이션 재실행")
        
        print(f"\n  테스트 명령:")
        print(f"    python reset_motors.py --hand_type {args.hand_type}")
        
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"[프로그램 중단]")
        print(f"{'='*70}")
        print(f"\n  Ctrl+C가 감지되었습니다.")
        print(f"  프로그램을 안전하게 종료합니다.")
        print(f"\n  중단된 캘리브레이션은 저장되지 않았습니다.")
        print(f"  다시 실행하여 캘리브레이션을 완료하세요.")
        
    except Exception as e:
        print(f"\n\n{'='*70}")
        print(f"[에러 발생]")
        print(f"{'='*70}")
        print(f"\n  예기치 않은 에러가 발생했습니다:")
        print(f"    {type(e).__name__}: {e}")
        print(f"\n  트러블슈팅:")
        print(f"    1. USB 연결 확인")
        print(f"    2. 로봇 손 전원 확인")
        print(f"    3. 시리얼 포트 권한 확인")
        print(f"    4. 다른 프로그램에서 포트 사용 여부 확인")
        raise
    
    finally:
        # 안전한 종료 처리
        try:
            if 'calibrator' in locals():
                print(f"\n[안전 종료]")
                print(f"  → Hand 객체 종료 중...")
                calibrator.hand.close()
                print(f"  ✓ 모터 토크 비활성화 완료")
                print(f"  ✓ 시리얼 포트 닫기 완료")
        except:
            pass
        
        print(f"\n{'='*70}")
        print(f"프로그램이 종료되었습니다.")
        print(f"{'='*70}\n")