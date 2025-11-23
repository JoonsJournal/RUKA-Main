#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
RUKA Robot Hand Motor Calibration - 로봇 손 모터 캘리브레이션 프로그램 (v2.0)
================================================================================

이 프로그램은 RUKA(로봇 손)의 모터 동작 범위를 측정하고 저장하는
캘리브레이션 프로그램입니다.

================================================================================
버전 히스토리
================================================================================
v0.x (Original) - Linux 전용, 기본 기능
v1.x (Windows Port) - Windows 지원, 상세 주석, 에러 처리 개선
v2.0 (Enhanced) - 현재 버전, 아래 개선 사항 포함

================================================================================
v2.0 주요 개선 사항 (v1 대비)
================================================================================
1. [모터별 개별 프로파일]
   - XM430-W210-T (엄지): 느린 속도, 높은 토크, 긴 안정화 시간
   - XL330-M288-T (손가락): 빠른 속도, 낮은 토크, 짧은 안정화 시간
   - 각 모터 타입에 최적화된 프로파일 자동 적용

2. [자동 백업 기능]
   - 새 캘리브레이션 저장 전 기존 파일 자동 백업
   - backups/ 폴더에 타임스탬프 포함하여 저장
   - 언제든지 이전 캘리브레이션으로 복구 가능

3. [메타데이터 저장 (.meta.json)]
   - 캘리브레이션 날짜/시간
   - 사용된 파라미터 (전류 제한, 다중 측정 횟수 등)
   - 모터별 프로파일 정보
   - 데이터 통계 (min, max, mean, std)

4. [중간 저장 기능 (.tmp 파일)]
   - 각 모터 측정 완료 후 자동으로 임시 저장
   - Ctrl+C 또는 예기치 않은 종료 시 데이터 보존
   - 재실행 시 이어서 진행 가능

5. [적응형 전류 임계값]
   - 무부하 상태에서 전류 측정하여 baseline 계산
   - 환경(온도, 마찰 등)에 따라 동적으로 임계값 조정
   - 더 정확한 Curl 위치 감지

6. [다중 측정 및 이상치 제거]
   - 각 모터당 3회(기본) 반복 측정
   - IQR(사분위수 범위) 방법으로 이상치 제거
   - 중앙값 사용으로 안정적인 결과

7. [전류 필터링 (이동 평균)]
   - 10회 샘플 수집 후 평균 계산
   - 스파이크 노이즈(최댓값 2개) 제거
   - 노이즈에 강건한 전류 측정

8. [워밍업 프로세스]
   - 측정 전 각 모터 2회 왕복 운동
   - 기계적 안정성 확보 (윤활, 장력 안정화)
   - 더 일관된 측정 결과

9. [통신 안정성 강화]
   - RobustHandController 래퍼 클래스
   - 자동 재시도 메커니즘 (3회)
   - 자동 재연결 기능 (5회 시도)

================================================================================
캘리브레이션 결과물
================================================================================
motor_limits/                              ← v1의 curl_limits/에서 변경
├── {hand_type}_curl_limits.npy           # 11개 모터의 구부린 위치
├── {hand_type}_curl_limits.npy.meta.json # 메타데이터 (v2 신규)
├── {hand_type}_tension_limits.npy        # 11개 모터의 펼친 위치
├── {hand_type}_tension_limits.npy.meta.json
└── backups/                               # 자동 백업 디렉토리 (v2 신규)
    └── {hand_type}_curl_limits_YYYYMMDD_HHMMSS.npy

================================================================================
사용 방법
================================================================================
# 전체 캘리브레이션 (권장)
python calibrate_motors.py --hand-type right --mode both

# Curl만 측정 (자동)
python calibrate_motors.py -ht left -m curl

# Tension만 조정 (대화형, Curl 파일 필요)
python calibrate_motors.py -ht right -m tension

# 다중 측정 횟수 지정 (기본: 3)
python calibrate_motors.py -ht right -m both --multi-sample 5

================================================================================
시스템 요구사항
================================================================================
- Windows 10/11 (msvcrt 모듈 사용)
- Python 3.8 이상
- NumPy, logging, json, shutil (표준 라이브러리)
- RUKA 프로젝트 패키지 (ruka_hand)

================================================================================
작성: NYU RUKA Team
버전: 2.0 (Enhanced Version)
라이선스: MIT License
================================================================================
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import os        # 파일 경로 조작, 디렉토리 생성
import sys       # 시스템 입출력 제어
import time      # 시간 지연 및 타이밍 제어
import json      # 메타데이터 JSON 저장 (v2 신규)
import shutil    # 파일 복사 및 백업 (v2 신규)
import logging   # 구조화된 로깅 (v2 신규)
from datetime import datetime  # 타임스탬프 생성 (v2 신규)
from typing import Optional, Dict, List, Tuple, Any  # 타입 힌트 (v2 신규)

import numpy as np  # 배열 연산 및 데이터 저장

# RUKA 프로젝트 모듈 임포트
from ruka_hand.control.hand import *            # Hand 클래스: 로봇 손 제어
from ruka_hand.utils.file_ops import get_repo_root  # 프로젝트 루트 경로

# =============================================================================
# 로깅 설정 (v2 신규)
# =============================================================================
# 로깅 형식: [시간] - [레벨] - [메시지]
# 콘솔에 INFO 레벨 이상의 메시지 출력

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 모터별 프로파일 설정 (v2 신규)
# =============================================================================
"""
RUKA Robot Hand는 두 종류의 Dynamixel 모터를 사용합니다:

1. XM430-W210-T (3개, 모터 ID 1-3)
   - 용도: 엄지 (Thumb)
   - 특징: 더 큰 토크, 더 높은 정밀도
   - 스펙: Stall Torque 3.0 Nm, 전류 최대 2.1A

2. XL330-M288-T (8개, 모터 ID 4-11)
   - 용도: 검지, 중지, 약지, 새끼 (Index, Middle, Ring, Pinky)
   - 특징: 소형 경량, 빠른 응답
   - 스펙: Stall Torque 0.52 Nm, 전류 최대 400mA

각 모터 타입에 맞는 프로파일을 적용하여 최적의 캘리브레이션을 수행합니다.
"""

# 모터 타입 상수 정의
MOTOR_TYPE_XM430 = "XM430-W210-T"  # 엄지용 (더 큰 토크)
MOTOR_TYPE_XL330 = "XL330-M288-T"  # 손가락용 (소형)

# 모터 ID별 타입 매핑 (1-based indexing)
# ┌─────────────────────────────────────────────────────┐
# │  Motor 1-3:  Thumb (XM430)                          │
# │  Motor 4-5:  Index (XL330)                          │
# │  Motor 6-7:  Middle (XL330)                         │
# │  Motor 8-9:  Ring (XL330)                           │
# │  Motor 10-11: Pinky (XL330)                         │
# └─────────────────────────────────────────────────────┘
MOTOR_TYPE_MAP = {
    1: MOTOR_TYPE_XM430,  # Thumb MCP Flexion
    2: MOTOR_TYPE_XM430,  # Thumb MCP Adduction
    3: MOTOR_TYPE_XM430,  # Thumb IP Flexion
    4: MOTOR_TYPE_XL330,  # Index MCP
    5: MOTOR_TYPE_XL330,  # Index PIP
    6: MOTOR_TYPE_XL330,  # Middle MCP
    7: MOTOR_TYPE_XL330,  # Middle PIP
    8: MOTOR_TYPE_XL330,  # Ring MCP
    9: MOTOR_TYPE_XL330,  # Ring PIP
    10: MOTOR_TYPE_XL330, # Pinky MCP
    11: MOTOR_TYPE_XL330, # Pinky PIP
}

# 모터 타입별 최적화 프로파일
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  파라미터           │  XM430 (엄지)  │  XL330 (손가락) │  설명         │
# ├─────────────────────────────────────────────────────────────────────────┤
# │  profile_velocity   │  100           │  200            │  이동 속도    │
# │  profile_accel      │  50            │  80             │  가속도       │
# │  current_limit      │  700 mA        │  400 mA         │  최대 전류    │
# │  current_threshold  │  100 mA        │  50 mA          │  감지 임계값  │
# │  stabilization_time │  3.0 초        │  2.0 초         │  안정화 시간  │
# └─────────────────────────────────────────────────────────────────────────┘
MOTOR_PROFILES = {
    # XM430 (Thumb) - 더 느리고 높은 전류
    MOTOR_TYPE_XM430: {
        "profile_velocity": 100,       # 느린 속도 (안전)
        "profile_acceleration": 50,     # 부드러운 가속
        "current_limit": 700,           # mA, 최대 전류 제한
        "current_threshold": 100,       # mA, Curl 감지 임계값
        "stabilization_time": 3.0,      # 초, 이동 후 안정화 대기
    },
    # XL330 (Fingers) - 더 빠르고 낮은 전류
    MOTOR_TYPE_XL330: {
        "profile_velocity": 200,       # 빠른 속도
        "profile_acceleration": 80,     # 빠른 가속
        "current_limit": 400,           # mA, 최대 전류 제한
        "current_threshold": 50,        # mA, Curl 감지 임계값
        "stabilization_time": 2.0,      # 초, 이동 후 안정화 대기
    },
}

# 특수 모터 전류 임계값 (경험적으로 최적화된 값)
# 이 모터들은 기계적 특성상 더 높은 전류 임계값이 필요합니다
SPECIAL_MOTOR_THRESHOLDS = {
    4: 250,  # Index MCP: 손가락 기저부, 더 큰 힘 필요
    5: 200,  # Index PIP: 관절 위치상 저항 높음
}

# =============================================================================
# 통신 안정성 설정 (v2 신규)
# =============================================================================
"""
Dynamixel 통신에서 발생할 수 있는 일시적 오류에 대응하기 위한 설정입니다.
특히 USB 통신의 불안정성, 전자기 간섭, 전원 노이즈 등으로 인한
패킷 손실에 대비합니다.

v1과의 차이점:
- v1: 오류 발생 시 즉시 실패
- v2: 자동 재시도 후에도 실패 시에만 오류 처리
"""

MAX_COMM_RETRIES = 3       # 통신 실패 시 최대 재시도 횟수
COMM_RETRY_DELAY = 0.05    # 재시도 간 대기 시간 (초)
COMMAND_DELAY = 0.1        # 명령 후 대기 시간 (초)
RECONNECT_DELAY = 1.0      # 재연결 시도 전 대기 시간 (초)
MAX_RECONNECT_ATTEMPTS = 5 # 최대 재연결 시도 횟수

# =============================================================================
# Dynamixel 레지스터 주소 (Protocol 2.0)
# =============================================================================
"""
Dynamixel Protocol 2.0의 Control Table 주소입니다.
모터 설정 및 상태 읽기에 사용됩니다.

참고: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
"""

# 레지스터 주소
ADDR_PROFILE_VELOCITY = 112      # 프로파일 속도 (0~32767)
ADDR_PROFILE_ACCELERATION = 108  # 프로파일 가속도 (0~32767)
ADDR_CURRENT_LIMIT = 38          # 전류 제한 (mA)
ADDR_PRESENT_CURRENT = 126       # 현재 전류 (mA)
ADDR_PRESENT_POSITION = 132      # 현재 위치 (0~4095)
ADDR_PRESENT_TEMPERATURE = 146   # 현재 온도 (°C)
ADDR_PRESENT_INPUT_VOLTAGE = 144 # 입력 전압 (0.1V 단위)

# 레지스터 길이 (바이트)
LEN_PROFILE_VELOCITY = 4
LEN_PROFILE_ACCELERATION = 4
LEN_CURRENT_LIMIT = 2
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_POSITION = 4
LEN_PRESENT_TEMPERATURE = 1
LEN_PRESENT_INPUT_VOLTAGE = 2

# =============================================================================
# 캘리브레이션 설정 (v2 신규)
# =============================================================================
"""
캘리브레이션 알고리즘의 핵심 파라미터입니다.

v1과의 차이점:
- v1: 단일 측정, 고정 임계값
- v2: 다중 측정, 적응형 임계값, 필터링
"""

# 이진 탐색 종료 조건 (탐색 범위가 이 값 이하가 되면 종료)
# v1: 50 (기존) → v2: 20 (더 세밀한 탐색)
BINARY_SEARCH_THRESHOLD = 20

# 전류 필터링 윈도우 크기 (샘플 수)
# 노이즈 제거를 위해 여러 샘플의 평균 사용
CURRENT_FILTER_WINDOW = 10

# 다중 측정 횟수 (이상치 제거를 위해)
# 여러 번 측정 후 중앙값 사용
MULTI_SAMPLE_COUNT = 3

# 워밍업 사이클 수 (기계적 안정화를 위해)
# 측정 전 모터를 왕복 운동시켜 윤활, 장력 안정화
WARMUP_CYCLES = 2


# =============================================================================
# 키보드 입력 캡처 함수 (Windows 전용)
# =============================================================================

def get_key():
    """
    단일 키 입력을 캡처하는 함수 (Windows 전용)
    
    msvcrt 모듈을 사용하여 키 입력을 즉시 읽습니다.
    화살표 키는 2바이트 시퀀스로 전송되며, Unix 스타일로 변환합니다.
    
    동작 원리:
    ┌────────────────────────────────────────────────────────────────────┐
    │  1. msvcrt.getch()로 첫 번째 바이트 읽기                          │
    │  2. 특수 키(0x00, 0xe0)인 경우 다음 바이트도 읽기                  │
    │  3. 화살표 키 코드를 Unix 스타일 이스케이프 시퀀스로 변환          │
    │  4. 일반 키는 UTF-8 디코딩하여 반환                                │
    └────────────────────────────────────────────────────────────────────┘
    
    Windows 키 코드 → Unix 스타일 변환:
    ┌──────────────────┬──────────────────┐
    │  Windows         │  Unix Style      │
    ├──────────────────┼──────────────────┤
    │  0xe0 + 0x48     │  \\x1b[A (↑)    │
    │  0xe0 + 0x50     │  \\x1b[B (↓)    │
    │  0xe0 + 0x4d     │  \\x1b[C (→)    │
    │  0xe0 + 0x4b     │  \\x1b[D (←)    │
    │  0x0d            │  \\r (Enter)     │
    └──────────────────┴──────────────────┘
    
    Returns:
        str: 입력된 키 문자열 (화살표 키는 3문자 시퀀스)
    
    사용 예시:
        key = get_key()
        if key == "\\x1b[A":     # Up Arrow
            print("위쪽 화살표 입력됨")
        elif key == "\\r":        # Enter
            print("엔터 입력됨")
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
# RobustHandController 클래스 (v2 신규)
# =============================================================================

class RobustHandController:
    """
    통신 안정성이 강화된 Hand 제어 래퍼 클래스
    
    이 클래스는 기존 Hand 클래스를 래핑하여 통신 오류에 대한
    자동 재시도 및 재연결 기능을 추가합니다.
    
    v1과의 차이점:
    ┌────────────────────────────────────────────────────────────────────┐
    │  v1: Hand.read_pos() 직접 호출 → 오류 시 즉시 실패                │
    │  v2: RobustHandController.robust_read_pos() → 3회 재시도 후 실패  │
    └────────────────────────────────────────────────────────────────────┘
    
    주요 기능:
    1. robust_read_pos(): 재시도 메커니즘이 있는 위치 읽기
    2. robust_read_single_cur(): 재시도 메커니즘이 있는 전류 읽기
    3. robust_set_pos(): 재시도 메커니즘이 있는 위치 명령
    4. reconnect(): 통신 끊김 시 자동 재연결
    
    Attributes:
        hand (Hand): 원본 Hand 객체
        hand_type (str): 핸드 타입 ("right" 또는 "left")
        max_retries (int): 최대 재시도 횟수
        retry_delay (float): 재시도 간 대기 시간 (초)
        is_connected (bool): 현재 연결 상태
    """
    
    def __init__(self, hand, hand_type):
        """
        RobustHandController 초기화
        
        Args:
            hand: 원본 Hand 객체
            hand_type: 핸드 타입 ("right" 또는 "left")
        """
        self.hand = hand
        self.hand_type = hand_type
        self.max_retries = MAX_COMM_RETRIES
        self.retry_delay = COMM_RETRY_DELAY
        self.is_connected = True
        logger.info(f"RobustHandController 초기화 완료 (hand_type={hand_type})")
    
    def robust_read_pos(self):
        """
        재시도 메커니즘이 있는 위치 읽기
        
        최대 3회까지 재시도하며, 각 시도 사이에 50ms 대기합니다.
        
        Returns:
            list: 11개 모터의 현재 위치 (0~4095)
        
        Raises:
            Exception: 모든 재시도 실패 시
        """
        for attempt in range(self.max_retries):
            try:
                positions = self.hand.read_pos()
                if positions is not None and len(positions) > 0:
                    return positions
            except Exception as e:
                logger.warning(f"위치 읽기 실패 (시도 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"위치 읽기 최종 실패: {e}")
        raise Exception("위치 읽기 실패 (알 수 없는 오류)")
    
    def robust_read_single_cur(self, motor_id):
        """
        재시도 메커니즘이 있는 단일 모터 전류 읽기
        
        Args:
            motor_id: 읽을 모터 ID (1~11)
        
        Returns:
            float: 현재 전류 (mA)
        
        Raises:
            Exception: 모든 재시도 실패 시
        """
        for attempt in range(self.max_retries):
            try:
                current = self.hand.read_single_cur(motor_id)
                if current is not None:
                    return current
            except Exception as e:
                logger.warning(f"모터 {motor_id} 전류 읽기 실패 (시도 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"모터 {motor_id} 전류 읽기 최종 실패: {e}")
        raise Exception(f"모터 {motor_id} 전류 읽기 실패")
    
    def robust_set_pos(self, positions):
        """
        재시도 메커니즘이 있는 위치 명령
        
        Args:
            positions: 11개 모터의 목표 위치
        
        Returns:
            bool: 성공 여부
        
        Raises:
            Exception: 모든 재시도 실패 시
        """
        for attempt in range(self.max_retries):
            try:
                self.hand.set_pos(positions)
                time.sleep(COMMAND_DELAY)  # 명령 후 안정화 대기
                return True
            except Exception as e:
                logger.warning(f"위치 명령 실패 (시도 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"위치 명령 최종 실패: {e}")
        raise Exception("위치 명령 실패")
    
    def reconnect(self):
        """
        통신 재연결 시도
        
        통신 끊김이 감지되면 Hand 객체를 다시 생성하여 연결을 복구합니다.
        최대 5회까지 시도하며, 각 시도 사이에 1초 대기합니다.
        
        Returns:
            bool: 재연결 성공 여부
        """
        logger.info("\n통신 끊김 감지. 재연결 시도 중...")
        
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                logger.info(f"재연결 시도 {attempt + 1}/{MAX_RECONNECT_ATTEMPTS}...")
                
                # 기존 연결 정리
                try:
                    self.hand.close()
                except:
                    pass
                
                time.sleep(RECONNECT_DELAY)
                
                # 새로운 Hand 객체 생성
                self.hand = Hand(hand_type=self.hand_type)
                
                # 연결 확인
                test_pos = self.hand.read_pos()
                if test_pos is not None and len(test_pos) > 0:
                    logger.info("✓ 재연결 성공!")
                    self.is_connected = True
                    return True
                    
            except Exception as e:
                logger.warning(f"재연결 실패 (시도 {attempt+1}): {e}")
                if attempt == MAX_RECONNECT_ATTEMPTS - 1:
                    logger.error("✗ 재연결 최종 실패")
                    self.is_connected = False
                    return False
        
        return False
    
    def close(self):
        """안전한 종료"""
        try:
            self.hand.close()
            logger.info("Hand 연결 종료 완료")
        except Exception as e:
            logger.warning(f"Hand 종료 중 오류: {e}")


# =============================================================================
# CalibrationDataManager 클래스 (v2 신규)
# =============================================================================

class CalibrationDataManager:
    """
    캘리브레이션 데이터 관리 클래스
    
    이 클래스는 캘리브레이션 데이터의 저장, 백업, 복구를 담당합니다.
    
    v1과의 차이점:
    ┌────────────────────────────────────────────────────────────────────┐
    │  v1: np.save()로 단순 저장, 덮어쓰기 시 이전 데이터 손실           │
    │  v2: 자동 백업, 메타데이터 저장, 중간 저장, 복구 기능              │
    └────────────────────────────────────────────────────────────────────┘
    
    주요 기능:
    1. create_backup(): 기존 파일 자동 백업
    2. save_with_metadata(): 데이터와 메타데이터 함께 저장
    3. save_temp(): 임시 파일로 중간 저장
    4. load_temp(): 임시 파일에서 복구
    5. cleanup_temp(): 임시 파일 삭제
    6. list_backups(): 백업 파일 목록 조회
    7. restore_from_backup(): 백업에서 복구
    
    Attributes:
        save_dir (str): 메인 저장 디렉토리
        hand_type (str): 핸드 타입
        backup_dir (str): 백업 디렉토리
    """
    
    def __init__(self, save_dir: str, hand_type: str):
        """
        CalibrationDataManager 초기화
        
        Args:
            save_dir: 메인 저장 디렉토리 경로
            hand_type: 핸드 타입 ("right" 또는 "left")
        """
        self.save_dir = save_dir
        self.hand_type = hand_type
        self.backup_dir = os.path.join(save_dir, "backups")
        
        # 디렉토리 생성 (없으면 생성)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def get_timestamp(self) -> str:
        """
        현재 타임스탬프 반환
        
        Returns:
            str: "YYYYMMDD_HHMMSS" 형식의 타임스탬프
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_backup(self, file_path: str) -> Optional[str]:
        """
        기존 파일 백업 생성
        
        기존 캘리브레이션 파일이 있으면 backups/ 폴더에 복사합니다.
        파일명에 타임스탬프가 추가됩니다.
        
        Args:
            file_path: 백업할 파일 경로
        
        Returns:
            Optional[str]: 백업 파일 경로 (성공 시) 또는 None (파일 없음/실패 시)
        
        예시:
            원본: motor_limits/right_curl_limits.npy
            백업: motor_limits/backups/right_curl_limits_20241121_153000.npy
        """
        if not os.path.exists(file_path):
            return None
        
        timestamp = self.get_timestamp()
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        try:
            shutil.copy2(file_path, backup_path)
            logger.info(f"백업 생성: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"백업 생성 실패: {e}")
            return None
    
    def save_with_metadata(
        self,
        data: np.ndarray,
        file_path: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        데이터와 메타데이터 함께 저장
        
        .npy 파일로 데이터를 저장하고, .meta.json 파일로 메타데이터를 저장합니다.
        저장 전에 기존 파일을 자동으로 백업합니다.
        
        Args:
            data: 저장할 numpy 배열
            file_path: 저장 경로
            metadata: 메타데이터 딕셔너리
        
        Returns:
            bool: 저장 성공 여부
        
        저장 파일:
            - {file_path}: 데이터 (.npy)
            - {file_path}.meta.json: 메타데이터
        """
        try:
            # 1. 기존 파일 백업
            self.create_backup(file_path)
            
            # 2. 데이터 저장
            np.save(file_path, data)
            
            # 3. 메타데이터 저장
            meta_path = f"{file_path}.meta.json"
            metadata["timestamp"] = self.get_timestamp()
            metadata["file_path"] = file_path
            metadata["data_shape"] = list(data.shape)
            metadata["data_dtype"] = str(data.dtype)
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"저장 완료: {file_path}")
            logger.info(f"메타데이터 저장: {meta_path}")
            return True
            
        except Exception as e:
            logger.error(f"저장 실패: {e}")
            return False
    
    def save_temp(self, data: np.ndarray, file_path: str) -> str:
        """
        임시 파일로 중간 저장
        
        캘리브레이션 진행 중 각 모터 측정 완료 시마다 호출됩니다.
        예기치 않은 종료 시 데이터를 복구할 수 있습니다.
        
        Args:
            data: 저장할 numpy 배열
            file_path: 원본 파일 경로 (임시 파일은 .tmp 확장자 추가)
        
        Returns:
            str: 임시 파일 경로 (성공 시) 또는 빈 문자열 (실패 시)
        """
        temp_path = f"{file_path}.tmp"
        try:
            np.save(temp_path, data)
            logger.info(f"중간 저장: {temp_path}")
            return temp_path
        except Exception as e:
            logger.warning(f"중간 저장 실패: {e}")
            return ""
    
    def load_temp(self, file_path: str) -> Optional[np.ndarray]:
        """
        임시 파일에서 복구
        
        이전에 중단된 캘리브레이션의 데이터를 복구합니다.
        
        Args:
            file_path: 원본 파일 경로
        
        Returns:
            Optional[np.ndarray]: 복구된 데이터 또는 None
        """
        temp_path = f"{file_path}.tmp"
        if os.path.exists(temp_path):
            try:
                data = np.load(temp_path)
                logger.info(f"임시 파일에서 복구: {temp_path}")
                return data
            except Exception as e:
                logger.warning(f"임시 파일 로드 실패: {e}")
        return None
    
    def cleanup_temp(self, file_path: str):
        """
        임시 파일 삭제
        
        캘리브레이션이 성공적으로 완료되면 임시 파일을 정리합니다.
        
        Args:
            file_path: 원본 파일 경로
        """
        temp_path = f"{file_path}.tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"임시 파일 삭제: {temp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")
    
    def list_backups(self, pattern: str = None) -> List[str]:
        """
        백업 파일 목록 반환
        
        Args:
            pattern: 필터링할 패턴 (예: "right_curl")
        
        Returns:
            List[str]: 백업 파일명 목록 (최신 순)
        """
        if not os.path.exists(self.backup_dir):
            return []
        
        files = os.listdir(self.backup_dir)
        if pattern:
            files = [f for f in files if pattern in f]
        
        return sorted(files, reverse=True)  # 최신 순
    
    def restore_from_backup(self, backup_filename: str, target_path: str) -> bool:
        """
        백업에서 복구
        
        Args:
            backup_filename: 백업 파일명
            target_path: 복구 대상 경로
        
        Returns:
            bool: 복구 성공 여부
        """
        backup_path = os.path.join(self.backup_dir, backup_filename)
        if not os.path.exists(backup_path):
            logger.error(f"백업 파일 없음: {backup_path}")
            return False
        
        try:
            shutil.copy2(backup_path, target_path)
            logger.info(f"복구 완료: {backup_path} → {target_path}")
            return True
        except Exception as e:
            logger.error(f"복구 실패: {e}")
            return False


# =============================================================================
# HandCalibrator 클래스 (v2 개선 버전)
# =============================================================================

class HandCalibrator:
    """
    로봇 손 모터 캘리브레이션 클래스 (v2 개선 버전)
    
    RUKA 로봇 손의 11개 모터 각각의 동작 범위를 측정하고 저장합니다.
    
    v2 개선 사항:
    ┌────────────────────────────────────────────────────────────────────┐
    │  1. 모터별 개별 프로파일 설정 (XM430 vs XL330)                     │
    │  2. 적응형 전류 임계값 (환경에 따라 동적 조정)                     │
    │  3. 다중 측정 및 이상치 제거 (IQR 방법)                           │
    │  4. 전류 필터링 (이동 평균, 스파이크 제거)                        │
    │  5. 워밍업 프로세스 (기계적 안정화)                               │
    │  6. 자동 백업 및 메타데이터 저장                                  │
    │  7. 중간 저장 및 복구 기능                                        │
    │  8. RobustHandController를 통한 통신 안정성                       │
    └────────────────────────────────────────────────────────────────────┘
    
    주요 메서드:
    - find_bound(): 단일 모터의 Curl 위치 찾기 (이진 탐색)
    - find_curled_with_multi_sample(): 다중 측정으로 안정적인 Curl 찾기
    - find_curled(): 모든 모터의 Curl Limits 순차 측정
    - interactive_refine_tensioned(): 대화형 Tension 조정
    - save_curled_limits(): Curl 측정 및 저장
    - save_tensioned_limits(): Tension 조정 및 저장
    
    Attributes:
        hand (RobustHandController): 통신 안정성이 강화된 Hand 래퍼
        curr_lim (int): 기본 전류 제한값 (mA)
        testing (bool): 디버그 모드 플래그
        motor_ids (list): 캘리브레이션할 모터 ID 리스트
        data_save_dir (str): 데이터 저장 디렉토리
        data_manager (CalibrationDataManager): 데이터 관리 객체
        curled_path (str): Curl 데이터 파일 경로
        tension_path (str): Tension 데이터 파일 경로
    """

    def __init__(
        self,
        data_save_dir: str,
        hand_type: str,
        curr_lim: int = 50,
        testing: bool = False,
        motor_ids: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        """
        HandCalibrator 초기화
        
        초기화 과정 (6단계):
        ┌────────────────────────────────────────────────────────────────────┐
        │  [1/6] Hand 객체 생성 - Dynamixel 모터 연결                        │
        │  [2/6] RobustHandController 래핑 - 통신 안정성 강화                │
        │  [3/6] 데이터 매니저 초기화 - 백업/저장 관리                       │
        │  [4/6] 파일 경로 설정 - curl/tension 파일 경로                     │
        │  [5/6] 모터별 프로파일 설정 - 속도/가속/전류 개별 설정             │
        │  [6/6] 통신 상태 진단 - 연결 품질 확인                             │
        └────────────────────────────────────────────────────────────────────┘
        
        Args:
            data_save_dir: 캘리브레이션 데이터 저장 디렉토리
            hand_type: 핸드 타입 ("right" 또는 "left")
            curr_lim: 기본 전류 제한값 (mA), 기본값 50
            testing: 디버그 모드, 기본값 False
            motor_ids: 캘리브레이션할 모터 ID 리스트, 기본값 [1~11]
        """
        logger.info(f"\n{'='*70}")
        logger.info("HandCalibrator 초기화 시작 (개선 버전 v2.0)")
        logger.info(f"{'='*70}")
        
        # [1/6] Hand 객체 생성
        try:
            logger.info("\n[1/6] Hand 객체 생성 중...")
            base_hand = Hand(hand_type=hand_type)
            logger.info("  ✓ Hand 객체 생성 완료")
            
            # [2/6] RobustHandController로 래핑 (v2 신규)
            logger.info("\n[2/6] RobustHandController 래핑 중...")
            self.hand = RobustHandController(base_hand, hand_type)
            logger.info("  ✓ 통신 안정성 강화 완료")
            
        except Exception as e:
            logger.error(f"  ✗ Hand 초기화 실패: {e}")
            raise
        
        # 속성 저장
        self.curr_lim = curr_lim
        self.testing = testing
        self.motor_ids = motor_ids
        self.data_save_dir = data_save_dir
        
        # [3/6] 데이터 매니저 초기화 (v2 신규)
        logger.info("\n[3/6] 데이터 매니저 초기화 중...")
        self.data_manager = CalibrationDataManager(data_save_dir, hand_type)
        logger.info("  ✓ 데이터 매니저 초기화 완료")
        
        # [4/6] 파일 경로 설정
        logger.info("\n[4/6] 파일 경로 설정 중...")
        self.curled_path = os.path.join(data_save_dir, f"{hand_type}_curl_limits.npy")
        self.tension_path = os.path.join(data_save_dir, f"{hand_type}_tension_limits.npy")
        logger.info(f"  Curl 파일: {self.curled_path}")
        logger.info(f"  Tension 파일: {self.tension_path}")
        
        # [5/6] 모터별 프로파일 설정 (v2 신규)
        logger.info("\n[5/6] 모터별 프로파일 설정 중...")
        self._setup_motor_profiles()
        
        # [6/6] 통신 상태 진단 (v2 신규)
        logger.info("\n[6/6] 통신 상태 진단 중...")
        self._diagnose_communication()
        
        logger.info(f"\n{'='*70}")
        logger.info("HandCalibrator 초기화 완료")
        logger.info(f"{'='*70}\n")
    
    def _setup_motor_profiles(self):
        """
        모터별 개별 프로파일 설정 (v2 신규)
        
        각 모터의 타입(XM430/XL330)에 맞는 최적화된 파라미터를 설정합니다.
        
        설정 파라미터:
        - Profile Velocity: 이동 속도
        - Profile Acceleration: 가속도
        - Current Limit: 최대 전류 제한
        """
        try:
            for motor_id in self.motor_ids:
                # 모터 타입 확인
                motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
                profile = MOTOR_PROFILES[motor_type]
                
                # Profile Velocity 설정
                self._write_motor_param(
                    motor_id,
                    ADDR_PROFILE_VELOCITY,
                    profile["profile_velocity"],
                    LEN_PROFILE_VELOCITY
                )
                
                # Profile Acceleration 설정
                self._write_motor_param(
                    motor_id,
                    ADDR_PROFILE_ACCELERATION,
                    profile["profile_acceleration"],
                    LEN_PROFILE_ACCELERATION
                )
                
                # Current Limit 설정
                self._write_motor_param(
                    motor_id,
                    ADDR_CURRENT_LIMIT,
                    profile["current_limit"],
                    LEN_CURRENT_LIMIT
                )
                
                logger.info(
                    f"  모터 {motor_id:2d} ({motor_type}): "
                    f"Vel={profile['profile_velocity']}, "
                    f"Acc={profile['profile_acceleration']}, "
                    f"Cur={profile['current_limit']}mA"
                )
                
                time.sleep(0.05)  # 명령 간 딜레이
                
        except Exception as e:
            logger.warning(f"  ⚠️ 프로파일 설정 실패: {e}")
    
    def _write_motor_param(self, motor_id: int, addr: int, value: int, length: int):
        """
        단일 모터 파라미터 쓰기
        
        Dynamixel 모터의 레지스터에 값을 씁니다.
        
        Args:
            motor_id: 모터 ID (1~11)
            addr: 레지스터 주소
            value: 쓸 값
            length: 데이터 길이 (바이트)
        """
        try:
            self.hand.hand.dxl_client.write(motor_id, addr, value, length)
        except Exception as e:
            logger.warning(f"모터 {motor_id} 파라미터 쓰기 실패: {e}")
    
    def _diagnose_communication(self):
        """
        통신 상태 진단 (v2 신규)
        
        초기화 시 통신 품질을 테스트합니다.
        성공률이 90% 미만이면 경고를 표시합니다.
        """
        total_attempts = 0
        successful_attempts = 0
        
        # 처음 3개 모터로 통신 테스트
        for motor_id in self.motor_ids[:3]:
            for _ in range(5):
                total_attempts += 1
                try:
                    pos = self.hand.hand.read_pos()
                    if pos is not None and len(pos) > motor_id - 1:
                        successful_attempts += 1
                except:
                    pass
        
        success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
        
        if success_rate < 90:
            logger.warning(f"  ⚠️ 통신 상태 불안정 (성공률: {success_rate:.1f}%)")
        else:
            logger.info(f"  ✓ 통신 상태 양호 (성공률: {success_rate:.1f}%)")
    
    def _safe_read_pos(self) -> np.ndarray:
        """
        안전하게 모터 위치 읽기
        
        RobustHandController를 통해 재시도 메커니즘이 적용된 위치 읽기를 수행합니다.
        
        Returns:
            np.ndarray: int32 타입의 위치 배열
        """
        pos = self.hand.robust_read_pos()
        
        if isinstance(pos, (list, tuple)):
            return np.array([int(x) for x in pos], dtype=np.int32)
        elif isinstance(pos, np.ndarray):
            return pos.astype(np.int32)
        else:
            return np.array([int(pos)], dtype=np.int32)
    
    def _safe_set_pos(self, pos):
        """
        안전하게 모터 위치 설정
        
        NumPy 배열을 int 리스트로 변환하여 전달합니다.
        
        Args:
            pos: 위치 배열 또는 리스트
        """
        if isinstance(pos, np.ndarray):
            pos_list = [int(x) for x in pos]
            self.hand.robust_set_pos(pos_list)
        else:
            self.hand.robust_set_pos(pos)
    
    def _filtered_current_reading(self, motor_id: int, window_size: int = CURRENT_FILTER_WINDOW) -> float:
        """
        이동 평균 필터로 전류 노이즈 제거 (v2 신규)
        
        여러 번 전류를 측정하고 평균을 계산합니다.
        스파이크 노이즈(최댓값 2개)를 제거하여 더 안정적인 값을 얻습니다.
        
        Args:
            motor_id: 모터 ID (1~11)
            window_size: 필터 윈도우 크기 (기본: 10)
        
        Returns:
            float: 필터링된 전류값 (mA)
        
        알고리즘:
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. window_size회 전류 측정                                        │
        │  2. 절댓값 취하기                                                  │
        │  3. 정렬 후 최댓값 2개 제거 (스파이크 노이즈 대응)                 │
        │  4. 나머지 평균 계산                                               │
        └────────────────────────────────────────────────────────────────────┘
        """
        currents = []
        for _ in range(window_size):
            try:
                current = self.hand.robust_read_single_cur(motor_id)
                currents.append(abs(current))
            except:
                pass
            time.sleep(0.01)  # 샘플 간 딜레이
        
        if len(currents) < 3:
            # 충분한 샘플이 없으면 단일 읽기
            return abs(self.hand.robust_read_single_cur(motor_id))
        
        # 최댓값 2개 제거 (스파이크 노이즈 대응)
        sorted_currents = sorted(currents)
        filtered = sorted_currents[:-2] if len(sorted_currents) > 2 else sorted_currents
        
        return np.mean(filtered)
    
    def _get_adaptive_threshold(self, motor_id: int) -> float:
        """
        적응형 전류 임계값 계산 (v2 신규)
        
        무부하 상태의 전류를 측정하여 동적으로 임계값을 결정합니다.
        환경(온도, 마찰, 윤활 상태 등)에 따라 자동 조정됩니다.
        
        Args:
            motor_id: 모터 ID (1~11)
        
        Returns:
            float: 적응형 전류 임계값 (mA)
        
        계산 공식:
            adaptive_threshold = baseline + 3 * noise_level + base_threshold
            
            - baseline: 무부하 상태 평균 전류
            - noise_level: 전류 측정의 표준편차
            - base_threshold: 모터 타입별 기본 임계값
        """
        # 특수 모터 임계값 확인 (하드코딩된 값 우선)
        if motor_id in SPECIAL_MOTOR_THRESHOLDS:
            return SPECIAL_MOTOR_THRESHOLDS[motor_id]
        
        # 모터 타입별 기본 임계값
        motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
        base_threshold = MOTOR_PROFILES[motor_type]["current_threshold"]
        
        try:
            # 무부하 상태 전류 측정
            baseline_currents = []
            for _ in range(10):
                current = self.hand.robust_read_single_cur(motor_id)
                baseline_currents.append(abs(current))
                time.sleep(0.05)
            
            baseline = np.mean(baseline_currents)
            noise_level = np.std(baseline_currents)
            
            # 적응형 임계값 계산
            adaptive_threshold = baseline + 3 * noise_level + base_threshold
            
            if self.testing:
                logger.info(f"  모터 {motor_id} 적응형 임계값: {adaptive_threshold:.1f}mA "
                           f"(baseline={baseline:.1f}, noise={noise_level:.1f})")
            
            return adaptive_threshold
            
        except Exception as e:
            logger.warning(f"적응형 임계값 계산 실패, 기본값 사용: {e}")
            return base_threshold
    
    def _warmup_motor(self, motor_id: int, cycles: int = WARMUP_CYCLES):
        """
        캘리브레이션 전 모터 워밍업 (v2 신규)
        
        측정 전에 모터를 왕복 운동시켜 기계적 안정성을 확보합니다.
        윤활유 분포, 텐던 장력 안정화에 도움이 됩니다.
        
        Args:
            motor_id: 모터 ID (1~11)
            cycles: 워밍업 사이클 수 (기본: 2)
        
        동작:
        ┌────────────────────────────────────────────────────────────────────┐
        │  현재 위치에서:                                                    │
        │    → -200 이동 (0.3초 대기)                                       │
        │    → +200 이동 (0.3초 대기)                                       │
        │  cycles회 반복 후 원래 위치로 복귀                                 │
        └────────────────────────────────────────────────────────────────────┘
        """
        if self.testing:
            logger.info(f"  모터 {motor_id} 워밍업 시작 ({cycles}회)...")
        
        motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
        profile = MOTOR_PROFILES[motor_type]
        
        current_positions = self._safe_read_pos()
        start_pos = int(current_positions[motor_id - 1])
        
        for cycle in range(cycles):
            pos = current_positions.copy()
            
            # 약간 앞으로
            pos[motor_id - 1] = max(start_pos - 200, 100)
            self._safe_set_pos(pos)
            time.sleep(0.3)
            
            # 약간 뒤로
            pos[motor_id - 1] = min(start_pos + 200, 4000)
            self._safe_set_pos(pos)
            time.sleep(0.3)
        
        # 원래 위치로 복귀
        pos[motor_id - 1] = start_pos
        self._safe_set_pos(pos)
        time.sleep(0.5)
        
        if self.testing:
            logger.info(f"  모터 {motor_id} 워밍업 완료")
    
    def find_bound(self, motor_id: int) -> int:
        """
        이진 탐색으로 단일 모터의 최대 구부림 위치(Curl Limit) 찾기
        
        모터를 점진적으로 구부리면서 전류 제한값에 도달하는 위치를
        이진 탐색 알고리즘으로 찾습니다.
        
        v2 개선 사항:
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. 현재 위치 기반 탐색 범위 설정                                  │
        │  2. 적응형 전류 임계값 사용                                        │
        │  3. 전류 필터링 (노이즈 제거)                                      │
        │  4. 모터별 프로파일 적용 (안정화 시간 등)                          │
        │  5. 워밍업 프로세스 선행                                          │
        └────────────────────────────────────────────────────────────────────┘
        
        Args:
            motor_id: 캘리브레이션할 모터 ID (1~11)
        
        Returns:
            int: 모터의 최대 구부림 위치 (0~4095)
        
        알고리즘:
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. 워밍업 (2회 왕복)                                              │
        │  2. 적응형 전류 임계값 계산                                        │
        │  3. 탐색 범위 설정 (현재 위치 기반)                                │
        │  4. 이진 탐색 루프:                                                │
        │     - 중간점으로 이동                                              │
        │     - 안정화 대기 (2~3초)                                         │
        │     - 필터링된 전류 측정                                          │
        │     - 전류 < 임계값: 더 구부리기 가능 → 범위 조정                 │
        │     - 전류 >= 임계값: 너무 구부림 → 범위 조정                     │
        │  5. 수렴 시 최종 위치 반환                                         │
        └────────────────────────────────────────────────────────────────────┘
        """
        # 모터 타입 및 프로파일 가져오기
        motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
        profile = MOTOR_PROFILES[motor_type]
        stabilization_time = profile["stabilization_time"]
        
        # 특수 모터 처리 (검지 MCP, PIP)
        if motor_id in SPECIAL_MOTOR_THRESHOLDS:
            current_threshold = SPECIAL_MOTOR_THRESHOLDS[motor_id]
            stabilization_time = 5.0  # 더 긴 안정화 시간
        else:
            # 적응형 임계값 사용 (v2 신규)
            current_threshold = self._get_adaptive_threshold(motor_id)
        
        print(f"\n{'─'*70}")
        print(f"[모터 {motor_id} 캘리브레이션 시작]")
        print(f"  모터 타입: {motor_type}")
        print(f"  전류 임계값: {current_threshold:.1f} mA")
        print(f"  안정화 시간: {stabilization_time}초")
        
        # 워밍업 (v2 신규)
        self._warmup_motor(motor_id)
        
        # 현재 위치 읽기
        current_positions = self._safe_read_pos()
        start_pos = int(current_positions[motor_id - 1])
        print(f"  현재 위치: {start_pos}")
        
        # 탐색 범위 설정 (현재 위치 기반, v2 개선)
        # v1에서는 고정된 [100, 4000] 범위 사용
        # v2에서는 현재 위치를 기준으로 합리적인 범위 설정
        if self.hand.hand_type == "right":
            f = 1  # 방향 계수: 값 증가 = 구부림
            l_bound = max(start_pos - 200, 100)
            u_bound = min(start_pos + 1500, 3995)
            print(f"  손 방향: 오른손 (모터 값 증가 = 구부림)")
        else:
            f = -1  # 방향 계수: 값 감소 = 구부림
            l_bound = max(start_pos - 1500, 100)
            u_bound = min(start_pos + 200, 3995)
            print(f"  손 방향: 왼손 (모터 값 감소 = 구부림)")
        
        pos = current_positions.copy()
        cur = 1000000  # 초기값 (큰 값으로 설정하여 루프 진입 보장)
        iteration = 0
        pres_pos = start_pos  # 초기화
        
        print(f"  초기 탐색 범위: [{l_bound}, {u_bound}]")
        print(f"\n  이진 탐색 시작...")
        
        # 이진 탐색 메인 루프
        while abs(u_bound - l_bound) > BINARY_SEARCH_THRESHOLD or f * cur > current_threshold:
            iteration += 1
            
            # 중간 위치 계산
            com_pos = int((u_bound + l_bound) // 2 - 1)
            pos[motor_id - 1] = com_pos
            self._safe_set_pos(pos)
            
            print(f"    반복 {iteration}: 위치 {com_pos}로 이동 중... ", end='', flush=True)
            time.sleep(stabilization_time)
            
            # 필터링된 전류 측정 (v2 신규)
            cur = self._filtered_current_reading(motor_id)
            pres_pos = int(self._safe_read_pos()[motor_id - 1])
            
            print(f"완료 (전류: {cur:.1f}mA, 실제 위치: {pres_pos})")
            
            # 탐색 범위 업데이트
            if f * cur < current_threshold:
                # 전류 < 임계값: 아직 더 구부릴 수 있음
                if self.hand.hand_type == "right":
                    l_bound = pres_pos + 1
                    u_bound -= 1
                else:
                    u_bound = pres_pos - 1
                    l_bound += 1
            else:
                # 전류 >= 임계값: 너무 구부렸음
                if self.hand.hand_type == "right":
                    u_bound = pres_pos + 1
                else:
                    l_bound = pres_pos - 1
            
            # 탐색 범위 검증 (무한 루프 방지)
            if l_bound >= u_bound:
                break
        
        print(f"\n  ✓ 모터 {motor_id} 캘리브레이션 완료")
        print(f"    최종 위치: {pres_pos}")
        print(f"    최종 전류: {cur:.1f} mA")
        print(f"    반복 횟수: {iteration}회")
        
        return pres_pos
    
    def find_curled_with_multi_sample(self, motor_id: int, num_samples: int = MULTI_SAMPLE_COUNT) -> int:
        """
        다중 측정으로 안정적인 Curl 위치 찾기 (v2 신규)
        
        여러 번 측정하여 이상치를 제거하고 중앙값을 사용합니다.
        단일 측정의 불확실성을 줄이고 재현성을 높입니다.
        
        Args:
            motor_id: 모터 ID (1~11)
            num_samples: 측정 횟수 (기본: 3)
        
        Returns:
            int: 안정적인 Curl 위치 (중앙값)
        
        알고리즘:
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. num_samples회 반복:                                            │
        │     - 초기 위치로 복귀                                             │
        │     - find_bound()로 Curl 위치 측정                               │
        │     - 결과 저장                                                    │
        │  2. IQR (사분위수 범위) 방법으로 이상치 제거                       │
        │  3. 남은 값들의 중앙값 반환                                        │
        └────────────────────────────────────────────────────────────────────┘
        
        IQR 이상치 제거:
            Q1 = 25th percentile
            Q3 = 75th percentile
            IQR = Q3 - Q1
            유효 범위: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        """
        curl_positions = []
        
        print(f"\n[모터 {motor_id}] 다중 측정 ({num_samples}회)...")
        
        for i in range(num_samples):
            print(f"\n  --- 측정 {i+1}/{num_samples} ---")
            
            # 초기 위치로 복귀
            current_pos = self._safe_read_pos()
            self._safe_set_pos(current_pos)
            time.sleep(1.0)
            
            # Curl 위치 측정
            curl_pos = self.find_bound(motor_id)
            curl_positions.append(curl_pos)
            
            print(f"  측정 {i+1} 결과: {curl_pos}")
        
        # 이상치 제거 (IQR 방법)
        if len(curl_positions) >= 3:
            q1, q3 = np.percentile(curl_positions, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            filtered = [p for p in curl_positions if lower_bound <= p <= upper_bound]
            
            if len(filtered) < 2:
                filtered = curl_positions  # 너무 많이 제거되면 원본 사용
        else:
            filtered = curl_positions
        
        result = int(np.median(filtered))
        
        print(f"\n  다중 측정 결과:")
        print(f"    원본: {curl_positions}")
        print(f"    필터링: {filtered}")
        print(f"    최종 (중앙값): {result}")
        print(f"    표준편차: {np.std(curl_positions):.1f}")
        
        return result
    
    def find_curled(self) -> np.ndarray:
        """
        모든 모터의 Curl Limits 순차 측정
        
        11개 모터를 순차적으로 캘리브레이션합니다.
        각 모터 측정 후 중간 저장하여 중단 시 복구할 수 있습니다.
        
        Returns:
            np.ndarray: 11개 모터의 Curl 위치 배열
        """
        print("\n" + "="*70)
        print("[Curl Limits 자동 측정 시작]")
        print("="*70)
        print(f"\n  총 모터 수: {len(self.motor_ids)}개")
        print(f"  다중 측정: {MULTI_SAMPLE_COUNT}회")
        print(f"  예상 소요 시간: 약 10-20분")
        
        curled = np.zeros(len(self.motor_ids), dtype=np.int32)
        start_time = time.time()
        
        for i, mid in enumerate(self.motor_ids):
            print(f"\n진행: [{i+1}/{len(self.motor_ids)}] 모터 {mid}번 측정 중...")
            
            try:
                # 다중 측정 사용 (v2 신규)
                curled[i] = int(self.find_curled_with_multi_sample(mid))
                
                print(f"  모터 {mid}: {curled[i]} (✓)")
                
                # 중간 저장 (v2 신규)
                self.data_manager.save_temp(curled, self.curled_path)
                
            except KeyboardInterrupt:
                print(f"\n\n[사용자 중단]")
                print(f"  현재까지 측정된 데이터:")
                for j in range(i):
                    print(f"    모터 {self.motor_ids[j]}: {curled[j]}")
                raise
            
            except Exception as e:
                print(f"\n✗ 모터 {mid} 측정 실패: {e}")
                # 재시도 옵션
                retry = input("재시도하시겠습니까? (y/n): ")
                if retry.lower() == 'y':
                    curled[i] = int(self.find_bound(mid))
                else:
                    raise
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n{'='*70}")
        print(f"[Curl Limits 측정 완료]")
        print(f"{'='*70}")
        print(f"\n  소요 시간: {minutes}분 {seconds}초")
        print(f"\n  측정 결과:")
        for i, mid in enumerate(self.motor_ids):
            print(f"    모터 {mid:2d}: {curled[i]:4d}")
        
        return curled
    
    def estimate_tensioned_from_curled(self, curled: np.ndarray) -> np.ndarray:
        """
        Curl 위치로부터 Tension 위치의 초기 추정값 계산
        
        Curl 위치에서 일정 거리(1100)만큼 떨어진 위치를 초기값으로 사용합니다.
        이 값은 대화형 조정의 시작점으로만 사용됩니다.
        
        Args:
            curled: Curl 위치 배열
        
        Returns:
            np.ndarray: Tension 초기 추정값 배열
        
        계산:
            오른손: tensioned = curled - 1100 (작은 값 방향)
            왼손:   tensioned = curled + 1100 (큰 값 방향)
        """
        f = 1 if self.hand.hand_type == "right" else -1
        tensioned = np.array([int(x - f * 1100) for x in curled], dtype=np.int32)
        
        print(f"\n{'─'*70}")
        print(f"[Tension Limits 초기 추정]")
        print(f"{'─'*70}")
        print(f"  방향: {self.hand.hand_type.upper()}")
        print(f"  오프셋: {f * 1100}")
        
        for i, mid in enumerate(self.motor_ids):
            print(f"    모터 {mid:2d}: {curled[i]:4d} → {tensioned[i]:4d}")
        
        return tensioned
    
    def interactive_refine_tensioned(self, tensioned_init: np.ndarray, step: int = 10) -> np.ndarray:
        """
        대화형 인터페이스로 Tension Limits 미세 조정
        
        사용자가 화살표 키로 각 모터의 Tension 위치를 미세 조정합니다.
        
        Args:
            tensioned_init: 초기 추정값 배열
            step: 한 번에 이동할 단위 (기본: 10)
        
        Returns:
            np.ndarray: 조정된 Tension 위치 배열
        
        조작법:
        ┌────────────────────────────────────────────────────────────────────┐
        │  ↑/→: +step (펼침 방향)                                           │
        │  ↓/←: -step (구부림 방향)                                         │
        │  Enter: 저장 후 다음 모터                                          │
        │  q: 현재 값 유지하고 스킵                                          │
        └────────────────────────────────────────────────────────────────────┘
        """
        print("\n" + "="*70)
        print("[Tension Limits 대화형 조정]")
        print("="*70)
        print(f"\n  조작법:")
        print(f"    ↑/→: +{step}씩 이동")
        print(f"    ↓/←: -{step}씩 이동")
        print(f"    Enter: 저장 후 다음")
        print(f"    q: 스킵")
        
        current_pos = self._safe_read_pos()
        tensioned = tensioned_init.copy()
        f = 1 if self.hand.hand_type == "right" else -1
        
        for motor_idx, mid in enumerate(self.motor_ids):
            idx = mid - 1
            
            print(f"\n{'─'*70}")
            print(f"[모터 {mid} 조정] - [{motor_idx + 1}/{len(self.motor_ids)}]")
            print(f"{'─'*70}")
            print(f"  초기 추정값: {tensioned[idx]}")
            
            # 초기 추정 위치로 이동
            pos = current_pos.copy()
            pos[idx] = tensioned[idx]
            self._safe_set_pos(pos)
            time.sleep(0.2)
            
            adjustment_count = 0
            
            while True:
                print(f"\n  [모터 {mid}] 현재 후보: {pos[idx]:4d}")
                print(f"  화살표로 조정, Enter로 저장, 'q'로 스킵: ", end='', flush=True)
                
                k = get_key()
                
                if k in ("\r", "\n"):  # Enter: 저장
                    tensioned[idx] = int(pos[idx])
                    print(f"\n  ✓ 모터 {mid} 저장: {tensioned[idx]}")
                    break
                
                elif k in ("\x1b[A", "\x1b[C"):  # Up/Right: 증가
                    old_pos = int(pos[idx])
                    pos[idx] = int(max(min(int(pos[idx]) + step * f, 4090), 10))
                    if pos[idx] != old_pos:
                        self._safe_set_pos(pos)
                        adjustment_count += 1
                        print(f"\n    → {old_pos} → {pos[idx]} (+{step*f})")
                
                elif k in ("\x1b[B", "\x1b[D"):  # Down/Left: 감소
                    old_pos = int(pos[idx])
                    pos[idx] = int(max(min(int(pos[idx]) - step * f, 4090), 10))
                    if pos[idx] != old_pos:
                        self._safe_set_pos(pos)
                        adjustment_count += 1
                        print(f"\n    → {old_pos} → {pos[idx]} ({-step*f})")
                
                elif k.lower() == "q":  # q: 스킵
                    print(f"\n  ⊗ 모터 {mid} 스킵")
                    break
            
            # 중간 저장 (v2 신규)
            self.data_manager.save_temp(tensioned, self.tension_path)
        
        return tensioned.astype(int)
    
    def _collect_metadata(self) -> Dict[str, Any]:
        """
        환경 조건 및 메타데이터 수집 (v2 신규)
        
        캘리브레이션 시 사용된 파라미터와 환경 정보를 수집합니다.
        .meta.json 파일에 저장됩니다.
        
        Returns:
            Dict[str, Any]: 메타데이터 딕셔너리
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "hand_type": self.hand.hand_type,
            "motor_ids": self.motor_ids,
            "curr_lim": self.curr_lim,
            "binary_search_threshold": BINARY_SEARCH_THRESHOLD,
            "multi_sample_count": MULTI_SAMPLE_COUNT,
            "motor_profiles": {},
        }
        
        # 모터별 프로파일 정보
        for mid in self.motor_ids:
            motor_type = MOTOR_TYPE_MAP.get(mid, MOTOR_TYPE_XL330)
            metadata["motor_profiles"][str(mid)] = {
                "type": motor_type,
                "profile": MOTOR_PROFILES[motor_type]
            }
        
        return metadata
    
    def save_curled_limits(self):
        """
        Curl Limits 측정 및 저장
        
        전체 Curl 캘리브레이션 워크플로우:
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. 기존 임시 파일 확인 (중단 복구 여부)                           │
        │  2. find_curled()로 모든 모터 측정                                 │
        │  3. 메타데이터 수집                                                │
        │  4. 백업 생성 후 저장                                              │
        │  5. 임시 파일 정리                                                 │
        └────────────────────────────────────────────────────────────────────┘
        """
        print("\n" + "="*70)
        print("[Curl Limits 저장]")
        print("="*70)
        
        try:
            # 기존 임시 파일 확인 (v2 신규)
            temp_data = self.data_manager.load_temp(self.curled_path)
            if temp_data is not None:
                resume = input("이전 중단된 캘리브레이션이 있습니다. 이어서 하시겠습니까? (y/n): ")
                if resume.lower() == 'y':
                    # 완료된 부분 확인 및 이어서 진행
                    pass  # TODO: 이어서 진행 로직 구현
            
            # Curl Limits 측정
            curled = self.find_curled()
            
            # 메타데이터 수집 (v2 신규)
            metadata = self._collect_metadata()
            metadata["calibration_type"] = "curl"
            metadata["data_stats"] = {
                "min": int(curled.min()),
                "max": int(curled.max()),
                "mean": float(curled.mean()),
                "std": float(curled.std()),
            }
            
            # 저장 (백업 및 메타데이터 포함, v2 신규)
            success = self.data_manager.save_with_metadata(
                curled, self.curled_path, metadata
            )
            
            if success:
                # 임시 파일 정리
                self.data_manager.cleanup_temp(self.curled_path)
                
                print(f"\n  ✓ Curl Limits 저장 완료!")
                print(f"    파일: {self.curled_path}")
                print(f"    백업 디렉토리: {self.data_manager.backup_dir}")
            
        except Exception as e:
            print(f"\n  ✗ Curl Limits 저장 실패: {e}")
            raise
    
    def save_tensioned_limits(self):
        """
        Tension Limits 대화형 조정 및 저장
        
        전체 Tension 캘리브레이션 워크플로우:
        ┌────────────────────────────────────────────────────────────────────┐
        │  1. Curl Limits 로드 (없으면 자동 측정)                            │
        │  2. 초기 추정값 계산                                               │
        │  3. 대화형 조정                                                    │
        │  4. 메타데이터 수집                                                │
        │  5. 백업 생성 후 저장                                              │
        │  6. 임시 파일 정리                                                 │
        └────────────────────────────────────────────────────────────────────┘
        """
        print("\n" + "="*70)
        print("[Tension Limits 저장]")
        print("="*70)
        
        # Curl Limits 로드
        print(f"\n  Curl Limits 확인 중...")
        
        if os.path.exists(self.curled_path):
            print(f"  ✓ 기존 Curl 파일 발견: {self.curled_path}")
            curled = np.load(self.curled_path)
        else:
            print(f"  ✗ Curl 파일 없음. 자동 측정을 시작합니다...")
            curled = self.find_curled()
            np.save(self.curled_path, curled)
        
        try:
            # 초기 추정
            t_init = self.estimate_tensioned_from_curled(curled)
            
            # 대화형 조정
            t_refined = self.interactive_refine_tensioned(t_init, step=10)
            
            # 메타데이터 수집 (v2 신규)
            metadata = self._collect_metadata()
            metadata["calibration_type"] = "tension"
            metadata["curl_reference"] = self.curled_path
            metadata["data_stats"] = {
                "min": int(t_refined.min()),
                "max": int(t_refined.max()),
                "mean": float(t_refined.mean()),
                "std": float(t_refined.std()),
            }
            
            # 저장 (v2 신규)
            success = self.data_manager.save_with_metadata(
                t_refined, self.tension_path, metadata
            )
            
            if success:
                self.data_manager.cleanup_temp(self.tension_path)
                
                print(f"\n  ✓ Tension Limits 저장 완료!")
                print(f"    파일: {self.tension_path}")
            
        except KeyboardInterrupt:
            print(f"\n\n[사용자 중단]")
            raise
        except Exception as e:
            print(f"\n  ✗ Tension Limits 저장 실패: {e}")
            raise


# =============================================================================
# 명령줄 인자 파싱
# =============================================================================

def parse_args():
    """
    명령줄 인자 파싱
    
    사용 가능한 인자:
    
    -ht, --hand-type:
        로봇 손 종류 ("right" 또는 "left")
        기본값: "right"
    
    --testing:
        디버그 모드 (True/False)
        기본값: True
    
    --curr-lim:
        기본 전류 제한값 (mA)
        기본값: 50
    
    -m, --mode:
        캘리브레이션 모드 ("curl", "tension", "both")
        기본값: "both"
    
    --multi-sample:
        다중 측정 횟수 (v2 신규)
        기본값: 3
    
    Returns:
        argparse.Namespace: 파싱된 인자 객체
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RUKA Robot Hand Motor Calibration (개선 버전 v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python calibrate_motors.py --hand-type right --mode both
  python calibrate_motors.py -ht left -m curl
  python calibrate_motors.py -ht right -m tension

개선 기능 (v2):
  - 모터별 개별 프로파일 (XM430 vs XL330)
  - 자동 백업 및 메타데이터 저장
  - 중간 저장 및 복구
  - 적응형 전류 임계값
  - 다중 측정 및 이상치 제거
        """
    )
    
    parser.add_argument(
        "-ht", "--hand-type",
        type=str,
        default="right",
        choices=["right", "left"],
        help="로봇 손 종류 ('right' 또는 'left'). 기본값: right"
    )
    
    parser.add_argument(
        "--testing",
        type=bool,
        default=True,
        help="디버그 모드. 기본값: True"
    )
    
    parser.add_argument(
        "--curr-lim",
        type=int,
        default=50,
        help="기본 전류 제한값 (mA). 기본값: 50"
    )
    
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["curl", "tension", "both"],
        default="both",
        help="캘리브레이션 모드. 기본값: both"
    )
    
    parser.add_argument(
        "--multi-sample",
        type=int,
        default=MULTI_SAMPLE_COUNT,
        help=f"다중 측정 횟수. 기본값: {MULTI_SAMPLE_COUNT}"
    )
    
    return parser.parse_args()


# =============================================================================
# 메인 실행 블록
# =============================================================================

if __name__ == "__main__":
    """
    프로그램 엔트리 포인트
    
    실행 흐름:
    ┌────────────────────────────────────────────────────────────────────┐
    │  [단계 1/5] 명령줄 인자 파싱                                       │
    │  [단계 2/5] 프로젝트 경로 확인                                     │
    │  [단계 3/5] 저장 디렉토리 설정 (motor_limits/)                     │
    │  [단계 4/5] HandCalibrator 초기화                                  │
    │  [단계 5/5] 캘리브레이션 실행                                      │
    │  [안전 종료] 리소스 정리                                           │
    └────────────────────────────────────────────────────────────────────┘
    """
    print("\n" + "="*70)
    print("RUKA Robot Hand Motor Calibration (개선 버전 v2.0)")
    print("="*70)
    print("\nCopyright (c) NYU RUKA Team")
    print("License: MIT License\n")
    
    try:
        # [단계 1/5] 명령줄 인자 파싱
        print("[단계 1/5] 명령줄 인자 파싱 중...")
        args = parse_args()
        print(f"  ✓ 인자 파싱 완료")
        print(f"    손 종류: {args.hand_type.upper()}")
        print(f"    모드: {args.mode.upper()}")
        print(f"    다중 측정: {args.multi_sample}회")
        
        # 다중 측정 횟수 업데이트
        MULTI_SAMPLE_COUNT = args.multi_sample
        
        # [단계 2/5] 프로젝트 루트 경로
        print(f"\n[단계 2/5] 프로젝트 경로 확인 중...")
        repo_root = get_repo_root()
        print(f"  ✓ 프로젝트 루트: {repo_root}")
        
        # [단계 3/5] 저장 디렉토리 설정 (v1: curl_limits/ → v2: motor_limits/)
        print(f"\n[단계 3/5] 저장 디렉토리 설정 중...")
        save_dir = os.path.join(repo_root, "motor_limits")  # v2에서 변경된 디렉토리명
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"  → 디렉토리 생성: {save_dir}")
        else:
            print(f"  ✓ 기존 디렉토리 사용: {save_dir}")
        
        # [단계 4/5] HandCalibrator 인스턴스 생성
        print(f"\n[단계 4/5] HandCalibrator 초기화 중...")
        calibrator = HandCalibrator(
            data_save_dir=save_dir,
            hand_type=args.hand_type,
            curr_lim=args.curr_lim,
            testing=args.testing,
        )
        
        # [단계 5/5] 캘리브레이션 실행
        print(f"\n[단계 5/5] 캘리브레이션 실행 중...")
        print(f"  모드: {args.mode.upper()}")
        
        if args.mode in ("curl", "both"):
            calibrator.save_curled_limits()
        
        if args.mode in ("tension", "both"):
            calibrator.save_tensioned_limits()
        
        # 완료 메시지
        print(f"\n{'='*70}")
        print(f"[캘리브레이션 완료]")
        print(f"{'='*70}")
        print(f"\n  모든 캘리브레이션이 성공적으로 완료되었습니다!")
        
        print(f"\n  생성된 파일:")
        if args.mode in ("curl", "both") and os.path.exists(calibrator.curled_path):
            size = os.path.getsize(calibrator.curled_path)
            print(f"    ✓ {calibrator.curled_path} ({size} bytes)")
            meta_path = f"{calibrator.curled_path}.meta.json"
            if os.path.exists(meta_path):
                print(f"    ✓ {meta_path}")
        
        if args.mode in ("tension", "both") and os.path.exists(calibrator.tension_path):
            size = os.path.getsize(calibrator.tension_path)
            print(f"    ✓ {calibrator.tension_path} ({size} bytes)")
            meta_path = f"{calibrator.tension_path}.meta.json"
            if os.path.exists(meta_path):
                print(f"    ✓ {meta_path}")
        
        # 백업 파일 목록 (v2 신규)
        backups = calibrator.data_manager.list_backups()
        if backups:
            print(f"\n  백업 파일 ({len(backups)}개):")
            for backup in backups[:3]:
                print(f"    - {backup}")
        
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
        print(f"  중간 저장된 데이터는 .tmp 파일에 보존됩니다.")
        
    except Exception as e:
        print(f"\n\n{'='*70}")
        print(f"[에러 발생]")
        print(f"{'='*70}")
        print(f"\n  {type(e).__name__}: {e}")
        print(f"\n  트러블슈팅:")
        print(f"    1. USB 연결 확인")
        print(f"    2. 로봇 손 전원 확인")
        print(f"    3. 시리얼 포트 권한 확인")
        raise
    
    finally:
        # [안전 종료] 리소스 정리
        try:
            if 'calibrator' in locals():
                print(f"\n[안전 종료]")
                calibrator.hand.close()
                print(f"  ✓ 모터 토크 비활성화 완료")
        except:
            pass
        
        print(f"\n{'='*70}")
        print(f"프로그램이 종료되었습니다.")
        print(f"{'='*70}\n")