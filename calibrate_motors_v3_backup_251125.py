#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
RUKA Robot Hand Motor Calibration - 로봇 손 모터 캘리브레이션 프로그램 (v3.0)
================================================================================

이 프로그램은 RUKA(로봇 손)의 모터 동작 범위를 측정하고 저장하는
캘리브레이션 프로그램입니다.

================================================================================
버전 히스토리
================================================================================
v0.x (Original) - Linux 전용, 기본 기능
v1.x (Windows Port) - Windows 지원, 상세 주석, 에러 처리 개선
v2.0 (Enhanced) - 모터별 프로파일, 백업, 메타데이터, 적응형 임계값
v3.0 (Bidirectional) - 현재 버전, 양방향 탐색 방식

================================================================================
v3.0 주요 개선 사항 (v2 대비)
================================================================================
1. [양방향 Limit 탐색]
   - 현재 위치에서 시작 (와이어 텐션 보존)
   - 움켜지는 방향 → curl_limit 찾기
   - start_pos로 복귀
   - 펴지는 방향 → tension_limit 찾기
   - 한 번의 캘리브레이션으로 curl/tension 동시 측정

2. [하이브리드 탐색 알고리즘]
   - Coarse Step (큰 스텝): 대략적 위치 빠르게 탐색
   - Fine Step (작은 스텝): 정밀 위치 조정
   - 시간 효율성과 정밀도 모두 확보

3. [안전한 복귀 전략]
   - curl limit 찾은 후 start_pos로 복귀
   - tension 탐색 전 와이어 상태 안정화
   - 갑작스러운 위치 변화 방지

4. [통합 캘리브레이션 모드]
   - --mode both-auto: curl과 tension 자동 동시 측정
   - 기존 모드(curl, tension, both)도 지원

================================================================================
캘리브레이션 결과물
================================================================================
motor_limits/
├── {hand_type}_curl_limits.npy           # 11개 모터의 구부린 위치
├── {hand_type}_curl_limits.npy.meta.json # 메타데이터
├── {hand_type}_tension_limits.npy        # 11개 모터의 펼친 위치
├── {hand_type}_tension_limits.npy.meta.json
└── backups/                               # 자동 백업 디렉토리

================================================================================
사용 방법
================================================================================
# 양방향 자동 캘리브레이션 (권장, v3 신규)
python calibrate_motors_v3.py --hand-type right --mode both-auto

# 기존 방식 (대화형 tension 조정)
python calibrate_motors_v3.py -ht right -m both

# Curl만 측정
python calibrate_motors_v3.py -ht left -m curl

================================================================================
작성: NYU RUKA Team
버전: 3.0 (Bidirectional Search)
라이선스: MIT License
================================================================================
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import os
import sys
import time
import json
import shutil
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

# RUKA 프로젝트 모듈 임포트
from ruka_hand.control.hand import *
from ruka_hand.utils.file_ops import get_repo_root

# =============================================================================
# 로깅 설정
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 모터별 프로파일 설정
# =============================================================================

MOTOR_TYPE_XM430 = "XM430-W210-T"
MOTOR_TYPE_XL330 = "XL330-M288-T"

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

MOTOR_PROFILES = {
    MOTOR_TYPE_XM430: {
        "profile_velocity": 100,
        "profile_acceleration": 50,
        "current_limit": 700,
        "current_threshold": 100,
        "stabilization_time": 3.0,
    },
    MOTOR_TYPE_XL330: {
        "profile_velocity": 200,
        "profile_acceleration": 80,
        "current_limit": 400,
        "current_threshold": 50,
        "stabilization_time": 2.0,
    },
}

SPECIAL_MOTOR_THRESHOLDS = {
    4: 250,
    5: 200,
}

# =============================================================================
# 통신 안정성 설정
# =============================================================================

MAX_COMM_RETRIES = 3
COMM_RETRY_DELAY = 0.05
COMMAND_DELAY = 0.1
RECONNECT_DELAY = 1.0
MAX_RECONNECT_ATTEMPTS = 5

# =============================================================================
# Dynamixel 레지스터 주소
# =============================================================================

ADDR_PROFILE_VELOCITY = 112
ADDR_PROFILE_ACCELERATION = 108
ADDR_CURRENT_LIMIT = 38
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_TEMPERATURE = 146
ADDR_PRESENT_INPUT_VOLTAGE = 144

LEN_PROFILE_VELOCITY = 4
LEN_PROFILE_ACCELERATION = 4
LEN_CURRENT_LIMIT = 2
LEN_PRESENT_CURRENT = 2
LEN_PRESENT_POSITION = 4
LEN_PRESENT_TEMPERATURE = 1
LEN_PRESENT_INPUT_VOLTAGE = 2

# =============================================================================
# 캘리브레이션 설정 (v3 개선)
# =============================================================================

BINARY_SEARCH_THRESHOLD = 20
CURRENT_FILTER_WINDOW = 10
MULTI_SAMPLE_COUNT = 3
WARMUP_CYCLES = 2

# v3 신규: 하이브리드 탐색 설정
# ┌────────────────────────────────────────────────────────────────────┐
# │  Coarse Phase: 큰 스텝으로 대략적 위치 탐색                        │
# │  Fine Phase: 작은 스텝으로 정밀 위치 조정                          │
# └────────────────────────────────────────────────────────────────────┘
COARSE_STEP = 100      # 대략적 탐색 스텝 크기
FINE_STEP = 10         # 정밀 탐색 스텝 크기
COARSE_THRESHOLD_MARGIN = 1.5  # Coarse 탐색에서 임계값 여유 (배수)

# 위치 제한 (안전 범위)
POSITION_MIN = 100
POSITION_MAX = 3995


# =============================================================================
# 키보드 입력 캡처 함수 (Windows 전용)
# =============================================================================

def get_key():
    """단일 키 입력을 캡처하는 함수 (Windows 전용)"""
    import msvcrt
    
    ch = msvcrt.getch()
    
    if ch in (b'\x00', b'\xe0'):
        ch2 = msvcrt.getch()
        key_map = {
            b'H': '\x1b[A',
            b'P': '\x1b[B',
            b'M': '\x1b[C',
            b'K': '\x1b[D',
        }
        return key_map.get(ch2, ch2.decode('utf-8', errors='ignore'))
    
    return ch.decode('utf-8', errors='ignore')


# =============================================================================
# RobustHandController 클래스
# =============================================================================

class RobustHandController:
    """통신 안정성이 강화된 Hand 제어 래퍼 클래스"""
    
    def __init__(self, hand, hand_type):
        self.hand = hand
        self.hand_type = hand_type
        self.max_retries = MAX_COMM_RETRIES
        self.retry_delay = COMM_RETRY_DELAY
        self.is_connected = True
    
    def robust_read_pos(self):
        """재시도 메커니즘이 있는 위치 읽기"""
        for attempt in range(self.max_retries):
            try:
                pos = self.hand.read_pos()
                if pos is not None:
                    return pos
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        return None
    
    def robust_read_single_cur(self, motor_id: int) -> float:
        """재시도 메커니즘이 있는 단일 모터 전류 읽기"""
        for attempt in range(self.max_retries):
            try:
                current = self.hand.read_single_cur(motor_id)
                if current is not None:
                    return current
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        return 0.0
    
    def robust_set_pos(self, pos):
        """재시도 메커니즘이 있는 위치 설정"""
        for attempt in range(self.max_retries):
            try:
                self.hand.set_pos(pos)
                return True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        return False
    
    def close(self):
        """연결 종료"""
        try:
            self.hand.close()
        except:
            pass


# =============================================================================
# CalibrationDataManager 클래스
# =============================================================================

class CalibrationDataManager:
    """캘리브레이션 데이터 관리 클래스"""
    
    def __init__(self, save_dir: str, hand_type: str):
        self.save_dir = save_dir
        self.hand_type = hand_type
        self.backup_dir = os.path.join(save_dir, "backups")
        
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def backup_existing(self, filepath: str) -> Optional[str]:
        """기존 파일 백업"""
        if not os.path.exists(filepath):
            return None
        
        filename = os.path.basename(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filename}_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        shutil.copy2(filepath, backup_path)
        logger.info(f"  백업 생성: {backup_path}")
        
        return backup_path
    
    def save_with_metadata(self, data: np.ndarray, filepath: str, metadata: Dict) -> bool:
        """데이터와 메타데이터 함께 저장"""
        try:
            self.backup_existing(filepath)
            np.save(filepath, data)
            
            meta_path = f"{filepath}.meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  저장 완료: {filepath}")
            return True
        except Exception as e:
            logger.error(f"  저장 실패: {e}")
            return False
    
    def save_temp(self, data: np.ndarray, filepath: str):
        """임시 파일로 중간 저장"""
        temp_path = f"{filepath}.tmp"
        np.save(temp_path, data)
    
    def load_temp(self, filepath: str) -> Optional[np.ndarray]:
        """임시 파일 로드"""
        temp_path = f"{filepath}.tmp"
        if os.path.exists(temp_path):
            return np.load(temp_path)
        return None
    
    def cleanup_temp(self, filepath: str):
        """임시 파일 정리"""
        temp_path = f"{filepath}.tmp"
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def list_backups(self) -> List[str]:
        """백업 파일 목록 반환"""
        if not os.path.exists(self.backup_dir):
            return []
        return sorted(os.listdir(self.backup_dir), reverse=True)


# =============================================================================
# HandCalibrator 클래스 (v3 개선)
# =============================================================================

class HandCalibrator:
    """
    RUKA Robot Hand 캘리브레이션 클래스 (v3.0)
    
    v3 주요 개선:
    ┌────────────────────────────────────────────────────────────────────┐
    │  1. 양방향 limit 탐색 (find_both_limits)                          │
    │  2. 하이브리드 탐색 (coarse → fine)                               │
    │  3. 안전한 복귀 전략 (curl 후 start_pos 복귀)                     │
    │  4. curl/tension 동시 자동 측정                                   │
    └────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        data_save_dir: str,
        hand_type: str,
        curr_lim: int = 50,
        testing: bool = False,
        motor_ids: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        """HandCalibrator 초기화"""
        logger.info(f"\n{'='*70}")
        logger.info("HandCalibrator 초기화 시작 (v3.0 - Bidirectional)")
        logger.info(f"{'='*70}")
        
        # Hand 객체 생성
        try:
            logger.info("\n[1/6] Hand 객체 생성 중...")
            base_hand = Hand(hand_type=hand_type)
            logger.info("  ✓ Hand 객체 생성 완료")
            
            logger.info("\n[2/6] RobustHandController 래핑 중...")
            self.hand = RobustHandController(base_hand, hand_type)
            logger.info("  ✓ 통신 안정성 강화 완료")
            
        except Exception as e:
            logger.error(f"  ✗ Hand 초기화 실패: {e}")
            raise
        
        self.curr_lim = curr_lim
        self.testing = testing
        self.motor_ids = motor_ids
        self.data_save_dir = data_save_dir
        
        # 데이터 매니저 초기화
        logger.info("\n[3/6] 데이터 매니저 초기화 중...")
        self.data_manager = CalibrationDataManager(data_save_dir, hand_type)
        logger.info("  ✓ 데이터 매니저 초기화 완료")
        
        # 파일 경로 설정
        logger.info("\n[4/6] 파일 경로 설정 중...")
        self.curled_path = os.path.join(data_save_dir, f"{hand_type}_curl_limits.npy")
        self.tension_path = os.path.join(data_save_dir, f"{hand_type}_tension_limits.npy")
        logger.info(f"  Curl 파일: {self.curled_path}")
        logger.info(f"  Tension 파일: {self.tension_path}")
        
        # 모터별 프로파일 설정
        logger.info("\n[5/6] 모터별 프로파일 설정 중...")
        self._setup_motor_profiles()
        
        # 통신 상태 진단
        logger.info("\n[6/6] 통신 상태 진단 중...")
        self._diagnose_communication()
        
        logger.info(f"\n{'='*70}")
        logger.info("HandCalibrator 초기화 완료")
        logger.info(f"{'='*70}\n")
    
    def _setup_motor_profiles(self):
        """모터별 개별 프로파일 설정"""
        try:
            for motor_id in self.motor_ids:
                motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
                profile = MOTOR_PROFILES[motor_type]
                
                self._write_motor_param(
                    motor_id, ADDR_PROFILE_VELOCITY,
                    profile["profile_velocity"], LEN_PROFILE_VELOCITY
                )
                self._write_motor_param(
                    motor_id, ADDR_PROFILE_ACCELERATION,
                    profile["profile_acceleration"], LEN_PROFILE_ACCELERATION
                )
                self._write_motor_param(
                    motor_id, ADDR_CURRENT_LIMIT,
                    profile["current_limit"], LEN_CURRENT_LIMIT
                )
                
                logger.info(
                    f"  모터 {motor_id:2d} ({motor_type}): "
                    f"Vel={profile['profile_velocity']}, "
                    f"Acc={profile['profile_acceleration']}, "
                    f"Cur={profile['current_limit']}mA"
                )
                time.sleep(0.05)
                
        except Exception as e:
            logger.warning(f"  ⚠️ 프로파일 설정 실패: {e}")
    
    def _write_motor_param(self, motor_id: int, addr: int, value: int, length: int):
        """단일 모터 파라미터 쓰기"""
        try:
            self.hand.hand.dxl_client.write(motor_id, addr, value, length)
        except Exception as e:
            logger.warning(f"모터 {motor_id} 파라미터 쓰기 실패: {e}")
    
    def _diagnose_communication(self):
        """통신 상태 진단"""
        total_attempts = 0
        successful_attempts = 0
        
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
        """안전하게 모터 위치 읽기"""
        pos = self.hand.robust_read_pos()
        
        if isinstance(pos, (list, tuple)):
            return np.array([int(x) for x in pos], dtype=np.int32)
        elif isinstance(pos, np.ndarray):
            return pos.astype(np.int32)
        else:
            return np.array([int(pos)], dtype=np.int32)
    
    def _safe_set_pos(self, pos):
        """안전하게 모터 위치 설정"""
        if isinstance(pos, np.ndarray):
            pos_list = [int(x) for x in pos]
            self.hand.robust_set_pos(pos_list)
        else:
            self.hand.robust_set_pos(pos)
    
    def _filtered_current_reading(self, motor_id: int, window_size: int = CURRENT_FILTER_WINDOW) -> float:
        """이동 평균 필터로 전류 노이즈 제거"""
        currents = []
        for _ in range(window_size):
            try:
                current = self.hand.robust_read_single_cur(motor_id)
                currents.append(abs(current))
            except:
                pass
            time.sleep(0.01)
        
        if len(currents) < 3:
            return abs(self.hand.robust_read_single_cur(motor_id))
        
        sorted_currents = sorted(currents)
        filtered = sorted_currents[:-2] if len(sorted_currents) > 2 else sorted_currents
        
        return np.mean(filtered)
    
    def _get_adaptive_threshold(self, motor_id: int) -> float:
        """적응형 전류 임계값 계산"""
        if motor_id in SPECIAL_MOTOR_THRESHOLDS:
            return SPECIAL_MOTOR_THRESHOLDS[motor_id]
        
        motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
        base_threshold = MOTOR_PROFILES[motor_type]["current_threshold"]
        
        try:
            baseline_currents = []
            for _ in range(10):
                current = self.hand.robust_read_single_cur(motor_id)
                baseline_currents.append(abs(current))
                time.sleep(0.05)
            
            baseline = np.mean(baseline_currents)
            noise_level = np.std(baseline_currents)
            
            adaptive_threshold = baseline + 3 * noise_level + base_threshold
            
            if self.testing:
                logger.info(f"  모터 {motor_id} 적응형 임계값: {adaptive_threshold:.1f}mA "
                           f"(baseline={baseline:.1f}, noise={noise_level:.1f})")
            
            return adaptive_threshold
            
        except Exception as e:
            logger.warning(f"적응형 임계값 계산 실패, 기본값 사용: {e}")
            return base_threshold
    
    def _warmup_motor(self, motor_id: int, cycles: int = WARMUP_CYCLES):
        """캘리브레이션 전 모터 워밍업"""
        if self.testing:
            logger.info(f"  모터 {motor_id} 워밍업 시작 ({cycles}회)...")
        
        current_positions = self._safe_read_pos()
        start_pos = int(current_positions[motor_id - 1])
        
        for cycle in range(cycles):
            pos = current_positions.copy()
            
            pos[motor_id - 1] = max(start_pos - 200, POSITION_MIN)
            self._safe_set_pos(pos)
            time.sleep(0.3)
            
            pos[motor_id - 1] = min(start_pos + 200, POSITION_MAX)
            self._safe_set_pos(pos)
            time.sleep(0.3)
        
        pos[motor_id - 1] = start_pos
        self._safe_set_pos(pos)
        time.sleep(0.5)
        
        if self.testing:
            logger.info(f"  모터 {motor_id} 워밍업 완료")

    # =========================================================================
    # v3 신규: 하이브리드 탐색 메서드
    # =========================================================================
    
    def _find_limit_in_direction(
        self,
        motor_id: int,
        start_pos: int,
        direction: int,
        current_threshold: float,
        stabilization_time: float,
        limit_type: str = "curl"
    ) -> int:
        """
        단방향 limit 탐색 (하이브리드: coarse → fine)
        
        v3 신규 메서드
        
        알고리즘:
        ┌────────────────────────────────────────────────────────────────────┐
        │  Phase 1: Coarse Search (대략적 탐색)                              │
        │    - COARSE_STEP 크기로 이동                                       │
        │    - 전류 > threshold * COARSE_THRESHOLD_MARGIN 시 멈춤           │
        │    - 빠르게 대략적 위치 파악                                       │
        │                                                                    │
        │  Phase 2: Fine Search (정밀 탐색)                                  │
        │    - coarse 위치에서 한 스텝 뒤로                                  │
        │    - FINE_STEP 크기로 세밀하게 이동                                │
        │    - 전류 > threshold 시 정확한 limit 위치                         │
        └────────────────────────────────────────────────────────────────────┘
        
        Args:
            motor_id: 모터 ID (1~11)
            start_pos: 시작 위치
            direction: 이동 방향 (+1: 값 증가, -1: 값 감소)
            current_threshold: 전류 임계값 (mA)
            stabilization_time: 안정화 대기 시간 (초)
            limit_type: "curl" 또는 "tension" (로깅용)
        
        Returns:
            int: 찾은 limit 위치
        """
        current_positions = self._safe_read_pos()
        pos = current_positions.copy()
        current_pos = start_pos
        
        # Coarse threshold (여유 있게)
        coarse_threshold = current_threshold * COARSE_THRESHOLD_MARGIN
        
        print(f"\n    [Phase 1: Coarse Search] step={COARSE_STEP}, threshold={coarse_threshold:.1f}mA")
        
        # =====================================================================
        # Phase 1: Coarse Search
        # =====================================================================
        coarse_iterations = 0
        while True:
            coarse_iterations += 1
            
            # 다음 위치 계산
            next_pos = current_pos + (direction * COARSE_STEP)
            
            # 범위 체크
            if next_pos < POSITION_MIN or next_pos > POSITION_MAX:
                print(f"      → 범위 도달 (pos={current_pos})")
                break
            
            # 이동
            pos[motor_id - 1] = next_pos
            self._safe_set_pos(pos)
            print(f"      [{coarse_iterations}] 이동: {current_pos} → {next_pos}...", end='', flush=True)
            
            time.sleep(stabilization_time * 0.5)  # coarse는 짧은 대기
            
            # 전류 측정
            cur = self._filtered_current_reading(motor_id)
            actual_pos = int(self._safe_read_pos()[motor_id - 1])
            print(f" 전류={cur:.1f}mA, 실제={actual_pos}")
            
            current_pos = actual_pos
            
            # 임계값 도달 체크
            if cur >= coarse_threshold:
                print(f"      → Coarse limit 발견! (전류={cur:.1f}mA)")
                break
            
            # 무한 루프 방지
            if coarse_iterations > 50:
                print(f"      → 최대 반복 도달")
                break
        
        # Coarse 위치에서 한 스텝 뒤로
        coarse_limit = current_pos
        step_back_pos = coarse_limit - (direction * COARSE_STEP)
        step_back_pos = max(POSITION_MIN, min(POSITION_MAX, step_back_pos))
        
        print(f"\n    [Phase 2: Fine Search] step={FINE_STEP}, threshold={current_threshold:.1f}mA")
        print(f"      시작 위치: {step_back_pos} (coarse={coarse_limit}에서 뒤로)")
        
        # 뒤로 이동
        pos[motor_id - 1] = step_back_pos
        self._safe_set_pos(pos)
        time.sleep(stabilization_time)
        current_pos = int(self._safe_read_pos()[motor_id - 1])
        
        # =====================================================================
        # Phase 2: Fine Search
        # =====================================================================
        fine_iterations = 0
        final_limit = current_pos
        
        while True:
            fine_iterations += 1
            
            # 다음 위치 계산
            next_pos = current_pos + (direction * FINE_STEP)
            
            # 범위 체크
            if next_pos < POSITION_MIN or next_pos > POSITION_MAX:
                print(f"      → 범위 도달")
                break
            
            # coarse limit를 넘어가면 중단
            if direction > 0 and next_pos > coarse_limit + FINE_STEP:
                break
            if direction < 0 and next_pos < coarse_limit - FINE_STEP:
                break
            
            # 이동
            pos[motor_id - 1] = next_pos
            self._safe_set_pos(pos)
            print(f"      [{fine_iterations}] 이동: {current_pos} → {next_pos}...", end='', flush=True)
            
            time.sleep(stabilization_time)
            
            # 전류 측정
            cur = self._filtered_current_reading(motor_id)
            actual_pos = int(self._safe_read_pos()[motor_id - 1])
            print(f" 전류={cur:.1f}mA, 실제={actual_pos}")
            
            current_pos = actual_pos
            
            # 정확한 임계값 도달 체크
            if cur >= current_threshold:
                final_limit = current_pos
                print(f"      → Fine limit 발견! pos={final_limit}")
                break
            
            final_limit = current_pos
            
            # 무한 루프 방지
            if fine_iterations > 30:
                print(f"      → 최대 반복 도달")
                break
        
        return final_limit

    def find_both_limits(self, motor_id: int) -> Tuple[int, int]:
        """
        양방향 limit 탐색 (v3 핵심 메서드)
        
        현재 위치에서 시작하여:
        1. 움켜지는 방향 → curl_limit 찾기
        2. start_pos로 복귀
        3. 펴지는 방향 → tension_limit 찾기
        
        이점:
        ┌────────────────────────────────────────────────────────────────────┐
        │  - 와이어 텐션 보존: 현재 위치에서 시작                           │
        │  - 안전성: 점진적 움직임, 갑작스러운 변화 없음                    │
        │  - 효율성: curl/tension 한 번에 측정                              │
        │  - 정확성: 하이브리드 탐색으로 정밀도 확보                        │
        └────────────────────────────────────────────────────────────────────┘
        
        Args:
            motor_id: 모터 ID (1~11)
        
        Returns:
            Tuple[int, int]: (curl_limit, tension_limit)
        """
        motor_type = MOTOR_TYPE_MAP.get(motor_id, MOTOR_TYPE_XL330)
        profile = MOTOR_PROFILES[motor_type]
        stabilization_time = profile["stabilization_time"]
        
        # 특수 모터 처리
        if motor_id in SPECIAL_MOTOR_THRESHOLDS:
            current_threshold = SPECIAL_MOTOR_THRESHOLDS[motor_id]
            stabilization_time = 5.0
        else:
            current_threshold = self._get_adaptive_threshold(motor_id)
        
        print(f"\n{'='*70}")
        print(f"[모터 {motor_id} 양방향 캘리브레이션]")
        print(f"{'='*70}")
        print(f"  모터 타입: {motor_type}")
        print(f"  전류 임계값: {current_threshold:.1f} mA")
        print(f"  안정화 시간: {stabilization_time}초")
        
        # 워밍업
        self._warmup_motor(motor_id)
        
        # 현재 위치 읽기 (시작점)
        current_positions = self._safe_read_pos()
        start_pos = int(current_positions[motor_id - 1])
        print(f"\n  시작 위치 (start_pos): {start_pos}")
        
        # 방향 결정
        if self.hand.hand_type == "right":
            curl_direction = +1    # 값 증가 = 움켜짐
            tension_direction = -1 # 값 감소 = 펴짐
        else:
            curl_direction = -1    # 왼손: 값 감소 = 움켜짐
            tension_direction = +1 # 왼손: 값 증가 = 펴짐
        
        print(f"  손 방향: {self.hand.hand_type.upper()}")
        print(f"  Curl 방향: {'값 증가' if curl_direction > 0 else '값 감소'}")
        print(f"  Tension 방향: {'값 증가' if tension_direction > 0 else '값 감소'}")
        
        # =====================================================================
        # Step 1: CURL limit 찾기 (움켜지는 방향)
        # =====================================================================
        print(f"\n{'─'*70}")
        print(f"  [STEP 1/3] CURL limit 탐색 (움켜지는 방향)")
        print(f"{'─'*70}")
        
        curl_limit = self._find_limit_in_direction(
            motor_id=motor_id,
            start_pos=start_pos,
            direction=curl_direction,
            current_threshold=current_threshold,
            stabilization_time=stabilization_time,
            limit_type="curl"
        )
        
        print(f"\n  ✓ CURL limit: {curl_limit}")
        
        # =====================================================================
        # Step 2: start_pos로 복귀
        # =====================================================================
        print(f"\n{'─'*70}")
        print(f"  [STEP 2/3] 시작 위치로 복귀")
        print(f"{'─'*70}")
        print(f"    {curl_limit} → {start_pos}")
        
        current_positions = self._safe_read_pos()
        current_positions[motor_id - 1] = start_pos
        self._safe_set_pos(current_positions)
        time.sleep(stabilization_time)
        
        actual_return = int(self._safe_read_pos()[motor_id - 1])
        print(f"    복귀 완료 (실제: {actual_return})")
        
        # 추가 안정화 대기
        print(f"    와이어 안정화 대기 (1초)...")
        time.sleep(1.0)
        
        # =====================================================================
        # Step 3: TENSION limit 찾기 (펴지는 방향)
        # =====================================================================
        print(f"\n{'─'*70}")
        print(f"  [STEP 3/3] TENSION limit 탐색 (펴지는 방향)")
        print(f"{'─'*70}")
        
        tension_limit = self._find_limit_in_direction(
            motor_id=motor_id,
            start_pos=start_pos,
            direction=tension_direction,
            current_threshold=current_threshold,
            stabilization_time=stabilization_time,
            limit_type="tension"
        )
        
        print(f"\n  ✓ TENSION limit: {tension_limit}")
        
        # =====================================================================
        # 완료: 안전한 중간 위치로 복귀
        # =====================================================================
        safe_pos = (curl_limit + tension_limit) // 2
        print(f"\n  안전 위치로 복귀: {safe_pos} (중간값)")
        
        current_positions = self._safe_read_pos()
        current_positions[motor_id - 1] = safe_pos
        self._safe_set_pos(current_positions)
        time.sleep(0.5)
        
        # 결과 요약
        print(f"\n{'─'*70}")
        print(f"  [모터 {motor_id} 캘리브레이션 완료]")
        print(f"{'─'*70}")
        print(f"    시작 위치:    {start_pos}")
        print(f"    CURL limit:   {curl_limit}")
        print(f"    TENSION limit:{tension_limit}")
        print(f"    동작 범위:    {abs(curl_limit - tension_limit)} steps")
        
        return curl_limit, tension_limit

    def find_both_limits_with_multi_sample(
        self,
        motor_id: int,
        num_samples: int = MULTI_SAMPLE_COUNT
    ) -> Tuple[int, int]:
        """
        다중 측정으로 안정적인 양방향 limits 찾기
        
        Args:
            motor_id: 모터 ID (1~11)
            num_samples: 측정 횟수 (기본: 3)
        
        Returns:
            Tuple[int, int]: (curl_limit, tension_limit) 중앙값
        """
        curl_positions = []
        tension_positions = []
        
        print(f"\n[모터 {motor_id}] 다중 측정 ({num_samples}회)...")
        
        for i in range(num_samples):
            print(f"\n  ═══ 측정 {i+1}/{num_samples} ═══")
            
            curl_limit, tension_limit = self.find_both_limits(motor_id)
            curl_positions.append(curl_limit)
            tension_positions.append(tension_limit)
            
            print(f"  측정 {i+1} 결과: curl={curl_limit}, tension={tension_limit}")
            
            # 다음 측정 전 안정화
            if i < num_samples - 1:
                time.sleep(1.0)
        
        # IQR 이상치 제거 및 중앙값 계산
        def get_robust_median(values):
            if len(values) >= 3:
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                filtered = [v for v in values if lower <= v <= upper]
                if len(filtered) < 2:
                    filtered = values
            else:
                filtered = values
            return int(np.median(filtered))
        
        curl_result = get_robust_median(curl_positions)
        tension_result = get_robust_median(tension_positions)
        
        print(f"\n  다중 측정 결과:")
        print(f"    Curl: {curl_positions} → 최종: {curl_result}")
        print(f"    Tension: {tension_positions} → 최종: {tension_result}")
        
        return curl_result, tension_result

    def find_all_limits_bidirectional(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        모든 모터의 양방향 limits 자동 측정 (v3 핵심 메서드)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (curl_limits, tension_limits)
        """
        print("\n" + "="*70)
        print("[양방향 자동 캘리브레이션 시작] (v3)")
        print("="*70)
        print(f"\n  총 모터 수: {len(self.motor_ids)}개")
        print(f"  다중 측정: {MULTI_SAMPLE_COUNT}회")
        print(f"  하이브리드 탐색: Coarse({COARSE_STEP}) → Fine({FINE_STEP})")
        print(f"  예상 소요 시간: 약 15-30분")
        
        curl_limits = np.zeros(len(self.motor_ids), dtype=np.int32)
        tension_limits = np.zeros(len(self.motor_ids), dtype=np.int32)
        start_time = time.time()
        
        for i, mid in enumerate(self.motor_ids):
            print(f"\n\n{'#'*70}")
            print(f"# 진행: [{i+1}/{len(self.motor_ids)}] 모터 {mid}번")
            print(f"{'#'*70}")
            
            try:
                curl, tension = self.find_both_limits_with_multi_sample(mid)
                
                curl_limits[i] = curl
                tension_limits[i] = tension
                
                print(f"\n  ✓ 모터 {mid}: curl={curl}, tension={tension}")
                
                # 중간 저장
                self.data_manager.save_temp(curl_limits, self.curled_path)
                self.data_manager.save_temp(tension_limits, self.tension_path)
                
            except KeyboardInterrupt:
                print(f"\n\n[사용자 중단]")
                print(f"  현재까지 측정된 데이터:")
                for j in range(i):
                    print(f"    모터 {self.motor_ids[j]}: curl={curl_limits[j]}, tension={tension_limits[j]}")
                raise
            
            except Exception as e:
                print(f"\n✗ 모터 {mid} 측정 실패: {e}")
                retry = input("재시도하시겠습니까? (y/n): ")
                if retry.lower() == 'y':
                    curl, tension = self.find_both_limits(mid)
                    curl_limits[i] = curl
                    tension_limits[i] = tension
                else:
                    raise
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n\n{'='*70}")
        print(f"[양방향 캘리브레이션 완료]")
        print(f"{'='*70}")
        print(f"\n  소요 시간: {minutes}분 {seconds}초")
        print(f"\n  측정 결과:")
        print(f"  {'─'*50}")
        print(f"  {'모터':^8} │ {'Curl':^10} │ {'Tension':^10} │ {'범위':^10}")
        print(f"  {'─'*50}")
        for i, mid in enumerate(self.motor_ids):
            range_val = abs(curl_limits[i] - tension_limits[i])
            print(f"  {mid:^8} │ {curl_limits[i]:^10} │ {tension_limits[i]:^10} │ {range_val:^10}")
        print(f"  {'─'*50}")
        
        return curl_limits, tension_limits

    # =========================================================================
    # 기존 메서드 (v2 호환)
    # =========================================================================
    
    def find_bound(self, motor_id: int) -> int:
        """기존 단방향 curl 탐색 (v2 호환)"""
        curl_limit, _ = self.find_both_limits(motor_id)
        return curl_limit
    
    def find_curled_with_multi_sample(self, motor_id: int, num_samples: int = MULTI_SAMPLE_COUNT) -> int:
        """기존 다중 측정 curl (v2 호환)"""
        curl_limit, _ = self.find_both_limits_with_multi_sample(motor_id, num_samples)
        return curl_limit
    
    def find_curled(self) -> np.ndarray:
        """기존 curl만 측정 (v2 호환)"""
        curl_limits, _ = self.find_all_limits_bidirectional()
        return curl_limits
    
    def estimate_tensioned_from_curled(self, curled: np.ndarray) -> np.ndarray:
        """Curl 위치로부터 Tension 위치 초기 추정 (v2 호환)"""
        f = 1 if self.hand.hand_type == "right" else -1
        tensioned = np.array([int(x - f * 1100) for x in curled], dtype=np.int32)
        return tensioned
    
    def interactive_refine_tensioned(self, tensioned_init: np.ndarray, step: int = 10) -> np.ndarray:
        """대화형 tension 조정 (v2 호환)"""
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
            print(f"  초기 추정값: {tensioned[idx]}")
            
            pos = current_pos.copy()
            pos[idx] = tensioned[idx]
            self._safe_set_pos(pos)
            time.sleep(0.2)
            
            while True:
                print(f"\n  [모터 {mid}] 현재 후보: {pos[idx]:4d}")
                print(f"  화살표로 조정, Enter로 저장, 'q'로 스킵: ", end='', flush=True)
                
                k = get_key()
                
                if k in ("\r", "\n"):
                    tensioned[idx] = int(pos[idx])
                    print(f"\n  ✓ 모터 {mid} 저장: {tensioned[idx]}")
                    break
                
                elif k in ("\x1b[A", "\x1b[C"):
                    old_pos = int(pos[idx])
                    pos[idx] = int(max(min(int(pos[idx]) + step * f, 4090), 10))
                    if pos[idx] != old_pos:
                        self._safe_set_pos(pos)
                        print(f"\n    → {old_pos} → {pos[idx]}")
                
                elif k in ("\x1b[B", "\x1b[D"):
                    old_pos = int(pos[idx])
                    pos[idx] = int(max(min(int(pos[idx]) - step * f, 4090), 10))
                    if pos[idx] != old_pos:
                        self._safe_set_pos(pos)
                        print(f"\n    → {old_pos} → {pos[idx]}")
                
                elif k.lower() == "q":
                    print(f"\n  ⊗ 모터 {mid} 스킵")
                    break
            
            self.data_manager.save_temp(tensioned, self.tension_path)
        
        return tensioned.astype(int)
    
    def _collect_metadata(self) -> Dict[str, Any]:
        """메타데이터 수집"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "3.0",
            "hand_type": self.hand.hand_type,
            "motor_ids": self.motor_ids,
            "curr_lim": self.curr_lim,
            "coarse_step": COARSE_STEP,
            "fine_step": FINE_STEP,
            "multi_sample_count": MULTI_SAMPLE_COUNT,
            "motor_profiles": {},
        }
        
        for mid in self.motor_ids:
            motor_type = MOTOR_TYPE_MAP.get(mid, MOTOR_TYPE_XL330)
            metadata["motor_profiles"][str(mid)] = {
                "type": motor_type,
                "profile": MOTOR_PROFILES[motor_type]
            }
        
        return metadata

    # =========================================================================
    # 저장 메서드
    # =========================================================================
    
    def save_both_limits_auto(self):
        """
        양방향 캘리브레이션 및 저장 (v3 권장 방식)
        
        curl과 tension을 자동으로 동시에 측정하고 저장합니다.
        """
        print("\n" + "="*70)
        print("[양방향 자동 캘리브레이션] (v3)")
        print("="*70)
        
        try:
            # 양방향 측정
            curl_limits, tension_limits = self.find_all_limits_bidirectional()
            
            # 메타데이터 수집
            metadata = self._collect_metadata()
            metadata["calibration_type"] = "both-auto"
            metadata["calibration_method"] = "bidirectional_hybrid"
            
            # Curl 저장
            curl_metadata = metadata.copy()
            curl_metadata["data_stats"] = {
                "min": int(curl_limits.min()),
                "max": int(curl_limits.max()),
                "mean": float(curl_limits.mean()),
                "std": float(curl_limits.std()),
            }
            
            success_curl = self.data_manager.save_with_metadata(
                curl_limits, self.curled_path, curl_metadata
            )
            
            # Tension 저장
            tension_metadata = metadata.copy()
            tension_metadata["data_stats"] = {
                "min": int(tension_limits.min()),
                "max": int(tension_limits.max()),
                "mean": float(tension_limits.mean()),
                "std": float(tension_limits.std()),
            }
            
            success_tension = self.data_manager.save_with_metadata(
                tension_limits, self.tension_path, tension_metadata
            )
            
            if success_curl and success_tension:
                self.data_manager.cleanup_temp(self.curled_path)
                self.data_manager.cleanup_temp(self.tension_path)
                
                print(f"\n  ✓ 양방향 캘리브레이션 저장 완료!")
                print(f"    Curl 파일: {self.curled_path}")
                print(f"    Tension 파일: {self.tension_path}")
            
        except Exception as e:
            print(f"\n  ✗ 저장 실패: {e}")
            raise
    
    def save_curled_limits(self):
        """Curl Limits만 측정 및 저장 (v2 호환)"""
        print("\n" + "="*70)
        print("[Curl Limits 저장]")
        print("="*70)
        
        try:
            curled = self.find_curled()
            
            metadata = self._collect_metadata()
            metadata["calibration_type"] = "curl"
            metadata["data_stats"] = {
                "min": int(curled.min()),
                "max": int(curled.max()),
                "mean": float(curled.mean()),
                "std": float(curled.std()),
            }
            
            success = self.data_manager.save_with_metadata(
                curled, self.curled_path, metadata
            )
            
            if success:
                self.data_manager.cleanup_temp(self.curled_path)
                print(f"\n  ✓ Curl Limits 저장 완료!")
                print(f"    파일: {self.curled_path}")
            
        except Exception as e:
            print(f"\n  ✗ Curl Limits 저장 실패: {e}")
            raise
    
    def save_tensioned_limits(self):
        """Tension Limits 대화형 조정 및 저장 (v2 호환)"""
        print("\n" + "="*70)
        print("[Tension Limits 저장]")
        print("="*70)
        
        print(f"\n  Curl Limits 확인 중...")
        
        if os.path.exists(self.curled_path):
            print(f"  ✓ 기존 Curl 파일 발견: {self.curled_path}")
            curled = np.load(self.curled_path)
        else:
            print(f"  ✗ Curl 파일 없음. 자동 측정을 시작합니다...")
            curled = self.find_curled()
            np.save(self.curled_path, curled)
        
        try:
            t_init = self.estimate_tensioned_from_curled(curled)
            t_refined = self.interactive_refine_tensioned(t_init, step=10)
            
            metadata = self._collect_metadata()
            metadata["calibration_type"] = "tension"
            metadata["curl_reference"] = self.curled_path
            metadata["data_stats"] = {
                "min": int(t_refined.min()),
                "max": int(t_refined.max()),
                "mean": float(t_refined.mean()),
                "std": float(t_refined.std()),
            }
            
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
    """명령줄 인자 파싱"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RUKA Robot Hand Motor Calibration (v3.0 - Bidirectional)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 양방향 자동 캘리브레이션 (권장)
  python calibrate_motors_v3.py --hand-type right --mode both-auto
  
  # 기존 방식 (대화형 tension 조정)
  python calibrate_motors_v3.py -ht right -m both
  
  # Curl만 측정
  python calibrate_motors_v3.py -ht left -m curl

v3 신규 기능:
  - both-auto 모드: curl과 tension 자동 동시 측정
  - 하이브리드 탐색: coarse → fine 2단계 탐색
  - 양방향 탐색: 현재 위치 기준 curl/tension 순차 탐색
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
        choices=["curl", "tension", "both", "both-auto"],
        default="both-auto",
        help="캘리브레이션 모드. 기본값: both-auto (v3 권장)"
    )
    
    parser.add_argument(
        "--multi-sample",
        type=int,
        default=MULTI_SAMPLE_COUNT,
        help=f"다중 측정 횟수. 기본값: {MULTI_SAMPLE_COUNT}"
    )
    
    parser.add_argument(
        "--coarse-step",
        type=int,
        default=COARSE_STEP,
        help=f"Coarse 탐색 스텝 크기. 기본값: {COARSE_STEP}"
    )
    
    parser.add_argument(
        "--fine-step",
        type=int,
        default=FINE_STEP,
        help=f"Fine 탐색 스텝 크기. 기본값: {FINE_STEP}"
    )
    
    return parser.parse_args()


# =============================================================================
# 메인 실행 블록
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUKA Robot Hand Motor Calibration (v3.0 - Bidirectional)")
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
        print(f"    다중 측정: {args.multi_sample}회")
        print(f"    Coarse 스텝: {args.coarse_step}")
        print(f"    Fine 스텝: {args.fine_step}")
        
        # 전역 설정 업데이트
        MULTI_SAMPLE_COUNT = args.multi_sample
        COARSE_STEP = args.coarse_step
        FINE_STEP = args.fine_step
        
        # 프로젝트 루트 경로
        print(f"\n[단계 2/5] 프로젝트 경로 확인 중...")
        repo_root = get_repo_root()
        print(f"  ✓ 프로젝트 루트: {repo_root}")
        
        # 저장 디렉토리 설정
        print(f"\n[단계 3/5] 저장 디렉토리 설정 중...")
        save_dir = os.path.join(repo_root, "motor_limits")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"  → 디렉토리 생성: {save_dir}")
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
        
        if args.mode == "both-auto":
            # v3 권장: 양방향 자동 캘리브레이션
            calibrator.save_both_limits_auto()
        
        elif args.mode == "curl":
            calibrator.save_curled_limits()
        
        elif args.mode == "tension":
            calibrator.save_tensioned_limits()
        
        elif args.mode == "both":
            # v2 호환: curl 자동 + tension 대화형
            calibrator.save_curled_limits()
            calibrator.save_tensioned_limits()
        
        # 완료 메시지
        print(f"\n{'='*70}")
        print(f"[캘리브레이션 완료]")
        print(f"{'='*70}")
        print(f"\n  모든 캘리브레이션이 성공적으로 완료되었습니다!")
        
        print(f"\n  생성된 파일:")
        if os.path.exists(calibrator.curled_path):
            size = os.path.getsize(calibrator.curled_path)
            print(f"    ✓ {calibrator.curled_path} ({size} bytes)")
        
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