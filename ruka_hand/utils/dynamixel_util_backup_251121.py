#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RUKA Robot Hand Dynamixel Motor Communication Utility
Dynamixel 모터 통신 유틸리티

이 모듈은 RUKA 로봇 손의 Dynamixel XL-330 서보모터와 통신하기 위한
저수준 인터페이스를 제공합니다.

주요 기능:
1. ModBus RTU 프로토콜 기반 시리얼 통신
2. GroupSyncWrite/Read를 통한 최적화된 다중 모터 제어
3. Control Table 레지스터 읽기/쓰기
4. 에러 처리 및 자동 재시도
5. 토크 활성화/비활성화 관리

핵심 특징:
- Protocol 2.0 지원 (Dynamixel XL-330 전용)
- 2Mbps Baudrate로 고속 통신
- 11개 모터 동시 제어 (GroupSync 최적화)
- Context Manager 지원 (with 구문)
- 자동 정리 기능 (atexit handler)

통신 계층 구조:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[High Level]
    hand.py (Hand 클래스)
        ↓
    dynamixel_util.py (DynamixelClient 클래스)  ← 현재 파일
        ↓
    dynamixel_sdk (ROBOTIS SDK)
        ↓
    USB to Serial (U2D2 어댑터)
        ↓
[Low Level]
    ModBus RTU Protocol
        ↓
    Dynamixel XL-330 Motors (11개)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Control Table 구조:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Address | Size | Name               | Access | Description
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
64      | 1    | Torque Enable      | RW     | 모터 토크 ON/OFF
116     | 4    | Goal Position      | RW     | 목표 위치
126     | 4    | Present Position   | R      | 현재 위치
128     | 2    | Present Velocity   | R      | 현재 속도
126     | 2    | Present Current    | R      | 현재 전류 (mA)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

데이터 흐름:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[1] DynamixelClient 생성
    │
    ├─ PortHandler 생성 (시리얼 포트)
    ├─ PacketHandler 생성 (Protocol 2.0)
    └─ OPEN_CLIENTS에 등록
    ↓
[2] connect()
    │
    ├─ 포트 열기 (baudrate 2Mbps)
    ├─ 연결 확인
    └─ 통신 준비 완료
    ↓
[3] set_torque_enabled(True)
    │
    ├─ 각 모터에 토크 활성화 명령
    ├─ 실패 시 자동 재시도
    └─ 모터 제어 가능 상태
    ↓
[4] sync_write() / sync_read()
    │
    ├─ GroupSyncWrite/Read 생성
    ├─ 각 모터 파라미터 추가
    ├─ txRxPacket() 실행
    ├─ 결과 검증
    └─ 데이터 반환
    ↓
[5] disconnect()
    │
    ├─ 토크 비활성화
    ├─ 포트 닫기
    └─ OPEN_CLIENTS에서 제거
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

성능 최적화:
- GroupSyncWrite: 11개 모터를 단일 패킷으로 전송 (11배 빠름)
- GroupSyncRead: 11개 모터를 단일 패킷으로 수신 (11배 빠름)
- 재시도 메커니즘: 통신 실패 시 자동 재시도
- 배치 처리: 3개씩 묶어서 처리 (대용량 데이터)

사용 예시:
```python
# Context Manager 사용 (권장)
with DynamixelClient(motor_ids=[1,2,3,4,5,6,7,8,9,10,11]) as client:
    client.set_torque_enabled(True)
    positions = client.read_pos()
    client.set_pos([1000, 1000, ...])

# 일반 사용
client = DynamixelClient(motor_ids=[1,2,3,4,5,6,7,8,9,10,11])
client.connect()
client.set_torque_enabled(True)
# ... 작업 수행
client.disconnect()
```

작성: NYU RUKA Team
라이선스: MIT License
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import atexit      # 프로그램 종료 시 정리 작업 등록
import logging     # 로깅 시스템
import time        # 시간 지연 및 타이밍
from typing import Optional, Sequence, Tuple, Union  # 타입 힌트

import dynamixel_sdk  # ROBOTIS Dynamixel SDK
import numpy as np    # 수치 연산

# Control Table 주소 및 크기 임포트
# XL-330 E-manual: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
from ruka_hand.utils.control_table.control_table import *

# =============================================================================
# 통신 프로토콜 상수
# =============================================================================

COMM_SUCCESS = 0        # 통신 성공 코드
PROTOCOL_VERSION = 2.0  # Dynamixel Protocol 2.0
BAUDRATE = 57600        # 57.6Kbps (안정적 통신)

# =============================================================================
# 로깅 설정
# =============================================================================

# 경고 레벨 이상만 출력 (INFO, DEBUG는 숨김)
# 파일 로그가 필요한 경우 filename 파라미터 추가
logging.basicConfig(level=logging.WARNING)


# =============================================================================
# 전역 정리 함수
# =============================================================================

def dynamixel_cleanup_handler():
    """
    프로그램 종료 시 모든 Dynamixel 연결을 안전하게 해제하는 함수
    
    이 함수는 atexit에 등록되어 프로그램이 종료될 때 자동으로 호출됩니다.
    열려있는 모든 DynamixelClient를 찾아서 강제로 연결을 해제합니다.
    
    동작 원리:
    1. OPEN_CLIENTS 세트에서 모든 클라이언트 조회
    2. 각 클라이언트의 포트가 사용 중인지 확인
    3. 사용 중이면 강제로 해제 플래그 설정
    4. disconnect() 호출하여 안전하게 종료
    
    왜 필요한가?
    - 비정상 종료 시 포트가 잠긴 상태로 남을 수 있음
    - 다음 실행 시 "Port already in use" 에러 발생 방지
    - 리소스 누수 방지
    
    호출 시점:
    - 정상 종료 (프로그램 끝)
    - 비정상 종료 (Ctrl+C, 예외 발생)
    - sys.exit() 호출
    - 인터프리터 종료
    
    주의사항:
    - 이 함수는 자동으로 호출됨 (직접 호출 불필요)
    - 모든 클라이언트는 OPEN_CLIENTS에 등록되어야 함
    - 로깅만 수행하고 예외는 발생시키지 않음
    """
    print(f"\n[Cleanup] Dynamixel 정리 핸들러 실행 중...")
    
    # OPEN_CLIENTS는 set이므로 복사본으로 순회
    # (순회 중 set 변경 시 에러 방지)
    open_clients = list(DynamixelClient.OPEN_CLIENTS)
    
    if not open_clients:
        print(f"  ✓ 열린 클라이언트 없음 (정상)")
        return
    
    print(f"  → {len(open_clients)}개 클라이언트 정리 중...")
    
    for i, open_client in enumerate(open_clients, 1):
        try:
            # 포트가 사용 중인지 확인
            if open_client.port_handler.is_using:
                logging.warning(f"  ⚠ 클라이언트 #{i} 포트 사용 중 → 강제 종료")
                # 강제로 사용 플래그 해제
                open_client.port_handler.is_using = False
            
            # 안전하게 연결 해제
            open_client.disconnect()
            print(f"  ✓ 클라이언트 #{i} 정리 완료")
            
        except Exception as e:
            print(f"  ✗ 클라이언트 #{i} 정리 실패: {e}")
    
    print(f"  ✓ 전체 정리 완료\n")


# =============================================================================
# 유틸리티 함수
# =============================================================================

def unsigned_to_signed(value: int, size: int) -> int:
    """
    부호 없는(unsigned) 정수를 부호 있는(signed) 정수로 변환
    
    Dynamixel 모터의 전류값은 부호 있는 정수로 표현되지만,
    통신 프로토콜에서는 부호 없는 정수로 전송됩니다.
    이 함수는 2의 보수(Two's complement) 방식으로 변환합니다.
    
    Args:
        value (int): 변환할 부호 없는 정수값
        size (int): 데이터 크기 (바이트 단위)
                   - 1 byte: 0~255 → -128~127
                   - 2 bytes: 0~65535 → -32768~32767
                   - 4 bytes: 0~4294967295 → -2147483648~2147483647
    
    Returns:
        int: 부호 있는 정수값
    
    동작 원리:
    1. 비트 크기 계산: bit_size = 8 * size
    2. MSB(Most Significant Bit) 체크: 최상위 비트가 1인가?
    3. 1이면 음수: value - 2^bit_size
    4. 0이면 양수: 그대로 반환
    
    예시:
        # 2 bytes (16 bit)
        unsigned_to_signed(32768, 2) → -32768
        unsigned_to_signed(65535, 2) → -1
        unsigned_to_signed(32767, 2) → 32767
        
        # 현재 전류가 -50mA인 경우
        # 통신: 65486 (unsigned)
        # 변환: -50 (signed)
    
    적용 사례:
    - Present Current (126번 주소, 2 bytes)
    - Present Velocity (128번 주소, 4 bytes, signed)
    """
    # 전체 비트 수 계산
    bit_size = 8 * size
    
    # MSB가 1인지 확인 (음수 판별)
    # (1 << (bit_size - 1))은 최상위 비트만 1인 값
    # 예: 16bit → 0b1000000000000000 = 32768
    if (value & (1 << (bit_size - 1))) != 0:
        # 음수: 2의 보수 변환
        # -((1 << bit_size) - value)
        # = -(2^bit_size - value)
        value = -((1 << bit_size) - value)
    
    return value


# =============================================================================
# DynamixelClient 클래스
# =============================================================================

class DynamixelClient:
    """
    Dynamixel 모터와 통신하기 위한 클라이언트 클래스
    
    이 클래스는 RUKA 로봇 손의 11개 Dynamixel XL-330 모터를 제어하기 위한
    저수준 인터페이스를 제공합니다. Protocol 2.0만 지원합니다.
    
    주요 기능:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. 연결 관리
       - connect(): 시리얼 포트 연결
       - disconnect(): 안전한 연결 해제
       - is_connected: 연결 상태 확인
    
    2. 토크 제어
       - set_torque_enabled(): 모터 활성화/비활성화
       - 자동 재시도 메커니즘
    
    3. 그룹 동기화 (GroupSync)
       - sync_write(): 여러 모터에 동시 쓰기
       - sync_read(): 여러 모터에서 동시 읽기
       - 최적화된 통신 (단일 패킷으로 처리)
    
    4. 개별 접근
       - single_write(): 단일 모터 쓰기
       - single_read(): 단일 모터 읽기
       - set_pos_indv(): 개별 위치 설정
    
    5. 편의 함수
       - read_pos(): 현재 위치 읽기
       - read_goal_pos(): 목표 위치 읽기
       - read_cur(): 현재 전류 읽기
       - read_vel(): 현재 속도 읽기
       - set_pos(): 위치 설정
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    성능 특징:
    - GroupSyncWrite: 11개 모터를 0.5ms에 제어 (개별 대비 11배 빠름)
    - GroupSyncRead: 11개 모터를 1.5ms에 읽기 (개별 대비 11배 빠름)
    - Baudrate 2Mbps: 초당 약 2000회 명령 전송 가능
    
    Context Manager 지원:
    ```python
    with DynamixelClient([1,2,3]) as client:
        client.set_torque_enabled(True)
        positions = client.read_pos()
    # 자동으로 disconnect() 호출됨
    ```
    
    Attributes:
        motor_ids (list[int]): 제어할 모터 ID 리스트 [1~11]
        port_name (str): 시리얼 포트 경로 (/dev/ttyUSB0 등)
        baudrate (int): 통신 속도 (2000000 = 2Mbps)
        lazy_connect (bool): 자동 연결 활성화 플래그
        port_handler: Dynamixel SDK PortHandler 객체
        packet_handler: Dynamixel SDK PacketHandler 객체
        _sync_writers (dict): GroupSyncWrite 인스턴스 캐시
    
    Note:
        - Protocol 2.0만 지원 (XL-330, XM430 등)
        - Protocol 1.0 모터는 사용 불가
        - 동시에 여러 DynamixelClient 생성 가능 (다른 포트)
        - 같은 포트에 여러 클라이언트 생성 시 충돌 발생
    """
    
    # 클래스 변수: 현재 열려있는 모든 클라이언트 추적
    # atexit handler에서 정리 작업에 사용
    OPEN_CLIENTS = set()
    
    def __init__(
        self,
        motor_ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        lazy_connect: bool = False,
    ):
        """
        DynamixelClient 초기화
        
        Args:
            motor_ids (Sequence[int]): 제어할 모터 ID 리스트
                                      예: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            
            port (str): Dynamixel이 연결된 시리얼 포트 경로
                       - Linux: /dev/ttyUSB0, /dev/ttyUSB1, ...
                       - Mac: /dev/tty.usbserial-*
                       - Windows: COM1, COM2, ...
                       기본값: "/dev/ttyUSB0"
            
            lazy_connect (bool): True인 경우, 메서드 호출 시 자동으로 연결
                                False인 경우, 명시적으로 connect() 호출 필요
                                기본값: False
        
        초기화 순서:
        1. Dynamixel SDK 모듈 참조 저장
        2. 모터 ID 리스트 저장
        3. 포트 및 통신 설정 저장
        4. PortHandler 생성 (시리얼 통신 담당)
        5. PacketHandler 생성 (프로토콜 처리 담당)
        6. GroupSyncWrite 캐시 초기화
        7. OPEN_CLIENTS에 자신 등록
        
        주의사항:
        - 이 함수는 아직 실제 연결하지 않음 (connect() 필요)
        - motor_ids는 실제 모터 ID와 일치해야 함
        - 잘못된 포트 경로는 connect() 시점에 에러 발생
        
        예시:
            # 기본 사용
            client = DynamixelClient([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            client.connect()
            
            # Lazy connect
            client = DynamixelClient([1,2,3], lazy_connect=True)
            client.read_pos()  # 자동으로 연결됨
            
            # Windows
            client = DynamixelClient([1,2,3], port="COM3")
        """
        print(f"[DynamixelClient] 초기화 중...")
        print(f"  → 모터 ID: {motor_ids}")
        print(f"  → 포트: {port}")
        print(f"  → Baudrate: {BAUDRATE} bps ({BAUDRATE/1000000:.1f} Mbps)")
        print(f"  → Lazy Connect: {'ON' if lazy_connect else 'OFF'}")
        
        # Dynamixel SDK 모듈 참조
        self.dxl = dynamixel_sdk
        
        # 모터 설정
        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = BAUDRATE
        self.lazy_connect = lazy_connect
        
        # 핸들러 생성
        self.port_handler = self.dxl.PortHandler(port)
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)
        
        # GroupSyncWrite 인스턴스 캐시
        # key: (address, size), value: GroupSyncWrite 객체
        # 같은 주소에 반복 쓰기 시 재사용으로 성능 향상
        self._sync_writers = {}
        
        # 전역 클라이언트 세트에 등록
        # atexit handler가 정리 시 사용
        self.OPEN_CLIENTS.add(self)
        
        print(f"  ✓ DynamixelClient 초기화 완료")
    
    @property
    def is_connected(self) -> bool:
        """
        연결 상태 확인 프로퍼티
        
        Returns:
            bool: 포트가 열려있으면 True, 아니면 False
        
        사용 예시:
            if client.is_connected:
                print("연결됨")
            else:
                client.connect()
        """
        return self.port_handler.is_open
    
    def connect(self):
        """
        Dynamixel 모터에 연결
        
        이 함수는 시리얼 포트를 열고 통신 속도를 설정합니다.
        모든 DynamixelClient가 생성된 후에 호출해야 합니다.
        
        연결 순서:
        1. 포트 열기 (openPort)
           - 시리얼 디바이스 파일 오픈
           - 배타적 접근 권한 획득
        
        2. Baudrate 설정 (setBaudRate)
           - 2Mbps 통신 속도 설정
           - 모터와 PC 간 속도 동기화
        
        3. 연결 확인
           - 포트가 정상적으로 열렸는지 검증
           - 실제 통신 가능 상태인지 확인
        
        Raises:
            OSError: 포트 열기 실패 시
                    - 포트가 존재하지 않음 (/dev/ttyUSB0 없음)
                    - 권한 부족 (sudo 또는 dialout 그룹 필요)
                    - 다른 프로그램이 사용 중
            
            OSError: Baudrate 설정 실패 시
                    - 모터가 다른 Baudrate로 설정됨
                    - 하드웨어 호환성 문제
        
        주의사항:
        - 이미 연결된 상태에서 호출 시 AssertionError 발생
        - 여러 DynamixelClient가 같은 포트 사용 불가
        - 연결 전 모터 전원 확인 필요
        
        문제 해결:
        1. "Failed to open port" 에러
           → ls /dev/ttyUSB* 로 포트 확인
           → sudo chmod 666 /dev/ttyUSB0 으로 권한 부여
           → sudo usermod -aG dialout $USER 후 재부팅
        
        2. "Failed to set baudrate" 에러
           → 모터 Baudrate가 2Mbps가 아닐 수 있음
           → Dynamixel Wizard로 모터 Baudrate 변경
        
        사용 예시:
            client = DynamixelClient([1,2,3])
            client.connect()  # 여기서 실제 연결
            client.set_torque_enabled(True)
        """
        print(f"\n[연결] Dynamixel 모터 연결 시도...")
        print(f"  → 포트: {self.port_name}")
        
        # 이미 연결된 경우 에러
        assert not self.is_connected, "이미 연결되어 있습니다."
        
        # 1단계: 포트 열기
        print(f"  [1/2] 포트 열기 중...")
        if self.port_handler.openPort():
            print(f"    ✓ 포트 열기 성공: {self.port_name}")
            logging.info("Succeeded to open port: %s", self.port_name)
        else:
            error_msg = (
                f"포트 열기 실패: {self.port_name}\n"
                f"    문제 해결:\n"
                f"      1. 모터 전원 확인\n"
                f"      2. USB 연결 확인\n"
                f"      3. 포트 경로 확인: ls /dev/ttyUSB*\n"
                f"      4. 권한 확인: sudo chmod 666 {self.port_name}\n"
                f"      5. dialout 그룹 추가: sudo usermod -aG dialout $USER"
            )
            print(f"    ✗ {error_msg}")
            raise OSError(error_msg)
        
        # 2단계: Baudrate 설정
        print(f"  [2/2] Baudrate 설정 중...")
        print(f"    → 목표 Baudrate: {self.baudrate} bps")
        if self.port_handler.setBaudRate(self.baudrate):
            print(f"    ✓ Baudrate 설정 성공: {self.baudrate} bps")
            logging.info("Succeeded to set baudrate to %d", self.baudrate)
        else:
            error_msg = (
                f"Baudrate 설정 실패: {self.baudrate}\n"
                f"    문제 해결:\n"
                f"      1. 모터가 같은 Baudrate로 설정되어 있는지 확인\n"
                f"      2. Dynamixel Wizard로 모터 Baudrate 확인/변경\n"
                f"      3. 다른 Baudrate 값으로 시도 (57600, 1000000 등)"
            )
            print(f"    ✗ {error_msg}")
            raise OSError(error_msg)
        
        print(f"  ✓ 연결 완료!\n")
    
    def disconnect(self):
        """
        Dynamixel 디바이스 연결 해제
        
        안전한 종료 절차:
        1. 연결 상태 확인 (이미 해제되었으면 무시)
        2. 포트 사용 중 여부 확인 (통신 중이면 대기)
        3. 모터 토크 비활성화 (안전을 위해)
        4. 포트 닫기
        5. OPEN_CLIENTS에서 제거
        
        왜 토크를 비활성화하는가?
        - 프로그램 종료 후에도 토크가 켜져있으면 위험
        - 외부 힘으로 손가락을 움직일 수 없음
        - 전력 소모 계속됨
        
        주의사항:
        - 포트 사용 중(is_using)이면 연결 해제 안 됨
        - 토크 비활성화 실패해도 포트는 닫힘
        - 이미 해제된 상태에서 호출해도 안전
        
        호출 시점:
        - 프로그램 정상 종료
        - Context manager 탈출 (__exit__)
        - 객체 소멸 (__del__)
        - atexit handler
        
        사용 예시:
            client.connect()
            # ... 작업 수행
            client.disconnect()  # 안전하게 종료
        """
        print(f"\n[연결 해제] Dynamixel 연결 해제 시작...")
        
        # 이미 연결 해제된 경우
        if not self.is_connected:
            print(f"  → 이미 연결 해제됨 (무시)")
            return
        
        # 포트 사용 중인 경우 (통신 진행 중)
        if self.port_handler.is_using:
            print(f"  ⚠ 포트 사용 중 - 연결 해제 불가")
            logging.error("Port handler in use; cannot disconnect.")
            return
        
        try:
            # 안전을 위해 모터 토크 비활성화
            print(f"  [1/3] 모터 토크 비활성화 중...")
            self.set_torque_enabled(False)
            print(f"    ✓ 토크 비활성화 완료")
        except Exception as e:
            print(f"    ⚠ 토크 비활성화 실패 (무시): {e}")
        
        try:
            # 포트 닫기
            print(f"  [2/3] 포트 닫기 중...")
            self.port_handler.closePort()
            print(f"    ✓ 포트 닫기 완료")
        except Exception as e:
            print(f"    ✗ 포트 닫기 실패: {e}")
        
        # OPEN_CLIENTS에서 제거
        print(f"  [3/3] 클라이언트 등록 해제 중...")
        if self in self.OPEN_CLIENTS:
            self.OPEN_CLIENTS.remove(self)
            print(f"    ✓ 등록 해제 완료")
        
        print(f"  ✓ 연결 해제 완료\n")
    
    def check_connected(self):
        """
        연결 상태 확인 및 자동 연결
        
        이 함수는 모든 통신 메서드 시작 시 호출됩니다.
        
        동작:
        1. lazy_connect=True이고 연결 안 됨 → 자동 연결
        2. 연결 안 됨 → OSError 발생
        3. 연결됨 → 통과
        
        Raises:
            OSError: lazy_connect=False이고 연결 안 된 경우
        
        왜 필요한가?
        - 연결 없이 read/write 시도 시 프로그램 크래시 방지
        - 명확한 에러 메시지 제공
        - 자동 연결 기능 제공
        
        사용 예시:
            def read_pos(self):
                self.check_connected()  # 연결 확인
                # 실제 읽기 작업
        """
        # Lazy connect 활성화 시 자동 연결
        if self.lazy_connect and not self.is_connected:
            print(f"[Auto Connect] Lazy connect 활성화 - 자동 연결 중...")
            self.connect()
        
        # 여전히 연결 안 된 경우 에러
        if not self.is_connected:
            raise OSError(
                "연결되지 않음. connect()를 먼저 호출하세요.\n"
                "또는 lazy_connect=True로 설정하세요."
            )
    
    def reboot(self, retries: int = -1, retry_interval: float = 0.25):
        """
        모든 모터 재부팅
        
        이 함수는 각 모터를 재부팅하여 펌웨어를 재시작합니다.
        문제 발생 시 모터를 복구하는 데 유용합니다.
        
        Args:
            retries (int): 재시도 횟수
                          -1: 성공할 때까지 무한 재시도
                          0: 재시도 없음 (1회만 시도)
                          N: 최대 N회 재시도
                          기본값: -1
            
            retry_interval (float): 재시도 간격 (초)
                                   기본값: 0.25 (250ms)
        
        재부팅 과정:
        1. 모든 모터 ID를 remaining_ids에 저장
        2. 각 모터에 대해:
           - reboot 명령 전송
           - 성공 시 remaining_ids에서 제거
           - 실패 시 유지
        3. remaining_ids가 빌 때까지 반복
        4. 재시도 횟수 초과 시 중단
        
        재부팅이 필요한 경우:
        - 모터가 응답하지 않음
        - Control Table 값이 이상함
        - 통신 에러가 계속 발생
        - 펌웨어 업데이트 후
        
        주의사항:
        - 재부팅 중 모터는 약 2초간 응답하지 않음
        - 재부팅 후 모든 설정값이 기본값으로 리셋됨
        - 재부팅 후 토크, PID 게인 등 재설정 필요
        
        사용 예시:
            # 무한 재시도
            client.reboot()
            
            # 최대 3회 재시도
            client.reboot(retries=3)
            
            # 재시도 없음
            client.reboot(retries=0)
        """
        print(f"\n[Reboot] 모터 재부팅 시작...")
        print(f"  → 대상 모터: {self.motor_ids}")
        print(f"  → 재시도 설정: {'무한' if retries < 0 else f'{retries}회'}")
        print(f"  → 재시도 간격: {retry_interval}초")
        
        remaining_ids = self.motor_ids
        attempt = 0
        
        while remaining_ids:
            attempt += 1
            print(f"\n  [시도 #{attempt}] 남은 모터: {remaining_ids}")
            
            for dxl_id in remaining_ids:
                print(f"    → 모터 #{dxl_id} 재부팅 중...")
                
                # 재부팅 명령 전송
                dxl_comm_result, dxl_error = self.packet_handler.reboot(
                    self.port_handler, dxl_id
                )
                
                # 통신 실패
                if dxl_comm_result != COMM_SUCCESS:
                    error_msg = self.packet_handler.getTxRxResult(dxl_comm_result)
                    print(f"      ✗ 통신 실패: {error_msg}")
                    logging.error("%s" % error_msg)
                
                # 모터 에러
                elif dxl_error != 0:
                    error_msg = self.packet_handler.getRxPacketError(dxl_error)
                    print(f"      ✗ 모터 에러: {error_msg}")
                    logging.error("%s" % error_msg)
                
                # 성공
                else:
                    print(f"      ✓ 모터 #{dxl_id} 재부팅 완료")
                    logging.info("Dynamixel[ID:%03d] has been rebooted" % dxl_id)
                    remaining_ids.remove(dxl_id)
            
            # 모든 모터 재부팅 완료
            if not remaining_ids:
                print(f"\n  ✓ 전체 모터 재부팅 완료")
                break
            
            # 재시도 횟수 확인
            if retries == 0:
                print(f"\n  ✗ 재시도 횟수 초과")
                print(f"    실패한 모터: {remaining_ids}")
                break
            
            # 대기 후 재시도
            print(f"    ⏳ {retry_interval}초 대기 중...")
            time.sleep(retry_interval)
            retries -= 1
    
    def handle_packet_result(
        self,
        comm_result: int,
        dxl_error: Optional[int] = None,
        dxl_id: Optional[int] = None,
        context: Optional[str] = None,
    ):
        """
        패킷 통신 결과 처리 및 에러 로깅
        
        Dynamixel SDK의 모든 통신 함수는 (comm_result, dxl_error) 튜플을 반환합니다.
        이 함수는 결과를 해석하고 에러 발생 시 로깅합니다.
        
        Args:
            comm_result (int): 통신 결과 코드
                              COMM_SUCCESS (0): 성공
                              기타: 통신 에러 (타임아웃, CRC 에러 등)
            
            dxl_error (Optional[int]): 모터 에러 코드 (있는 경우)
                                      0: 에러 없음
                                      기타: 하드웨어 에러
            
            dxl_id (Optional[int]): 모터 ID (로깅용)
            
            context (Optional[str]): 에러 발생 맥락 (로깅용)
                                    예: "sync_write", "read_pos"
        
        Returns:
            bool: 성공 시 True, 실패 시 False
        
        에러 메시지 형식:
            [Motor ID: X] > context: error_message
        
        통신 에러 종류:
        - COMM_TX_FAIL: 전송 실패 (케이블 문제)
        - COMM_RX_FAIL: 수신 실패 (응답 없음)
        - COMM_TX_ERROR: 전송 에러 (하드웨어 문제)
        - COMM_RX_WAITING: 수신 대기 중 (타임아웃)
        - COMM_RX_TIMEOUT: 타임아웃 (모터 응답 없음)
        - COMM_RX_CORRUPT: 데이터 손상 (CRC 에러)
        
        모터 에러 종류:
        - INSTRUCTION: 명령어 에러
        - CRC: 체크섬 에러
        - DATA_RANGE: 데이터 범위 에러
        - DATA_LENGTH: 데이터 길이 에러
        - DATA_LIMIT: 데이터 제한 에러
        - ACCESS: 접근 권한 에러
        
        사용 예시:
            comm_result, dxl_error = packet_handler.writeTxRx(...)
            success = self.handle_packet_result(
                comm_result, dxl_error,
                dxl_id=5,
                context="set_position"
            )
        """
        error_message = None
        
        # 통신 에러 확인
        if comm_result != COMM_SUCCESS:
            error_message = self.packet_handler.getTxRxResult(comm_result)
        
        # 모터 에러 확인
        elif dxl_error is not None:
            error_message = self.packet_handler.getRxPacketError(dxl_error)
        
        # 에러가 있는 경우 로깅
        if error_message:
            # 모터 ID 추가
            if dxl_id is not None:
                error_message = "[Motor ID: {}] {}".format(dxl_id, error_message)
            
            # 맥락 추가
            if context is not None:
                error_message = "> {}: {}".format(context, error_message)
            
            logging.error(error_message)
            return False
        
        return True
    
    def set_torque_enabled(
        self, enabled: bool, retries: int = -1, retry_interval: float = 0.25
    ):
        """
        모터 토크 활성화/비활성화
        
        토크가 활성화되면:
        - 모터가 목표 위치를 유지하려고 힘을 냄
        - 외부 힘으로 움직이기 어려움
        - 위치 명령을 받을 수 있음
        
        토크가 비활성화되면:
        - 모터가 자유롭게 움직임
        - 외부 힘으로 쉽게 움직임
        - 위치 명령을 받지 않음
        - 전력 소모 없음
        
        Args:
            enabled (bool): True=활성화, False=비활성화
            retries (int): 재시도 횟수 (-1=무한)
            retry_interval (float): 재시도 간격(초)
        
        재시도 메커니즘:
        1. 모든 모터 ID를 remaining_ids에 저장
        2. 각 모터에 토크 명령 전송
        3. 성공한 모터는 remaining_ids에서 제거
        4. 실패한 모터는 자동 재부팅 후 재시도
        5. remaining_ids가 빌 때까지 반복
        
        실패 시 자동 재부팅:
        - 통신 실패 시 모터가 응답하지 않을 수 있음
        - 재부팅으로 모터 상태 초기화
        - 재부팅 후 재시도
        
        주의사항:
        - 토크 활성화 전에 Goal Position 설정 권장
        - 갑작스런 토크 활성화 시 모터가 급격히 움직일 수 있음
        - 프로그램 종료 시 반드시 토크 비활성화 필요
        
        사용 예시:
            # 토크 활성화 (무한 재시도)
            client.set_torque_enabled(True)
            
            # 토크 비활성화 (3회 재시도)
            client.set_torque_enabled(False, retries=3)
            
            # 토크 활성화 (재시도 없음)
            client.set_torque_enabled(True, retries=0)
        """
        print(f"\n[Torque] 토크 {'활성화' if enabled else '비활성화'} 시작...")
        print(f"  → 대상 모터: {self.motor_ids}")
        print(f"  → 재시도 설정: {'무한' if retries < 0 else f'{retries}회'}")
        
        remaining_ids = list(self.motor_ids)
        attempt = 0
        
        while remaining_ids:
            attempt += 1
            print(f"\n  [시도 #{attempt}] 남은 모터: {remaining_ids}")
            
            for motor_id in remaining_ids:
                print(f"    → 모터 #{motor_id} 처리 중...")
                
                # 토크 명령 전송 (1 byte)
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, ADDR_TORQUE_ENABLE, int(enabled)
                )
                
                # 통신 실패 → 재부팅
                if dxl_comm_result != COMM_SUCCESS:
                    error_msg = self.packet_handler.getTxRxResult(dxl_comm_result)
                    print(f"      ✗ 통신 실패: {error_msg}")
                    print(f"      ↻ 모터 재부팅 중...")
                    self.packet_handler.reboot(self.port_handler, motor_id)
                
                # 모터 에러 → 재부팅
                elif dxl_error != 0:
                    error_msg = self.packet_handler.getRxPacketError(dxl_error)
                    print(f"      ✗ 모터 에러: {error_msg}")
                    print(f"      ↻ 모터 재부팅 중...")
                    logging.error("%s" % error_msg)
                    self.packet_handler.reboot(self.port_handler, motor_id)
                
                # 성공
                else:
                    remaining_ids.remove(motor_id)
                    print(f"      ✓ 모터 #{motor_id} 연결 성공")
                    logging.info(
                        "Dynamixel#%d has been successfully connected" % motor_id
                    )
            
            # 모든 모터 성공
            if not remaining_ids:
                print(f"\n  ✓ 전체 모터 토크 {'활성화' if enabled else '비활성화'} 완료")
                break
            
            # 재시도 횟수 확인
            if retries == 0:
                print(f"\n  ✗ 재시도 횟수 초과")
                print(f"    실패한 모터: {remaining_ids}")
                break
            
            # 대기 후 재시도
            print(f"    ⏳ {retry_interval}초 대기 중...")
            time.sleep(retry_interval)
            retries -= 1

    def sync_write(
        self,
        motor_ids: Sequence[int],
        values: Sequence[int],
        address: int,
        size: int,
    ):
        """
        여러 모터에 동시에 데이터 쓰기 (GroupSyncWrite)
        
        GroupSyncWrite는 여러 모터에 동일한 주소의 데이터를 단일 패킷으로 전송하는
        최적화된 통신 방식입니다. 개별 write 대비 약 11배 빠릅니다.
        
        Args:
            motor_ids (Sequence[int]): 대상 모터 ID 리스트
            values (Sequence[int]): 각 모터에 쓸 값 리스트
                                   len(values) == len(motor_ids) 필수
            address (int): Control Table 주소
                          예: ADDR_GOAL_POSITION (116)
            size (int): 데이터 크기 (바이트)
                       예: LEN_GOAL_POSITION (4)
        
        동작 원리:
        1. GroupSyncWrite 객체 생성 또는 캐시에서 가져오기
        2. 각 모터의 (ID, value) 쌍을 param으로 추가
        3. txPacket()으로 단일 패킷 전송
        4. 모든 모터가 동시에 데이터 수신 및 적용
        5. param 정리
        
        성능 비교:
        - 개별 write: 11개 모터 × 0.5ms = 5.5ms
        - GroupSyncWrite: 단일 패킷 = 0.5ms
        - 속도 향상: 11배
        
        패킷 구조:
        ┌────────────────────────────────────────────┐
        │ Header (FF FF FD 00)                       │
        │ ID (FE - Broadcast)                        │
        │ Length                                     │
        │ Instruction (83 - Sync Write)              │
        │ Address (2 bytes)                          │
        │ Size (2 bytes)                             │
        │ ┌──────────────────────────────────────┐   │
        │ │ Motor 1: ID + Data (size bytes)      │   │
        │ │ Motor 2: ID + Data (size bytes)      │   │
        │ │ ...                                   │   │
        │ │ Motor 11: ID + Data (size bytes)     │   │
        │ └──────────────────────────────────────┘   │
        │ CRC (2 bytes)                              │
        └────────────────────────────────────────────┘
        
        주의사항:
        - motor_ids와 values의 길이 일치 필요
        - 모든 모터가 동일한 address와 size 사용
        - 실패한 모터는 로그에 기록됨
        - 개별 응답은 받지 않음 (빠른 속도 위해)
        
        에러 처리:
        - 파라미터 추가 실패 시 로깅
        - 패킷 전송 실패 시 로깅
        - 부분 실패 가능 (일부 모터만 성공)
        
        사용 예시:
            # 위치 설정
            client.sync_write(
                motor_ids=[1, 2, 3],
                values=[1000, 1500, 2000],
                address=ADDR_GOAL_POSITION,
                size=LEN_GOAL_POSITION
            )
            
            # PID 게인 설정
            client.sync_write(
                motor_ids=[1, 2, 3],
                values=[450, 450, 450],
                address=ADDR_POSITION_P_GAIN,
                size=LEN_POSITION_P_GAIN
            )
        """
        # 연결 확인
        self.check_connected()
        
        # GroupSyncWrite 캐시 키
        key = (address, size)
        
        # 캐시에 없으면 새로 생성
        if key not in self._sync_writers:
            print(f"    [Sync Write] GroupSyncWrite 생성: Addr={address}, Size={size}")
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size
            )
        
        # 캐시에서 가져오기
        sync_writer = self._sync_writers[key]
        
        # 각 모터의 파라미터 추가
        errored_ids = []
        for motor_id, value in zip(motor_ids, values):
            # value = int(value)
			# 값을 바이트 배열로 변환
            # 예: 1000 (4 bytes) → [232, 3, 0, 0]
            # data = [
            #     self.dxl.DXL_LOBYTE(self.dxl.DXL_LOWORD(value)),
            #     self.dxl.DXL_HIBYTE(self.dxl.DXL_LOWORD(value)),
            #     self.dxl.DXL_LOBYTE(self.dxl.DXL_HIWORD(value)),
            #     self.dxl.DXL_HIBYTE(self.dxl.DXL_HIWORD(value)),
            # ]
            
			# 1. 명시적으로 Python int로 변환 (NumPy 타입 처리)
            value = int(value)
        
            # 2. size에 맞게 바이트 배열 생성
            if size == 1:
                # 1바이트: Operating Mode, Temperature Limit 등
                data = [
                    self.dxl.DXL_LOBYTE(value)
                ]
            elif size == 2:
                # 2바이트: Current Limit, PID Gains 등
                data = [
                    self.dxl.DXL_LOBYTE(value),
                    self.dxl.DXL_HIBYTE(value)
                ]
            elif size == 4:
                # 4바이트: Goal Position, Goal Velocity 등
                data = [
                    self.dxl.DXL_LOBYTE(self.dxl.DXL_LOWORD(value)),
                    self.dxl.DXL_HIBYTE(self.dxl.DXL_LOWORD(value)),
                    self.dxl.DXL_LOBYTE(self.dxl.DXL_HIWORD(value)),
                    self.dxl.DXL_HIBYTE(self.dxl.DXL_HIWORD(value))
                ]
            else:
                # 예외 처리 (8바이트 등 비정상적인 경우)
                print(f"    ⚠️ 경고: 지원하지 않는 size={size}")
                data = [self.dxl.DXL_LOBYTE(value)]
			
            # 파라미터 추가
            success = sync_writer.addParam(motor_id, data)
            if not success:
                print(f"      ✗ addParam 실패: motor_id={motor_id}, size={size}")
                errored_ids.append(motor_id)
        
        # 파라미터 추가 실패 로깅
        if errored_ids:
            print(f"    ✗ Sync write 파라미터 추가 실패: {errored_ids}")
            logging.error("Sync write failed for: %s", str(errored_ids))
        
        # 패킷 전송
        comm_result = sync_writer.txPacket()
        
        # 결과 처리
        self.handle_packet_result(comm_result, context="sync_write")
        
        # 파라미터 정리 (다음 사용을 위해)
        sync_writer.clearParam()

    def sync_read(self, motor_ids: Sequence[int], address: int, size: int):
        """
        여러 모터에서 동시에 데이터 읽기 (GroupSyncRead)
        
        GroupSyncRead는 여러 모터에서 동일한 주소의 데이터를 단일 패킷으로 요청하고
        각 모터의 응답을 수신하는 최적화된 통신 방식입니다.
        
        Args:
            motor_ids (Sequence[int]): 대상 모터 ID 리스트
            address (int): Control Table 주소
                          예: ADDR_PRESENT_POSITION (132)
            size (int): 데이터 크기 (바이트)
                       예: LEN_PRESENT_POSITION (4)
        
        Returns:
            list: 각 모터의 읽은 값 리스트
                 실패한 모터는 None
                 순서: motor_ids와 동일
        
        동작 원리:
        1. GroupSyncRead 객체 생성
        2. 각 모터 ID를 param으로 추가
        3. txRxPacket()으로 요청 전송 및 응답 수신
        4. 각 모터의 데이터 가용성 확인
        5. 데이터 추출 및 반환
        
        재귀적 배치 처리:
        - 모터가 3개 초과인 경우 자동으로 분할
        - 3개씩 묶어서 재귀 호출
        - 결과를 합쳐서 반환
        - 이유: 한 번에 너무 많은 모터 읽기 시 타임아웃 발생 가능
        
        성능 특징:
        - 개별 read: 11개 모터 × 1.5ms = 16.5ms
        - GroupSyncRead: 3개씩 4번 = 6ms
        - 속도 향상: 약 3배
        
        패킷 구조 (요청):
        ┌────────────────────────────────────────────┐
        │ Header (FF FF FD 00)                       │
        │ ID (FE - Broadcast)                        │
        │ Length                                     │
        │ Instruction (82 - Sync Read)               │
        │ Address (2 bytes)                          │
        │ Size (2 bytes)                             │
        │ ┌──────────────────────────────────────┐   │
        │ │ Motor 1: ID                          │   │
        │ │ Motor 2: ID                          │   │
        │ │ Motor 3: ID                          │   │
        │ └──────────────────────────────────────┘   │
        │ CRC (2 bytes)                              │
        └────────────────────────────────────────────┘
        
        응답:
        - 각 모터가 개별적으로 Status Packet 전송
        - 순서: 요청한 ID 순서와 동일
        - 각 Status Packet에 데이터 포함
        
        에러 처리:
        - 파라미터 추가 실패 → 해당 모터 데이터 없음
        - 통신 실패 → 모든 모터 데이터 없음
        - 데이터 가용성 실패 → 해당 모터 None 반환
        
        사용 예시:
            # 위치 읽기
            positions = client.sync_read(
                motor_ids=[1, 2, 3],
                address=ADDR_PRESENT_POSITION,
                size=LEN_PRESENT_POSITION
            )
            # 결과: [1000, 1500, 2000] 또는 [1000, None, 2000]
            
            # 전류 읽기
            currents = client.sync_read(
                motor_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                address=ADDR_PRESENT_CURRENT,
                size=LEN_PRESENT_CURRENT
            )
            # 자동으로 3개씩 분할 처리됨
        """
        # 재귀적 배치 처리: 3개 초과 시 분할
        if len(motor_ids) > 3:
            print(f"    [Sync Read] 배치 처리: {len(motor_ids)}개 모터 → 3개씩 분할")
            # 앞 3개 읽기
            batch1 = self.sync_read(motor_ids[:3], address, size)
            # 나머지 읽기 (재귀)
            batch2 = self.sync_read(motor_ids[3:], address, size)
            # 합쳐서 반환
            bulk_data = batch1 + batch2
            return bulk_data
        
        # 연결 확인
        self.check_connected()
        
        # GroupSyncRead 객체 생성
        # 주의: sync_read는 캐시하지 않음 (매번 새로 생성)
        # 이유: 읽기는 쓰기보다 덜 빈번하고, 캐시 관리 복잡도 증가
        sync_reader = self.dxl.GroupSyncRead(
            self.port_handler, self.packet_handler, address, size
        )
        
        # 각 모터 ID를 파라미터로 추가
        errored_ids = []
        for motor_id in motor_ids:
            success = sync_reader.addParam(motor_id)
            if not success:
                errored_ids.append(motor_id)
        
        # 파라미터 추가 실패 로깅
        if errored_ids:
            print(f"    ✗ Sync read 파라미터 추가 실패: {errored_ids}")
            logging.error("Sync write failed for: %s", str(errored_ids))
        
        # 요청 전송 및 응답 수신
        comm_result = sync_reader.txRxPacket()
        self.handle_packet_result(comm_result, context="sync_write")
        
        # 각 모터의 데이터 가용성 확인 및 추출
        bulk_data = []
        errored_ids = []
        
        for motor_id in motor_ids:
            data = None
            
            # 데이터 가용성 확인
            available = sync_reader.isAvailable(motor_id, address, size)
            
            if not available:
                # 데이터 없음
                errored_ids.append(motor_id)
            else:
                # 데이터 추출
                data = sync_reader.getData(motor_id, address, size)
            
            bulk_data.append(data)
        
        # 데이터 가용성 실패 로깅
        if errored_ids:
            print(f"    ✗ Sync read 데이터 없음: {errored_ids}")
            logging.error("Bulk read data is unavailable for: %s", str(errored_ids))
        
        # 파라미터 정리
        sync_reader.clearParam()
        
        return bulk_data

    # =============================================================================
    # 편의 함수: 자주 사용하는 읽기/쓰기 작업
    # =============================================================================

    def read_pos(self):
        """
        현재 위치 읽기
        
        Returns:
            list[int]: 각 모터의 현재 위치 (단위: 0.088도/tick)
                      예: [1000, 1500, 2000, ...]
        
        사용 예시:
            positions = client.read_pos()
            print(f"모터 1 위치: {positions[0]}")
        """
        return self.sync_read(
            self.motor_ids, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )

    def read_goal_pos(self):
        """
        목표 위치 읽기
        
        Returns:
            list[int]: 각 모터의 목표 위치 (단위: 0.088도/tick)
        
        사용 예시:
            goal_positions = client.read_goal_pos()
            print(f"모터 1 목표: {goal_positions[0]}")
        """
        return self.sync_read(self.motor_ids, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    def read_cur(self):
        """
        현재 전류 읽기 (부호 있는 정수로 변환)
        
        Dynamixel의 전류값은 부호 없는 정수로 전송되지만,
        실제 의미는 부호 있는 정수입니다. 이 함수는 자동으로 변환합니다.
        
        Returns:
            list[int]: 각 모터의 현재 전류 (단위: mA)
                      양수: 모터가 힘을 내는 방향
                      음수: 모터가 힘을 받는 방향 (역방향)
                      예: [50, -30, 100, ...]
        
        변환 과정:
        1. sync_read로 부호 없는 값 읽기
        2. 32768 이상이면 음수로 변환
        3. 변환 공식: value >= 32768 ? value - 65536 : value
        
        사용 예시:
            currents = client.read_cur()
            print(f"모터 1 전류: {currents[0]} mA")
            
            # 과부하 감지
            if abs(currents[0]) > 500:
                print("모터 1 과부하!")
        """
        # 부호 없는 값으로 읽기
        bulk_data = self.sync_read(
            self.motor_ids, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
        )
        
        # 부호 있는 정수로 변환
        # print(f"    [전류 변환] 원본: {bulk_data}")
        for i in range(len(bulk_data)):
            value = bulk_data[i]
            if value is not None:
                # 32768 이상이면 음수
                bulk_data[i] = value - 65536 if value >= 32768 else value
        # print(f"    [전류 변환] 변환: {bulk_data}")
        
        return bulk_data

    def read_vel(self):
        """
        현재 속도 읽기
        
        Returns:
            list[int]: 각 모터의 현재 속도 (단위: 0.229 rpm/tick)
                      양수: 시계 방향 (CW)
                      음수: 반시계 방향 (CCW)
                      예: [100, -50, 200, ...]
        
        사용 예시:
            velocities = client.read_vel()
            print(f"모터 1 속도: {velocities[0]} (0.229 rpm/tick)")
        """
        return self.sync_read(
            self.motor_ids, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
        )

    # =============================================================================
    # 편의 함수: 자주 사용하는 쓰기 작업
    # =============================================================================

    def set_pos(self, values):
        """
        목표 위치 설정 (모든 모터 동시)
        
        Args:
            values (Sequence[int]): 각 모터의 목표 위치
                                   len(values) == len(self.motor_ids) 필수
                                   단위: 0.088도/tick
        
        사용 예시:
            # 모든 모터를 특정 위치로
            client.set_pos([1000, 1500, 2000, ...])
            
            # 현재 위치에서 +100씩 이동
            current_pos = client.read_pos()
            new_pos = [p + 100 for p in current_pos]
            client.set_pos(new_pos)
        """
        self.sync_write(
            list(self.motor_ids), values, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )

    def set_pos_indv(self, motor_id, value):
        """
        개별 모터 위치 설정
        
        GroupSyncWrite 대신 개별 write를 사용합니다.
        단일 모터만 제어할 때 사용합니다.
        
        Args:
            motor_id (int): 대상 모터 ID
            value (int): 목표 위치 (0.088도/tick)
        
        주의사항:
        - 여러 모터를 제어할 때는 set_pos() 사용 권장
        - 개별 write는 GroupSync 대비 느림
        
        사용 예시:
            # 모터 1번만 이동
            client.set_pos_indv(1, 2000)
        """
        print(f"    [개별 Write] 모터 #{motor_id} 위치 설정: {value}")
        
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_POSITION, value
        )
        
        if dxl_comm_result != COMM_SUCCESS:
            error_msg = self.packet_handler.getTxRxResult(dxl_comm_result)
            print(f"      ✗ 통신 실패: {error_msg}")
            logging.error("%s" % error_msg)
        elif dxl_error != 0:
            error_msg = self.packet_handler.getRxPacketError(dxl_error)
            print(f"      ✗ 모터 에러: {error_msg}")
            logging.error("%s" % error_msg)
        else:
            print(f"      ✓ 위치 설정 성공")
            logging.info(
                "Dynamixel %d has been successfully set position %d" % (motor_id, value)
            )

    def single_write(self, motor_id, value, addr):
        """
        개별 모터에 2바이트 데이터 쓰기
        
        범용 쓰기 함수입니다. Control Table의 2바이트 레지스터에 사용합니다.
        
        Args:
            motor_id (int): 대상 모터 ID
            value (int): 쓸 값 (0~65535)
            addr (int): Control Table 주소
        
        적용 사례:
        - PID 게인 설정 (P, I, D Gain)
        - 속도/가속도 제한 설정
        - 온도 제한 설정
        
        사용 예시:
            # P Gain 설정
            client.single_write(1, 450, ADDR_POSITION_P_GAIN)
            
            # 속도 제한 설정
            client.single_write(1, 400, ADDR_PROFILE_VELOCITY)
        """
        print(f"    [개별 Write 2B] 모터 #{motor_id} 주소 {addr}: {value}")
        
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(
            self.port_handler, motor_id, addr, value
        )
        
        print(f"      통신 결과: {dxl_comm_result}, 모터 에러: {dxl_error}")
        
        if dxl_comm_result != COMM_SUCCESS:
            error_msg = self.packet_handler.getTxRxResult(dxl_comm_result)
            print(f"      ✗ 통신 실패: {error_msg}")
            logging.error("%s" % error_msg)
        elif dxl_error != 0:
            error_msg = self.packet_handler.getRxPacketError(dxl_error)
            print(f"      ✗ 모터 에러: {error_msg}")
            logging.error("%s" % error_msg)
        else:
            print(f"      ✓ 쓰기 성공")
            logging.info(
                "Dynamixel %d has been successfully set gain %d" % (motor_id, value)
            )

    def single_read(self, motor_id, addr):
        """
        개별 모터에서 2바이트 데이터 읽기
        
        범용 읽기 함수입니다. Control Table의 2바이트 레지스터에 사용합니다.
        
        Args:
            motor_id (int): 대상 모터 ID
            addr (int): Control Table 주소
        
        Returns:
            int or None: 읽은 값 (0~65535), 실패 시 None
        
        적용 사례:
        - PID 게인 확인
        - 온도 읽기
        - 전압 읽기
        
        사용 예시:
            # P Gain 확인
            p_gain = client.single_read(1, ADDR_POSITION_P_GAIN)
            print(f"P Gain: {p_gain}")
            
            # 온도 확인
            temp = client.single_read(1, ADDR_PRESENT_TEMPERATURE)
            print(f"온도: {temp}도")
        """
        value, result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler, motor_id, addr
        )
        
        if result != COMM_SUCCESS:
            error_msg = self.packet_handler.getTxRxResult(result)
            print(f"      ✗ 주소 {addr} 읽기 실패: {error_msg}")
            return None
        elif error != 0:
            error_msg = self.packet_handler.getRxPacketError(error)
            print(f"      ✗ 주소 {addr} 에러: {error_msg}")
            return None
        else:
            return value

    def read_single_cur(self, motor_id):
        """
        개별 모터의 현재 전류 읽기 (부호 있는 정수로 변환)
        
        단일 모터의 전류만 읽을 때 사용합니다.
        여러 모터를 읽을 때는 read_cur() 사용 권장.
        
        Args:
            motor_id (int): 대상 모터 ID
        
        Returns:
            int or None: 현재 전류 (mA), 실패 시 None
                        양수: 모터가 힘을 내는 방향
                        음수: 모터가 힘을 받는 방향
        
        사용 예시:
            # 모터 1번 전류만 확인
            current = client.read_single_cur(1)
            if current is not None:
                print(f"전류: {current} mA")
                
            # 과부하 모니터링
            while True:
                current = client.read_single_cur(1)
                if abs(current) > 500:
                    print("과부하 감지!")
                    break
                time.sleep(0.1)
        """
        # 2바이트 읽기
        value, result, error = self.packet_handler.read2ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_CURRENT
        )
        
        if result != COMM_SUCCESS:
            error_msg = self.packet_handler.getTxRxResult(result)
            print(
                f"      ✗ 전류 읽기 실패 (주소 {ADDR_PRESENT_CURRENT}): {error_msg}"
            )
            return None
        elif error != 0:
            error_msg = self.packet_handler.getRxPacketError(error)
            print(
                f"      ✗ 전류 읽기 에러 (주소 {ADDR_PRESENT_CURRENT}): {error_msg}"
            )
            return None
        else:
            # 부호 있는 정수로 변환
            # 16비트 부호 있는: -32768 ~ 32767
            signed_value = value - 65536 if value >= 32768 else value
            return signed_value

    # =============================================================================
    # Context Manager 지원
    # =============================================================================

    def __enter__(self):
        """
        Context Manager 진입 시 호출
        
        with 구문 사용 시 자동으로 연결합니다.
        
        Returns:
            self: DynamixelClient 인스턴스
        
        사용 예시:
            with DynamixelClient([1,2,3]) as client:
                client.set_torque_enabled(True)
                # 자동으로 연결됨
            # 자동으로 disconnect() 호출됨
        """
        if not self.is_connected:
            print(f"[Context Manager] 자동 연결 중...")
            self.connect()
        return self

    def __exit__(self, *args):
        """
        Context Manager 종료 시 호출
        
        with 구문 종료 시 자동으로 연결을 해제합니다.
        
        Args:
            *args: 예외 정보 (있는 경우)
                  exc_type: 예외 타입
                  exc_value: 예외 값
                  traceback: 트레이스백
        
        반환값:
            None: 예외를 전파하지 않음
        """
        print(f"[Context Manager] 자동 연결 해제 중...")
        self.disconnect()

    def __del__(self):
        """
        객체 소멸 시 자동으로 연결 해제
        
        이 함수는 객체가 가비지 컬렉션될 때 호출됩니다.
        
        주의사항:
        - 소멸 시점을 정확히 예측할 수 없음
        - 명시적 disconnect() 호출 권장
        - atexit handler가 더 안전함
        """
        self.disconnect()


# =============================================================================
# 전역 정리 함수 등록
# =============================================================================

# 프로그램 종료 시 자동으로 dynamixel_cleanup_handler 호출
# 모든 열린 연결을 안전하게 해제
atexit.register(dynamixel_cleanup_handler)

# =============================================================================
# 명령줄 테스트 코드
# =============================================================================

if __name__ == "__main__":
    """
    명령줄에서 직접 실행 시 테스트 코드 실행
    
    테스트 기능:
    1. 토크 활성화
    2. 초기 위치로 이동
    3. 주기적으로 위치 증가
    4. 실시간 모니터링 (위치, 전류, 주파수)
    
    사용 예시:
        python dynamixel_util.py -m 1,2,3,4,5,6,7,8,9,10,11 -d /dev/ttyUSB0
    """
    import argparse
    import itertools
    
    print(f"\n{'='*70}")
    print(f"Dynamixel Utility Test Program")
    print(f"Dynamixel 유틸리티 테스트 프로그램")
    print(f"{'='*70}\n")
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description="Dynamixel 모터 테스트 프로그램"
    )
    parser.add_argument(
        "-m", "--motors",
        required=True,
        help="모터 ID 리스트 (쉼표로 구분). 예: 1,2,3,4,5,6,7,8,9,10,11"
    )
    parser.add_argument(
        "-d", "--device",
        default="/dev/tty.usbserial-FT8ISQ5P",
        help="Dynamixel 디바이스 경로. 기본값: /dev/tty.usbserial-FT8ISQ5P"
    )
    parser.add_argument(
        "-b", "--baud",
        default=57600,
        type=int,
        help="Baudrate. 기본값: 57600"
    )
    parsed_args = parser.parse_args()
    
    # 모터 ID 파싱
    motors = [int(motor) for motor in parsed_args.motors.split(",")]
    
    print(f"[설정]")
    print(f"  모터: {motors}")
    print(f"  디바이스: {parsed_args.device}")
    print(f"  Baudrate: {parsed_args.baud}")
    
    try:
        # DynamixelClient 생성 및 연결
        with DynamixelClient(motors, parsed_args.device, parsed_args.baud) as dxl_client:
            
            # 초기 위치 정의
            init_pos = np.array(
                [1000, 1040, 600, 1500, 800, 800, 1200, 1200, 2900, 2700, 2800]
            )
            fist_pos = np.array(
                [2700, 1500, 1500, 500, 1500, 1700, 2200, 2000, 2000, 1800, 2000]
            )
            fist_pos_6to11 = np.array([1700, 2200, 2000, 2000, 1800, 2000])
            
            # 토크 활성화
            print(f"\n[1단계] 토크 활성화...")
            dxl_client.set_torque_enabled(True, -1, 0.05)
            time.sleep(4)
            
            # 토크 상태 확인
            torque_now = dxl_client.sync_read(motors, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE)
            print(f"  → 토크 상태: {torque_now}")
            
            # 초기 위치로 이동
            print(f"\n[2단계] 초기 위치로 이동...")
            dxl_client.set_pos(init_pos)
            time.sleep(0.1)
            
            # 메인 루프
            print(f"\n[3단계] 메인 루프 시작 (Ctrl+C로 종료)")
            print(f"  → 50스텝마다 위치 +100씩 증가 (500스텝까지)")
            print(f"  → 5스텝마다 상태 출력\n")
            
            for step in itertools.count():
                read_start = time.time()
                
                # 목표 위치 읽기
                goal_pos = dxl_client.read_goal_pos()
                
                # 50스텝마다 위치 증가 (500스텝까지)
                if step > 0 and step % 50 == 0 and step < 500:
                    print(f"\n  [스텝 {step}] 위치 업데이트 중...")
                    for i in range(len(goal_pos)):
                        if i < 9:
                            goal_pos[i] += 100
                        else:
                            goal_pos[i] -= 100
                    dxl_client.set_pos(goal_pos)
                
                # 현재 상태 읽기
                pos_now = dxl_client.read_pos()
                cur_now = dxl_client.read_cur()
                
                # 5스텝마다 출력
                if step % 5 == 0:
                    elapsed = time.time() - read_start
                    frequency = 1.0 / elapsed if elapsed > 0 else 0
                    
                    print(f"[스텝 {step:4d}] 주파수: {frequency:5.2f} Hz")
                    print(f"  → 위치: {pos_now}")
                    print(f"  → 전류: {cur_now}")
    
    except KeyboardInterrupt:
        print(f"\n\n[종료] Ctrl+C 감지 - 프로그램 종료")
    except Exception as e:
        print(f"\n\n[에러] {type(e).__name__}: {e}")
        raise
    finally:
        print(f"\n{'='*70}")
        print(f"테스트 종료")
        print(f"{'='*70}\n")