#!/usr/bin/env python3

# -*- coding: utf-8 -*-

“””
RUKA Robot Hand Motor Reset - 로봇 손 모터 리셋 프로그램
이 프로그램은 RUKA(로봇 손)의 모든 모터를 초기 위치(텐션 상태)로
안전하게 리셋합니다.

주요 기능:

1. Dynamixel 모터를 안전한 초기 위치로 이동
1. 텐던 장력이 유지된 상태로 손가락을 완전히 펼침
1. 부드러운 궤적 생성으로 안전한 이동
1. Ctrl+C 또는 에러 발생 시 안전한 종료

사용 목적:

- 하드웨어 셋업 후 초기 테스트
- 캘리브레이션 후 정상 동작 확인
- 비정상 상태에서 안전한 위치로 복귀
- 프로그램 시작 전 초기화

작성: NYU RUKA Team
라이선스: MIT License
“””

# =============================================================================

# 라이브러리 임포트

# =============================================================================

import argparse  # 명령줄 인자 파싱 (hand_type 선택)
import time      # 시간 지연 및 타이밍 제어

# RUKA 프로젝트 모듈 임포트

from ruka_hand.control.hand import Hand              # Hand 클래스: 저수준 모터 제어
from ruka_hand.utils.trajectory import move_to_pos   # 궤적 생성 함수: 부드러운 모터 이동

# =============================================================================

# 의존성 파일 설명

# =============================================================================

“””
이 스크립트는 다음 파일들과 의존 관계를 가집니다:

┌─────────────────────────────────────────────────────────────────┐
│                     reset_motors.py (현재 파일)                  │
│                                                                   │
│  역할: RUKA 로봇 손을 안전한 초기 위치로 리셋                       │
└─────────────────────────────────────────────────────────────────┘
│
├─► [1] ruka_hand/control/hand.py
│     │
│     ├─ Hand 클래스 제공
│     ├─ Dynamixel 모터 저수준 제어
│     ├─ 모터 위치 읽기/쓰기
│     ├─ 전류, 속도, 온도 모니터링
│     └─ 모터 제한값 관리
│
├─► [2] ruka_hand/utils/trajectory.py
│     │
│     ├─ move_to_pos() 함수 제공
│     ├─ 선형 궤적 생성 (Interpolation)
│     ├─ 부드러운 모터 이동
│     └─ 급격한 움직임 방지
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
│     ├─ ModBus 프로토콜 통신
│     ├─ GroupSyncWrite/Read
│     └─ 저수준 하드웨어 인터페이스
│
├─► ruka_hand/utils/file_ops.py
│     ├─ get_repo_root(): 프로젝트 루트 경로
│     └─ 캘리브레이션 파일 경로 찾기
│
└─► motor_limits/*.npy (캘리브레이션 데이터)
├─ {hand_type}_curl_limits.npy
│   └─ 손가락이 완전히 구부러진 위치
└─ {hand_type}_tension_limits.npy
└─ 텐던 장력이 걸린 펼친 위치
“””

# =============================================================================

# 명령줄 인자 파싱 설정

# =============================================================================

“””
ArgumentParser 설정
사용자가 명령줄에서 로봇 손의 종류(왼손/오른손)를 선택할 수 있도록 합니다.

사용 예:
python reset_motors.py –hand_type right   # 오른손 리셋
python reset_motors.py –hand_type left    # 왼손 리셋
python reset_motors.py -ht right           # 축약형
python reset_motors.py                     # 기본값: left

인자 설명:
-ht, –hand_type: 로봇 손 종류 선택
- “left”: 왼손 (기본값)
- “right”: 오른손
“””

parser = argparse.ArgumentParser(
description=“RUKA Robot Hand Motor Reset - 로봇 손 모터를 안전한 초기 위치로 리셋합니다.”,
epilog=”””
사용 예:
python reset_motors.py –hand_type right    # 오른손 리셋
python reset_motors.py -ht left             # 왼손 리셋 (축약형)

주의사항:
- 로봇 손이 USB로 연결되어 있어야 합니다
- 캘리브레이션이 완료되어 있어야 합니다
- 충분한 작업 공간을 확보하세요
- Ctrl+C로 언제든지 안전하게 종료할 수 있습니다
“””,
formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
“-ht”,
“–hand_type”,
type=str,
help=“리셋할 로봇 손 종류 (‘left’ 또는 ‘right’)”,
default=“left”,
choices=[“left”, “right”],  # 허용된 값 제한
metavar=“HAND”  # 도움말에서 표시될 이름
)

# 명령줄 인자 파싱 실행

args = parser.parse_args()

# =============================================================================

# Hand 객체 초기화

# =============================================================================

“””
Hand 클래스 인스턴스 생성

ruka_hand.control.hand.Hand 클래스의 인스턴스를 생성하여
Dynamixel 모터와의 저수준 통신을 설정합니다.

Hand 클래스 초기화 과정 (hand.py 내부):

1. 모터 ID 설정
- self.motors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- 각 모터는 고유 ID로 식별됨
1. USB 포트 연결
- constants.py에서 USB_PORTS[hand_type] 로드
- 예: USB_PORTS = {“left”: “/dev/ttyUSB0”, “right”: “/dev/ttyUSB1”}
- DynamixelClient 초기화 (시리얼 통신)
1. 모터 설정
   a) Operating Mode 설정
- Mode 5: Current-based Position Control
- 위치 제어 + 전류 제한 (과부하 방지)
   
   b) PID Gains 설정
- MCP 관절 (손가락 뿌리):
  P_GAIN = 450, I_GAIN = 120, D_GAIN = 1000
- DIP/PIP 관절 (손가락 중간/끝):
  P_GAIN = 500, I_GAIN = 100, D_GAIN = 960
   
   c) 제한값 설정
- Current Limit: 700 (과전류 방지)
- Temperature Limit: 60°C (과열 방지)
- Goal Velocity: 400 (속도 제한)
1. 캘리브레이션 데이터 로드
- 경로: motor_limits/{hand_type}_curl_limits.npy
  motor_limits/{hand_type}_tension_limits.npy
1. Torque Enable
- 모든 모터의 토크를 활성화
- 이제 모터가 위치 명령을 받을 준비 완료
  “””

print(”\n” + “=”*70)
print(“RUKA Robot Hand Motor Reset”)
print(“로봇 손 모터 리셋 프로그램”)
print(”=”*70)
print(f”\n[설정]”)
print(f”  손 종류: {args.hand_type.upper()}”)
print(f”  목표 위치: 텐션 상태 (완전히 펼친 상태)”)
print(f”  궤적 길이: 50 포인트”)
print(f”  예상 소요 시간: 약 0.5초/회”)
print(f”\n[초기화 중…]”)

try:
# Hand 클래스 인스턴스 생성
hand = Hand(args.hand_type)
print(f”  ✓ Hand 클래스 초기화 완료”)
print(f”  ✓ Dynamixel 모터 연결 완료 (11개 모터)”)
print(f”  ✓ 캘리브레이션 데이터 로드 완료”)
print(f”  ✓ 모터 토크 활성화 완료”)

except Exception as e:
print(f”\n✗ 초기화 실패!”)
print(f”  에러: {e}”)
print(f”\n가능한 원인:”)
print(f”  1. USB 케이블이 연결되지 않음”)
print(f”  2. 로봇 손 전원이 꺼져 있음”)
print(f”  3. 시리얼 포트 권한 부족 (sudo usermod -aG dialout $USER)”)
print(f”  4. 캘리브레이션 미완료 (python calibrate_motors.py 실행)”)
print(f”  5. 다른 프로그램에서 포트 사용 중”)
exit(-1)

# =============================================================================

# 메인 리셋 루프

# =============================================================================

“””
무한 루프로 모터 리셋 반복 실행

이 루프는 다음 작업을 반복합니다:

1. 현재 모터 위치 읽기
1. 목표 위치(텐션 상태) 설정
1. 부드러운 궤적으로 이동
1. 위치 확인 및 출력
1. 0.5초 대기 후 반복

루프 종료 조건:

- Ctrl+C 입력 (KeyboardInterrupt)
- 예외 발생 (통신 오류, 하드웨어 문제 등)
- try-except 블록에서 모든 예외 캐치

안전 기능:

- 급격한 모터 움직임 방지 (궤적 생성)
- 모터 제한값 자동 적용 (전류, 온도, 속도)
- 예외 발생 시 안전하게 종료
- 최종적으로 hand.close() 호출 보장
  “””

print(f”\n[리셋 시작]”)
print(f”  Ctrl+C를 눌러 언제든지 종료할 수 있습니다.”)
print(f”  손가락이 부드럽게 펼쳐지는 것을 확인하세요.”)
print(”=”*70 + “\n”)

# 루프 카운터 초기화

loop_count = 0

# 무한 루프 시작

while True:
# =========================================================================
# 단계 1: 현재 모터 위치 읽기
# =========================================================================
“””
hand.read_pos() 상세 동작 (hand.py):

```
현재 모터 위치 읽기

Returns:
    list of int: 11개 모터의 현재 위치
    예: [2000, 1500, 1000, ...]

내부 동작:
    1. DynamixelClient.read_pos() 호출
    2. GroupSyncRead 실행
       - 주소: ADDR_PRESENT_POSITION (132)
       - 크기: LEN_PRESENT_POSITION (4 bytes)
       - 대상: 11개 모터
    3. ModBus 프로토콜로 데이터 수신
    4. 각 모터의 응답 파싱
    5. 리스트로 반환

통신 에러 처리:
    - 타임아웃 발생 시 재시도
    - None 값 수신 시 루프 반복
    - 최대 3회 재시도 후 실패
"""
curr_pos = hand.read_pos()

# =========================================================================
# 단계 2: 안정화 대기
# =========================================================================
"""
time.sleep(0.5)의 역할:

1. 모터 위치 읽기 후 안정화 시간 확보
2. ModBus 통신 버퍼 비우기
3. 다음 명령 전 준비 시간
4. 사용자가 로봇 동작을 시각적으로 확인할 시간
"""
time.sleep(0.5)

# =========================================================================
# 단계 3: 위치 정보 출력
# =========================================================================
"""
출력 정보:

curr_pos: 현재 모터 위치 (11개 값)
des_pos: 목표 모터 위치 (hand.tensioned_pos)

tensioned_pos 값의 의미:
    캘리브레이션으로 측정된 "텐션 상태" 위치
    손가락이 완전히 펼쳐지되 텐던이 느슨하지 않은 상태
"""
print(f"[루프 #{loop_count + 1}]")
print(f"  현재 위치: {curr_pos}")
print(f"  목표 위치: {hand.tensioned_pos}")

# =========================================================================
# 단계 4: 목표 위치 설정
# =========================================================================
test_pos = hand.tensioned_pos

# =========================================================================
# 단계 5: 궤적 생성 및 이동
# =========================================================================
"""
try-except 블록의 역할:

안전한 종료를 보장하기 위한 예외 처리

발생 가능한 예외:
    1. KeyboardInterrupt: Ctrl+C 입력
    2. ModbusException: 통신 오류
    3. serial.SerialException: 시리얼 포트 오류
    4. RuntimeError: 모터 응답 없음
    5. 기타 예기치 않은 오류
"""
try:
    print(f"  → 이동 중... (50 포인트 궤적)")
    
    # 궤적 생성 및 이동 실행
    move_to_pos(
        curr_pos=curr_pos,      # 시작점
        des_pos=test_pos,       # 목표점 (tensioned_pos)
        hand=hand,              # Hand 객체
        traj_len=50             # 50개 중간점
    )
    
    print(f"  ✓ 이동 완료")
    loop_count += 1

except KeyboardInterrupt:
    # Ctrl+C 입력 시 처리
    print(f"\n\n{'='*70}")
    print(f"[종료 요청]")
    print(f"  Ctrl+C가 감지되었습니다.")
    print(f"  안전하게 종료합니다...")
    print(f"  총 실행 횟수: {loop_count}회")
    print("="*70)
    break  # 루프 탈출

except Exception as e:
    # 기타 예외 처리
    print(f"\n\n{'='*70}")
    print(f"[예외 발생]")
    print(f"  에러 타입: {type(e).__name__}")
    print(f"  에러 메시지: {e}")
    print(f"  총 실행 횟수: {loop_count}회")
    print("="*70)
    break  # 루프 탈출
```

# =============================================================================

# 안전한 종료 처리

# =============================================================================

“””
hand.close() 상세 동작 (hand.py):

프로그램 종료 시 반드시 호출해야 하는 정리 함수

## 단계:

1. 모터 토크 비활성화
   self.dxl_client.set_torque_enabled(False)
   
   이유:
- 프로그램 종료 후에도 모터가 활성화되어 있으면
- 외부 힘에 저항하여 불필요한 전력 소비
- 발열 및 모터 수명 단축
- 안전사고 위험
1. DynamixelClient 연결 해제
   self.dxl_client.disconnect()
   
   내부 동작:
- GroupSyncWrite/Read 객체 정리
- PacketHandler 리소스 해제
- PortHandler 종료
1. 시리얼 포트 닫기
   self.port_handler.closePort()
   
   중요성:
- 포트를 닫지 않으면 다른 프로그램에서 사용 불가
- “Port already in use” 에러 원인
- 운영체제 리소스 누수 방지
  “””

print(f”\n[종료 처리]”)
print(f”  → 모터 토크 비활성화 중…”)

# Hand 객체 정리

hand.close()

print(f”  ✓ 모터 토크 비활성화 완료”)
print(f”  ✓ 시리얼 포트 닫기 완료”)
print(f”  ✓ 리소스 정리 완료”)

print(f”\n{’=’*70}”)
print(f”프로그램이 안전하게 종료되었습니다.”)
print(f”총 {loop_count}회 리셋을 수행했습니다.”)
print(”=”*70 + “\n”)