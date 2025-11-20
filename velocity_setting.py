# RUKA 캘리브레이션 속도 조절 가이드

## 개요

`calibrate_motors.py`에서 curl 측정 시 모터가 원점으로 돌아가는 속도를 조절하는 방법입니다.

-----

## 해결 방법 1: **init** 함수에 추가 (권장)

### 위치

`calibrate_motors.py` 파일의 `HandCalibrator.__init__()` 함수 끝 부분 (약 377번 라인)

### 추가할 코드

```python
def __init__(
    self,
    data_save_dir,
    hand_type,
    curr_lim=50,
    testing=False,
    motor_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    # ... (기존 코드) ...
    
    print(f"\n  저장 디렉토리: {data_save_dir}")
    print(f"  Curl 파일: {os.path.basename(self.curled_path)}")
    print(f"  Tension 파일: {os.path.basename(self.tension_path)}")
    print(f"\n[초기화 완료]")
    
    # ==================== 여기에 추가 ====================
    # 캘리브레이션용 모터 속도 설정
    print(f"\n[캘리브레이션 속도 설정]")
    
    # Import 필요 (파일 상단에 추가)
    from ruka_hand.utils.control_table import (
        ADDR_PROFILE_VELOCITY,
        LEN_PROFILE_VELOCITY,
        ADDR_PROFILE_ACCELERATION,
        LEN_PROFILE_ACCELERATION
    )
    
    # 속도 설정 (값이 작을수록 느림)
    calib_velocity = 150      # 기본값: 400 (권장: 50-200)
    calib_acceleration = 80   # 기본값: 200 (권장: 30-100)
    
    # 모든 모터에 적용
    self.hand.dxl_client.sync_write(
        self.motor_ids,
        [calib_velocity] * len(self.motor_ids),
        ADDR_PROFILE_VELOCITY,
        LEN_PROFILE_VELOCITY
    )
    
    self.hand.dxl_client.sync_write(
        self.motor_ids,
        [calib_acceleration] * len(self.motor_ids),
        ADDR_PROFILE_ACCELERATION,
        LEN_PROFILE_ACCELERATION
    )
    
    print(f"  ✓ Profile Velocity: {calib_velocity}")
    print(f"  ✓ Profile Acceleration: {calib_acceleration}")
    print(f"  ※ 캘리브레이션용으로 느리게 설정됨")
    # ====================================================
```

### 파일 상단에 Import 추가

`calibrate_motors.py` 파일 상단 (약 30-40번 라인 근처)에 추가:

```python
# RUKA 프로젝트 모듈 임포트
from ruka_hand.control.hand import *            # Hand 클래스: 로봇 손 제어
from ruka_hand.utils.file_ops import get_repo_root  # 프로젝트 루트 경로

# ========== 추가: Control Table 상수 임포트 ==========
from ruka_hand.utils.control_table import (
    ADDR_PROFILE_VELOCITY,
    LEN_PROFILE_VELOCITY,
    ADDR_PROFILE_ACCELERATION,
    LEN_PROFILE_ACCELERATION
)
# ===================================================
```

-----

## 해결 방법 2: 개별 모터별로 설정

특정 모터만 느리게 하고 싶다면 `find_bound()` 함수 시작 부분에 추가:

```python
def find_bound(self, motor_id):
    """
    이진 탐색으로 단일 모터의 최대 구부림 위치(Curl Limit) 찾기
    """
    # ========== 해당 모터만 속도 설정 ==========
    # 예: 검지 손가락 모터(4, 5번)만 더 느리게
    if motor_id in [4, 5]:
        motor_velocity = 100  # 검지는 더 느리게
    else:
        motor_velocity = 150  # 나머지는 보통
    
    self.hand.dxl_client.sync_write(
        [motor_id],
        [motor_velocity],
        ADDR_PROFILE_VELOCITY,
        LEN_PROFILE_VELOCITY
    )
    # ========================================
    
    # 특수 모터의 경우 전류 제한값과 대기 시간 조정
    t = 2  # 기본 대기 시간 2초
    if motor_id in [4, 5]:
        # ... (기존 코드)
```

-----

## 속도 파라미터 설명

### Profile Velocity (프로필 속도)

- **Control Table 주소**: 112
- **데이터 크기**: 4 bytes
- **단위**: 0.229 rev/min
- **범위**: 0 ~ 1023
- **0**: 무제한 속도 (매우 빠름, 위험!)
- **값이 작을수록 느림**

#### 권장 값

|용도        |Profile Velocity|설명         |
|----------|----------------|-----------|
|빠른 이동 (기본)|400             |hand.py 기본값|
|보통 속도     |200             |안정적인 캘리브레이션|
|느린 이동 (권장)|150             |정밀한 측정     |
|매우 느림     |50-100          |디버깅/안전     |

### Profile Acceleration (프로필 가속도)

- **Control Table 주소**: 108
- **데이터 크기**: 4 bytes
- **단위**: 214.577 rev/min²
- **범위**: 0 ~ 32767
- **0**: 무한 가속 (매우 급격함)
- **값이 작을수록 부드러운 가속**

#### 권장 값

|용도          |Profile Acceleration|설명         |
|------------|--------------------|-----------|
|빠른 가속 (기본)  |200                 |hand.py 기본값|
|보통 가속       |100                 |안정적인 가속    |
|부드러운 가속 (권장)|80                  |텐던 보호      |
|매우 부드러움     |30-50               |충격 최소화     |

-----

## 속도 조절의 효과

### Before (기본 속도: 400)

```
반복 1: 위치 2050로 이동 중... 완료 (0.5초)
반복 2: 위치 3025로 이동 중... 완료 (0.7초)
반복 3: 위치 3513로 이동 중... 완료 (0.4초)
```

- 빠르지만 관성으로 인한 오버슈트 가능
- 전류 측정값 불안정

### After (느린 속도: 150)

```
반복 1: 위치 2050로 이동 중... 완료 (1.2초)
반복 2: 위치 3025로 이동 중... 완료 (1.5초)
반복 3: 위치 3513로 이동 중... 완료 (1.0초)
```

- 느리지만 정확한 위치 도달
- 전류 측정값 안정적
- 텐던 손상 위험 감소

-----

## 주의사항

### 1. 너무 느린 속도의 문제

```python
calib_velocity = 10  # 너무 느림!
```

- 캘리브레이션 시간이 너무 길어짐 (30분 이상)
- 모터가 정지 상태로 인식될 수 있음

### 2. 속도 0 설정 금지

```python
calib_velocity = 0  # 절대 금지!
```

- 0 = 무제한 속도
- 매우 빠르게 이동하여 하드웨어 손상 위험

### 3. 가속도 0 설정 금지

```python
calib_acceleration = 0  # 절대 금지!
```

- 0 = 무한 가속
- 급격한 힘으로 텐던이 끊어질 수 있음

-----

## 테스트 방법

### 1. 단일 모터 테스트

```bash
python calibrate_motors.py --hand-type right --mode curl
```

- 모터가 천천히 움직이는지 확인
- 전류 측정값이 안정적인지 확인

### 2. 속도 조정

속도가 너무 느리면:

```python
calib_velocity = 200      # 150 → 200으로 증가
calib_acceleration = 100  # 80 → 100으로 증가
```

속도가 너무 빠르면:

```python
calib_velocity = 100      # 150 → 100으로 감소
calib_acceleration = 50   # 80 → 50으로 감소
```

### 3. 정상 동작 확인

```bash
# 캘리브레이션 후 테스트
python scripts/reset_motors.py --hand-type right
```

-----

## FAQ

### Q1: 속도를 변경해도 효과가 없어요

**A**: Import를 확인하세요. 파일 상단에 다음이 있어야 합니다:

```python
from ruka_hand.utils.control_table import (
    ADDR_PROFILE_VELOCITY,
    LEN_PROFILE_VELOCITY,
    ADDR_PROFILE_ACCELERATION,
    LEN_PROFILE_ACCELERATION
)
```

### Q2: 일부 모터만 느리게 할 수 있나요?

**A**: 네, `find_bound()` 함수 내에서 motor_id를 확인하여 개별 설정 가능합니다.

### Q3: 이 설정이 일반 동작에도 영향을 주나요?

**A**: 아니오. 캘리브레이션 시에만 적용되며, `hand.py`나 `reset_motors.py`는 자체 설정을 사용합니다.

### Q4: 권장 속도 값은?

**A**:

- **정확도 우선**: velocity=100, acceleration=50
- **균형**: velocity=150, acceleration=80 (권장)
- **속도 우선**: velocity=250, acceleration=150

-----

## 추가 개선 사항

### 명령줄 인자로 속도 조절

```python
parser.add_argument(
    "--calib-velocity",
    type=int,
    default=150,
    help="Calibration profile velocity (50-400)"
)

parser.add_argument(
    "--calib-acceleration",
    type=int,
    default=80,
    help="Calibration profile acceleration (30-200)"
)

# 사용
args = parse_args()
calib_velocity = args.calib_velocity
calib_acceleration = args.calib_acceleration
```

### 사용 예시

```bash
# 느린 속도로 캘리브레이션
python calibrate_motors.py --hand-type right --calib-velocity 100 --calib-acceleration 50

# 빠른 속도로 캘리브레이션
python calibrate_motors.py --hand-type right --calib-velocity 250 --calib-acceleration 150
```

-----

## 요약

**가장 간단한 해결 방법:**

1. `calibrate_motors.py` 파일 열기
1. 상단 import 부분에 Control Table 상수 추가
1. `HandCalibrator.__init__()` 함수 끝에 속도 설정 코드 추가
1. 권장 값: `velocity=150`, `acceleration=80`
1. 테스트 실행

이렇게 하면 캘리브레이션 중 모터가 천천히 안전하게 움직이면서 정확한 측정이 가능합니다!