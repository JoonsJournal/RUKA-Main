#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HandController read_pos() 캐싱 최적화 패치

이 파일은 controller.py의 step() 메서드에 적용할 최적화 코드입니다.
기존 코드에서 read_pos()가 여러 번 호출되는 것을 1회로 줄입니다.

적용 방법:
1. controller.py를 열기
2. step() 메서드를 아래 코드로 교체
3. _process_input_cached() 메서드 추가

예상 효과:
- 제어 주기당 Dynamixel 통신 3회 → 1회
- 약 20-40ms 절감
"""

# =============================================================================
# controller.py의 step() 메서드 교체 코드
# =============================================================================

def step(self, input_data, moving_average_info=None, move=True):
    """
    최적화된 step 메서드 - read_pos() 캐싱 적용
    
    변경점:
    - read_pos()를 한 번만 호출하고 캐시
    - _process_input_cached()로 캐시된 위치 사용
    - move_to_pos()에서도 캐시 재사용
    """
    # input_data: (5,3) - 5: fingers, 3: input_dim
    input_data = torch.FloatTensor(input_data)
    
    # 🆕 read_pos()를 한 번만 호출하고 캐시
    current_pos = self.hand.read_pos()
    
    for finger_name in self.learners.keys():
        learner = self.learners[finger_name]
        
        finger_id = FINGER_NAMES_TO_MANUS_IDS[finger_name]
        motor_ids = FINGER_NAMES_TO_MOTOR_IDS[finger_name]
        
        model_input = input_data[finger_id, :]  # (3)
        
        # 🆕 캐시된 위치를 사용하는 버전
        model_input = self._process_input_cached(
            input=model_input, 
            finger_name=finger_name,
            cached_pos=current_pos  # 캐시 전달
        )
        
        pred_motor_pos = learner.forward(model_input).detach().cpu()[0]
        
        robot_stats = torch.stack(
            [self.robot_stats[0][motor_ids], self.robot_stats[1][motor_ids]]
        )
        pred_motor_pos = handle_normalization(
            input=pred_motor_pos, stats=robot_stats, normalize=False, mean_std=False
        )
        
        self._process_output(
            output=pred_motor_pos, finger_name=finger_name, weighted_average=False
        )
    
    if not moving_average_info is None:
        self.hand_pos = moving_average(
            self.hand_pos,
            moving_average_info["queue"],
            moving_average_info["limit"],
        )
    
    if move:
        # 🆕 캐시된 위치 재사용 (read_pos() 다시 호출 안함!)
        self.move_to_pos(
            curr_pos=current_pos,  # 캐시 사용
            des_pos=self.hand_pos,
            traj_len=self.single_move_len,
        )
    else:
        return self.hand_pos


def _process_input_cached(self, input, finger_name, cached_pos):
    """
    캐시된 모터 위치를 사용하는 _process_input 버전
    
    Parameters:
    -----------
    input : torch.Tensor
        입력 데이터
    finger_name : str
        손가락 이름
    cached_pos : list or np.ndarray
        캐시된 모터 위치 (11개)
    
    Returns:
    --------
    torch.Tensor
        처리된 입력 데이터
    """
    cfg = self.cfgs[finger_name]
    
    if "state_as_input" in cfg.dataset and cfg.dataset.state_as_input:
        motor_ids = FINGER_NAMES_TO_MOTOR_IDS[finger_name]
        
        # 🆕 캐시된 위치 사용 (read_pos() 호출 안함!)
        curr_motor_pos = torch.FloatTensor(cached_pos)[motor_ids]
        
        input = handle_normalization(
            input=input,
            stats=self.finger_to_stats[finger_name]["input"],
            normalize=True,
            mean_std=(
                self.cfgs[finger_name].dataset.fingertip_mean_std_norm
                if "fingertip_mean_std_norm" in self.cfgs[finger_name].dataset
                else False
            ),
        )
        
        motor_norm = handle_normalization(
            input=curr_motor_pos,
            stats=self.finger_to_stats[finger_name]["motor"],
            normalize=True,
            mean_std=False,
        )
        
        input = torch.cat([input, motor_norm], dim=-1)
    
    else:
        input = handle_normalization(
            input=input,
            stats=self.finger_to_stats[finger_name]["input"],
            normalize=True,
            mean_std=(
                self.cfgs[finger_name].dataset.fingertip_mean_std_norm
                if "fingertip_mean_std_norm" in self.cfgs[finger_name].dataset
                else False
            ),
        )
    
    if "obs_horizon" in cfg.dataset:
        if not finger_name in self.past_observations:
            self.past_observations[finger_name] = input.repeat(
                cfg.dataset.obs_horizon
            ).reshape(-1, input.shape[0])
        else:
            self.past_observations[finger_name] = torch.cat(
                [
                    torch.roll(
                        self.past_observations[finger_name], shifts=-1, dims=0
                    )[:-1, :],
                    input.unsqueeze(0),
                ],
                dim=0,
            )
        input = self.past_observations[finger_name]
    
    return input


# =============================================================================
# _process_output도 캐시 사용하도록 수정 (선택적)
# =============================================================================

def _process_output_cached(self, output, finger_name, cached_pos, weighted_average=False):
    """
    캐시된 모터 위치를 사용하는 _process_output 버전
    
    Parameters:
    -----------
    output : torch.Tensor
        모델 출력
    finger_name : str
        손가락 이름
    cached_pos : list or np.ndarray
        캐시된 모터 위치 (11개)
    weighted_average : bool
        가중 평균 사용 여부
    """
    cfg = self.cfgs[finger_name]
    
    if "pred_horizon" in cfg.dataset:
        if weighted_average:
            pass  # TODO: 가중 평균 구현
        else:
            output = output[0, :]
    
    motor_ids = FINGER_NAMES_TO_MOTOR_IDS[finger_name]
    
    if "predict_residual" in cfg.dataset and cfg.dataset.predict_residual:
        # 🆕 캐시된 위치 사용
        curr_motor_pos = np.array(cached_pos)[motor_ids]
        output = curr_motor_pos + output
    
    output = np.clip(output, 0, 4000)
    
    for i in range(len(motor_ids)):
        self.hand_pos[motor_ids[i]] = output[i]


# =============================================================================
# 적용 방법
# =============================================================================

"""
1. controller.py 백업:
   cp controller.py controller.py.backup

2. controller.py 열기

3. 기존 step() 메서드를 위의 최적화된 버전으로 교체

4. _process_input_cached() 메서드 추가

5. (선택) _process_output_cached() 메서드 추가

6. 테스트:
   python teleop.py -m webcam -ht right

예상 결과:
- Dynamixel 통신 횟수: 3회/주기 → 1회/주기
- 시간 절감: 20-40ms/주기
- FPS 향상: 10-15 Hz → 20-30 Hz
"""