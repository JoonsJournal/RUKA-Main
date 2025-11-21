#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Dynamixel 통신 진단 스크립트

calibrate_motors.py 실행 전에 통신 상태를 점검하는 스크립트입니다.

사용법:
python diagnose_communication.py –hand-type right
python diagnose_communication.py -ht left
"""

import argparse
import time
import logging
from ruka_hand.control.hand import Hand

# 로깅 설정

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)

def diagnose_communication(hand, motor_ids, test_count=50):
	"""
	통신 상태 진단

	Args:
		hand: Hand 객체
		motor_ids: 테스트할 모터 ID 리스트
		test_count: 각 테스트 항목별 반복 횟수

	Returns:
		dict: 진단 결과
	"""
	results = {
		'total_attempts': 0,
		'successful_attempts': 0,
		'errors': [],
		'timeout_count': 0,
		'latency_ms': []
	}

	print("\n" + "="*70)
	print("Dynamixel 통신 진단")
	print("="*70)
	print(f"\n테스트 설정:")
	print(f"  - 모터 개수: {len(motor_ids)}")
	print(f"  - 테스트 횟수: {test_count}")
	print(f"  - 총 시도: {test_count * len(motor_ids)}")

	# 1. 기본 연결 테스트
	print(f"\n[테스트 1/4] 기본 연결 확인")
	try:
		pos = hand.read_pos()
		if pos is not None and len(pos) > 0:
			print(f"  ✓ 초기 연결 성공")
			print(f"  현재 위치: {pos[:5]}... (처음 5개 모터)")
		else:
			print(f"  ✗ 초기 연결 실패")
			return results
	except Exception as e:
		print(f"  ✗ 초기 연결 실패: {e}")
		return results

	# 2. 연속 읽기 테스트
	print(f"\n[테스트 2/4] 연속 읽기 테스트 ({test_count}회)")
	for i in range(test_count):
		results['total_attempts'] += 1
		start_time = time.time()
		
		try:
			pos = hand.read_pos()
			if pos is not None and len(pos) >= len(motor_ids):
				results['successful_attempts'] += 1
				latency = (time.time() - start_time) * 1000  # ms
				results['latency_ms'].append(latency)
			else:
				results['errors'].append(("read_pos", "Invalid response"))
				
		except Exception as e:
			error_str = str(e).lower()
			if 'timeout' in error_str:
				results['timeout_count'] += 1
			results['errors'].append(("read_pos", str(e)))
		
		if (i+1) % 10 == 0:
			print(f"  진행: {i+1}/{test_count}")

	success_rate = (results['successful_attempts'] / results['total_attempts'] * 100)
	print(f"  성공률: {success_rate:.2f}%")

	# 3. 개별 모터 전류 읽기 테스트
	print(f"\n[테스트 3/4] 개별 모터 전류 읽기 (각 모터 10회)")
	current_errors = 0

	for motor_id in motor_ids:
		for i in range(10):
			try:
				current = hand.read_single_cur(motor_id)
				if current is None:
					current_errors += 1
			except Exception as e:
				current_errors += 1
				results['errors'].append((f"motor_{motor_id}", str(e)))
		
		print(f"  모터 {motor_id:2d}: {'✓' if current_errors == 0 else f'✗ ({current_errors} 오류)'}")
		current_errors = 0

	# 4. 지연시간 분석
	print(f"\n[테스트 4/4] 통신 지연시간 분석")
	if results['latency_ms']:
		import numpy as np
		latency = np.array(results['latency_ms'])
		print(f"  - 평균: {latency.mean():.2f} ms")
		print(f"  - 최소: {latency.min():.2f} ms")
		print(f"  - 최대: {latency.max():.2f} ms")
		print(f"  - 표준편차: {latency.std():.2f} ms")
		
		if latency.mean() > 50:
			print(f"  ⚠️ 지연시간이 높습니다 (>50ms)")
		else:
			print(f"  ✓ 지연시간 정상")

	# 최종 결과
	print(f"\n" + "="*70)
	print(f"진단 결과 요약")
	print(f"="*70)
	print(f"\n통신 통계:")
	print(f"  - 총 시도: {results['total_attempts']}")
	print(f"  - 성공: {results['successful_attempts']}")
	print(f"  - 실패: {len(results['errors'])}")
	print(f"  - 타임아웃: {results['timeout_count']}")
	print(f"  - 성공률: {success_rate:.2f}%")

	# 판정
	print(f"\n최종 판정:")
	if success_rate >= 98:
		print(f"  ✓ 통신 상태 매우 양호")
		print(f"  calibrate_motors.py 실행 가능")
	elif success_rate >= 90:
		print(f"  ⚠ 통신 상태 양호 (주의 필요)")
		print(f"  calibrate_motors.py 실행 가능하나 간헐적 오류 발생 가능")
	elif success_rate >= 70:
		print(f"  ⚠️ 통신 불안정")
		print(f"  하드웨어 점검 후 진행 권장")
	else:
		print(f"  ✗ 통신 매우 불안정")
		print(f"  반드시 하드웨어 점검 필요")

	# 권장 조치
	if success_rate < 95:
		print(f"\n권장 조치:")
		print(f"  1. USB 케이블 연결 확인")
		print(f"  2. 전원 공급 상태 확인")
		print(f"  3. Windows USB 전원 관리 비활성화")
		print(f"  4. BAUDRATE를 57600으로 낮추기 (dynamixel_util.py)")
		print(f"  5. U2D2 커넥터 청소 및 재연결")

	if results['timeout_count'] > results['total_attempts'] * 0.1:
		print(f"\n타임아웃이 많이 발생했습니다:")
		print(f"  - Packet Timeout 설정 증가 권장")
		print(f"  - BAUDRATE 낮추기 권장")

	print(f"\n" + "="*70)

	return results


def main():
	"""메인 함수"""
	parser = argparse.ArgumentParser(
		description="Dynamixel 통신 진단 스크립트"
	)
	parser.add_argument(
		"-ht", "–hand-type",
		type=str,
		default="right",
		choices=["right", "left"],
		help="로봇 손 종류 (기본값: right)"
	)
	parser.add_argument(
		"–test-count",
		type=int,
		default=50,
		help="각 테스트 반복 횟수 (기본값: 50)"
	)

	args = parser.parse_args()

	try:
		print("\nHand 객체 초기화 중...")
		hand = Hand(hand_type=args.hand_type)
		print("✓ Hand 초기화 완료\n")
		
		motor_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
		
		results = diagnose_communication(hand, motor_ids, args.test_count)
		
		# Hand 종료
		hand.close()
		print("\n프로그램을 종료합니다.")
		
	except KeyboardInterrupt:
		print("\n\n사용자가 프로그램을 중단했습니다.")
		try:
			hand.close()
		except:
			pass
		
	except Exception as e:
		print(f"\n오류 발생: {e}")
		print(f"\n트러블슈팅:")
		print(f"  1. USB 연결 확인")
		print(f"  2. 로봇 손 전원 확인")
		print(f"  3. 시리얼 포트 권한 확인")
		raise
	

if __name__ == "__main__":
	main()