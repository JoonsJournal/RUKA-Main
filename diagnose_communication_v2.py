#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Dynamixel 통신 종합 진단 스크립트

calibrate_motors.py 실행 전에 통신 상태, 전원, 하드웨어를 종합적으로 점검하는 스크립트입니다.

진단 항목:

1. 기본 통신 연결 테스트
1. 연속 읽기 안정성 테스트
1. 개별 모터 전류 읽기 테스트
1. 통신 지연시간 분석
1. BAUDRATE 최적화 테스트 (선택)
1. 전원 공급 전압 모니터링 (선택)
1. 모터 온도 확인
1. 하드웨어 에러 상태 점검
1. 통신 품질 종합 분석

사용법:
python diagnose_communication.py –hand-type right
python diagnose_communication.py -ht left –full-test
python diagnose_communication.py –baudrate-test-only
python diagnose_communication.py –voltage-monitor
"""

import argparse
import time
import logging
import sys
import os

# NumPy import

try:
	import numpy as np
except ImportError:
	print("오류: numpy를 찾을 수 없습니다.")
	print("설치: pip install numpy")
	sys.exit(1)

# Dynamixel SDK import

try:
	from dynamixel_sdk import *
except ImportError:
	print("오류: dynamixel_sdk를 찾을 수 없습니다.")
	print("설치: pip install dynamixel-sdk")
	sys.exit(1)

# RUKA 모듈 import

try:
	from ruka_hand.control.hand import Hand
	from ruka_hand.utils.constants import USB_PORTS
	from ruka_hand.utils.control_table.control_table import (
		ADDR_PRESENT_VOLTAGE,
		LEN_PRESENT_VOLTAGE,
		ADDR_PRESENT_TEMP,
		LEN_PRESENT_TEMP,
		ADDR_HARDWARE_ERROR,
		LEN_HARDWARE_ERROR,
		ADDR_PRESENT_CURRENT,
		LEN_PRESENT_CURRENT,
		ADDR_PRESENT_POSITION,
		LEN_PRESENT_POSITION,
	)
except ImportError as e:
	print(f"오류: RUKA 모듈을 찾을 수 없습니다: {e}")
	print("PYTHONPATH를 확인하거나 프로젝트 루트에서 실행하세요.")
	sys.exit(1)

# 로깅 설정

logging.basicConfig(
	level=logging.WARNING,
	format='%(asctime)s - %(levelname)s - %(message)s'
)

# 통신 상수

COMM_SUCCESS = 0
PROTOCOL_VERSION = 2.0

# 테스트할 BAUDRATE 목록

BAUDRATE_OPTIONS = [
	(57600, "매우 안정적 (권장: 캘리브레이션)"),
	(115200, "안정적 (권장: 일반 제어)"),
	(1000000, "빠름 (고품질 케이블 필요)"),
	(2000000, "매우 빠름 (짧은 고품질 케이블 필수)"),
]

def test_baudrate(port, baudrate, test_count=50):
	"""
	특정 BAUDRATE에서 통신 품질 테스트
	
	Args:
		port: 시리얼 포트 경로
		baudrate: 테스트할 BAUDRATE
		test_count: 테스트 반복 횟수

		Returns:
		dict: 테스트 결과
	"""
	port_handler = PortHandler(port)
	packet_handler = PacketHandler(PROTOCOL_VERSION)

	results = {
	'baudrate': baudrate,
	'success_count': 0,
	'error_count': 0,
	'timeout_count': 0,
	'latency_ms': [],
	'success_rate': 0.0
	}
	try:
		# 포트 열기
		if not port_handler.openPort():
			results['error_count'] = test_count
			return results

		# BAUDRATE 설정
		if not port_handler.setBaudRate(baudrate):
			results['error_count'] = test_count
			port_handler.closePort()
			return results

		# Packet timeout 설정
		port_handler.setPacketTimeout(100)  # 100ms

		# 연속 ping 테스트
		for i in range(test_count):
			start_time = time.time()
			
			try:
				model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, 1)
				
				latency = (time.time() - start_time) * 1000  # ms
				
				if dxl_comm_result == COMM_SUCCESS:
					results['success_count'] += 1
					results['latency_ms'].append(latency)
				else:
					results['error_count'] += 1
					result_str = packet_handler.getTxRxResult(dxl_comm_result).lower()
					if 'timeout' in result_str:
						results['timeout_count'] += 1
						
			except Exception as e:
				results['error_count'] += 1

		# 성공률 계산
		results['success_rate'] = (results['success_count'] / test_count) * 100
		
	except Exception as e:
		logging.error(f"BAUDRATE {baudrate} 테스트 중 오류: {e}")
		results['error_count'] = test_count

	finally:
		try:
			port_handler.closePort()
		except:
			pass

	return results

def diagnose_baudrate(port):
	"""
	여러 BAUDRATE에서 통신 품질을 테스트하고 최적값 추천
	Args:
		port: 시리얼 포트 경로
	Returns:
		tuple: (all_results, recommended_baudrate)
	"""
	print("\n" + "="*70)
	print("BAUDRATE 최적화 테스트")
	print("="*70)
	print("\n이 테스트는 약 30초 소요됩니다...")
	print("각 BAUDRATE에서 50회씩 통신 테스트를 진행합니다.\n")
	
	all_results = []
	best_baudrate = None
	best_score = 0
	
	for baudrate, description in BAUDRATE_OPTIONS:
		print(f"[테스트] {baudrate} bps - {description}")
		print(f"  진행 중...", end="", flush=True)
		
		results = test_baudrate(port, baudrate, test_count=50)
		all_results.append(results)
		
		# 점수 계산 (성공률 70% + 지연시간 30%)
		score = results['success_rate'] * 0.7
		if results['latency_ms']:
			avg_latency = np.mean(results['latency_ms'])
			# 지연시간 점수: 10ms 이하=30점, 50ms 이상=0점
			latency_score = max(0, 30 - (avg_latency - 10) * 0.75)
			score += latency_score
		
		if score > best_score:
			best_score = score
			best_baudrate = baudrate
		
		# 결과 출력
		print(f"\r  성공률: {results['success_rate']:.1f}%", end="")
		
		if results['latency_ms']:
			latency = np.array(results['latency_ms'])
			print(f" | 평균 지연: {latency.mean():.2f}ms", end="")
		
		if results['timeout_count'] > 0:
			print(f" | 타임아웃: {results['timeout_count']}회", end="")
		
		# 판정
		if results['success_rate'] >= 98:
			print(" | ✓ 매우 좋음")
		elif results['success_rate'] >= 95:
			print(" | ✓ 좋음")
		elif results['success_rate'] >= 85:
			print(" | ⚠ 보통")
		elif results['success_rate'] >= 70:
			print(" | ⚠ 주의")
		else:
			print(" | ✗ 불량")
		
		time.sleep(0.5)  # 테스트 간 짧은 대기

	# 최종 추천
	print("\n" + "="*70)
	print("BAUDRATE 추천")
	print("="*70)

	# 성공률 95% 이상인 것 중 가장 빠른 것 추천
	recommended = None
	for results in reversed(all_results):  # 빠른 것부터
		if results['success_rate'] >= 95:
			recommended = results['baudrate']
			break

	if recommended:
		print(f"\n✓ 권장 BAUDRATE: {recommended} bps")
		print(f"  - 성공률 95% 이상을 만족하는 가장 빠른 속도입니다.")
	else:
		# 95% 미만이면 성공률이 가장 높은 것 추천
		best_result = max(all_results, key=lambda x: x['success_rate'])
		print(f"\n⚠ 권장 BAUDRATE: {best_result['baudrate']} bps")
		print(f"  - 테스트 결과 중 가장 안정적입니다.")
		print(f"  - 주의: 성공률이 {best_result['success_rate']:.1f}%로 낮습니다.")
		print(f"  - 하드웨어 점검을 권장합니다.")

	# 용도별 권장사항
	print(f"\n용도별 권장 BAUDRATE:")
	print(f"  - 캘리브레이션: 57600 (안정성 최우선)")
	print(f"  - 일반 제어: 115200 ~ 1000000 (균형)")
	print(f"  - 고속 제어: 1000000 ~ 2000000 (속도 우선, 환경 좋을 때)")

	return all_results, recommended

def check_voltage(hand, motor_ids, duration=10):
	"""
	전원 공급 전압 모니터링
	Args:
		hand: Hand 객체
		motor_ids: 모니터링할 모터 ID 리스트
		duration: 모니터링 시간 (초)

	Returns:
		dict: 전압 데이터
	"""
	print("\n" + "="*70)
	print("전원 공급 전압 모니터링")
	print("="*70)
	print(f"\n{duration}초 동안 전압을 모니터링합니다...")
	print("모터를 동작시키면서 전압 강하를 확인합니다.\n")

	voltage_data = {motor_id: [] for motor_id in motor_ids[:3]}  # 처음 3개 모터만

	start_time = time.time()

	print("시간(s) | ", end="")
	for motor_id in voltage_data.keys():
		print(f"모터 {motor_id:2d} | ", end="")
	print("상태")
	print("-" * 70)

	try:
		while time.time() - start_time < duration:
			elapsed = time.time() - start_time
			
			print(f"{elapsed:6.1f}  | ", end="")
			
			for motor_id in voltage_data.keys():
				try:
					voltage_raw, result, error = hand.dxl_client.packet_handler.read2ByteTxRx(
						hand.dxl_client.port_handler,
						motor_id,
						ADDR_PRESENT_VOLTAGE
					)
					
					if result == COMM_SUCCESS:
						voltage_v = voltage_raw / 10.0  # 0.1V 단위
						voltage_data[motor_id].append(voltage_v)
						print(f"{voltage_v:5.2f}V | ", end="")
					else:
						print("  ERR  | ", end="")
						
				except Exception as e:
					print("  ERR  | ", end="")
			
			# 전압 상태 판정
			if voltage_data[list(voltage_data.keys())[0]]:
				current_voltage = voltage_data[list(voltage_data.keys())[0]][-1]
				if current_voltage >= 11.5:
					print("✓ 정상")
				elif current_voltage >= 11.0:
					print("⚠ 주의")
				else:
					print("✗ 낮음")
			else:
				print()
			
			time.sleep(1.0)  # 1초 간격
			
	except KeyboardInterrupt:
		print("\n\n모니터링 중단됨 (Ctrl+C)")

	# 통계 계산
	print("\n" + "="*70)
	print("전압 통계")
	print("="*70)

	for motor_id, voltages in voltage_data.items():
		if not voltages:
			continue
			
		voltages = np.array(voltages)
		
		print(f"\n모터 {motor_id:2d}:")
		print(f"  - 평균 전압: {voltages.mean():.3f}V")
		print(f"  - 최소 전압: {voltages.min():.3f}V")
		print(f"  - 최대 전압: {voltages.max():.3f}V")
		print(f"  - 전압 변동: {voltages.std():.3f}V")
		print(f"  - 샘플 수: {len(voltages)}")
		
		# 판정
		avg_voltage = voltages.mean()
		voltage_drop = voltages.max() - voltages.min()
		
		print(f"\n  판정:")
		if avg_voltage >= 11.5:
			print(f"    ✓ 평균 전압: 정상")
		elif avg_voltage >= 11.0:
			print(f"    ⚠ 평균 전압: 약간 낮음")
		else:
			print(f"    ✗ 평균 전압: 매우 낮음 - 전원 공급 장치 점검 필요")
		
		if voltage_drop < 0.3:
			print(f"    ✓ 전압 변동: 안정적 ({voltage_drop:.3f}V)")
		elif voltage_drop < 0.5:
			print(f"    ⚠ 전압 변동: 약간 큼 ({voltage_drop:.3f}V)")
		else:
			print(f"    ✗ 전압 변동: 매우 큼 ({voltage_drop:.3f}V) - 전원 공급 불안정")

	# 전체 권장사항
	print("\n" + "="*70)
	print("권장사항:")
	print("  - XL-330: 권장 전압 5V")
	print("  - XM430: 권장 전압 11.1V ~ 12V")
	print("  - 전압 변동 < 0.5V: 안정적")
	print("  - 전원 공급 장치 용량: 최소 5A 이상 권장")

	return voltage_data


def check_temperature(hand, motor_ids):
	"""
	모터 온도 확인

	Args:
		hand: Hand 객체
		motor_ids: 확인할 모터 ID 리스트

	Returns:
		dict: 모터별 온도
	"""
	print("\n" + "="*70)
	print("모터 온도 확인")
	print("="*70)
	print()

	temperatures = {}

	for motor_id in motor_ids:
		try:
			temp_raw, result, error = hand.dxl_client.packet_handler.read1ByteTxRx(
				hand.dxl_client.port_handler,
				motor_id,
				ADDR_PRESENT_TEMP
			)
			
			if result == COMM_SUCCESS:
				temperatures[motor_id] = temp_raw
				status = ""
				if temp_raw < 50:
					status = "✓ 정상"
				elif temp_raw < 60:
					status = "⚠ 주의"
				else:
					status = "✗ 높음"
				
				print(f"  모터 {motor_id:2d}: {temp_raw}°C  {status}")
			else:
				print(f"  모터 {motor_id:2d}: 읽기 실패")
				
		except Exception as e:
			print(f"  모터 {motor_id:2d}: 오류 - {e}")

	# 통계
	if temperatures:
		temps = np.array(list(temperatures.values()))
		
		print(f"\n온도 통계:")
		print(f"  - 평균: {temps.mean():.1f}°C")
		print(f"  - 최소: {temps.min()}°C")
		print(f"  - 최대: {temps.max()}°C")
		
		if temps.max() >= 60:
			print(f"\n  ⚠️ 경고: 일부 모터 온도가 높습니다!")
			print(f"  - 과열 위험이 있습니다.")
			print(f"  - 모터를 쉬게 하거나 냉각을 개선하세요.")

	return temperatures

def check_hardware_errors(hand, motor_ids):
	"""
	하드웨어 에러 상태 확인

	Args:
		hand: Hand 객체
		motor_ids: 확인할 모터 ID 리스트

	Returns:
		dict: 모터별 에러 상태
	"""
	print("\n" + "="*70)
	print("하드웨어 에러 상태 확인")
	print("="*70)
	print()

	# 에러 비트 의미
	ERROR_BITS = {
		0x01: "Input Voltage Error",
		0x04: "Overheating Error",
		0x08: "Motor Encoder Error",
		0x10: "Electrical Shock Error",
		0x20: "Overload Error",
	}

	errors = {}
	has_error = False

	for motor_id in motor_ids:
		try:
			error_raw, result, error = hand.dxl_client.packet_handler.read1ByteTxRx(
				hand.dxl_client.port_handler,
				motor_id,
				ADDR_HARDWARE_ERROR
			)
			
			if result == COMM_SUCCESS:
				errors[motor_id] = error_raw
				
				if error_raw == 0:
					print(f"  모터 {motor_id:2d}: ✓ 정상 (에러 없음)")
				else:
					has_error = True
					print(f"  모터 {motor_id:2d}: ✗ 에러 발생 (0x{error_raw:02X})")
					
					# 에러 상세 내용
					for bit, description in ERROR_BITS.items():
						if error_raw & bit:
							print(f"            - {description}")
			else:
				print(f"  모터 {motor_id:2d}: 읽기 실패")
				
		except Exception as e:
			print(f"  모터 {motor_id:2d}: 오류 - {e}")

	if has_error:
		print(f"\n  ⚠️ 경고: 일부 모터에 하드웨어 에러가 있습니다!")
		print(f"  - 전원을 껐다 켜서 에러를 리셋하세요.")
		print(f"  - 문제가 지속되면 해당 모터를 점검하세요.")
	else:
		print(f"\n  ✓ 모든 모터 정상")

	return errors

def diagnose_communication(hand, motor_ids, test_count=50):
	"""
	통신 상태 진단
	Args:
		hand: Hand 객체
		motor_ids: 테스트할 모터 ID 리스트
		test_count: 테스트 반복 횟수
	Returns:
		dict: 진단 결과
	"""
	print("\n" + "="*70)
	print("기본 통신 테스트")
	print("="*70)
	print(f"\n테스트 설정:")
	print(f"  - 모터 개수: {len(motor_ids)}")
	print(f"  - 테스트 횟수: {test_count}")

	results = {
		'total_attempts': 0,
		'successful_attempts': 0,
		'errors': [],
		'timeout_count': 0,
		'latency_ms': []
	}

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
		
		if current_errors == 0:
			print(f"  모터 {motor_id:2d}: ✓")
		else:
			print(f"  모터 {motor_id:2d}: ✗ ({current_errors} 오류)")
		current_errors = 0

	# 4. 지연시간 분석
	print(f"\n[테스트 4/4] 통신 지연시간 분석")
	if results['latency_ms']:
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
	print(f"통신 진단 결과")
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

	return results

def main():
	"""메인 함수"""
	parser = argparse.ArgumentParser(
		description="Dynamixel 통신 종합 진단 스크립트",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
		진단 모드:
		기본 모드: 기본 통신, 온도, 에러 확인
		–full-test: 모든 진단 항목 실행
		–baudrate-test-only: BAUDRATE 테스트만 실행
		–voltage-monitor: 전압 모니터링만 실행

		사용 예시:
		python diagnose_communication.py –hand-type right
		python diagnose_communication.py -ht left –full-test
		python diagnose_communication.py –baudrate-test-only
		python diagnose_communication.py -ht right –voltage-monitor –duration 20
		"""
	)

	parser.add_argument(
		"-ht", "--hand-type",
		type=str,
		default="right",
		choices=["right", "left"],
		help="로봇 손 종류 (기본값: right)"
	)
	parser.add_argument(
		"--test-count",
		type=int,
		default=50,
		help="기본 통신 테스트 반복 횟수 (기본값: 50)"
	)
	parser.add_argument(
		"--full-test",
		action="store_true",
		help="모든 진단 항목 실행 (기본, BAUDRATE, 전압)"
	)
	parser.add_argument(
		"--baudrate-test-only",
		action="store_true",
		help="BAUDRATE 테스트만 실행"
	)
	parser.add_argument(
		"--voltage-monitor",
		action="store_true",
		help="전압 모니터링만 실행"
	)
	parser.add_argument(
		"--duration",
		type=int,
		default=10,
		help="전압 모니터링 시간 (초, 기본값: 10)"
	)

	args = parser.parse_args()

	print("\n" + "="*70)
	print("Dynamixel 통신 종합 진단")
	print("="*70)
	print(f"\nCopyright (c) NYU RUKA Team")
	print(f"License: MIT License\n")

	motor_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

	# BAUDRATE 테스트만 실행
	if args.baudrate_test_only:
		try:
			port = USB_PORTS.get(args.hand_type, "COM3")
			diagnose_baudrate(port)
		except Exception as e:
			print(f"\n오류 발생: {e}")
		return

	# Hand 객체 초기화
	try:
		print("\nHand 객체 초기화 중...")
		hand = Hand(hand_type=args.hand_type)
		print("✓ Hand 초기화 완료\n")
		
		# 전압 모니터링만 실행
		if args.voltage_monitor:
			check_voltage(hand, motor_ids, duration=args.duration)
			hand.close()
			return
		
		# 기본 진단
		results = diagnose_communication(hand, motor_ids, args.test_count)
		
		# 온도 확인
		check_temperature(hand, motor_ids)
		
		# 하드웨어 에러 확인
		check_hardware_errors(hand, motor_ids)
		
		# 전체 테스트 모드
		if args.full_test:
			# BAUDRATE 테스트
			port = USB_PORTS.get(args.hand_type, "COM3")
			hand.close()  # 기존 연결 종료
			time.sleep(1)
			diagnose_baudrate(port)
			
			# 다시 연결
			time.sleep(1)
			hand = Hand(hand_type=args.hand_type)
			
			# 전압 모니터링
			check_voltage(hand, motor_ids, duration=args.duration)
		
		# Hand 종료
		hand.close()
		
		print("\n" + "="*70)
		print("진단 완료")
		print("="*70)
		print("\n다음 단계:")
		print("  1. 문제가 발견되면 권장 조치 수행")
		print("  2. calibrate_motors.py 실행")
		print(f"\n명령: python calibrate_motors.py --hand-type {args.hand_type}")
		
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
		import traceback
		traceback.print_exc()

if __name__ == "__main__":
	main()