#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
RUKA Motor Calibration Data Viewer
모터 캘리브레이션 데이터(.npy) 확인 스크립트
"""

import os
import sys
import numpy as np
from pathlib import Path

def check_calibration_file(filepath):
    """
    .npy 캘리브레이션 파일 내용 확인

    Args:
    filepath: .npy 파일 경로
    """
    if not os.path.exists(filepath):
        print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
        return

    try:
        # 파일 로드
        data = np.load(filepath)
        
        # 파일 정보
        print("\n" + "="*70)
        print(f"파일: {filepath}")
        print("="*70)
        
        # 기본 정보
        print(f"\n[기본 정보]")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        print(f"  Size: {data.size} elements")
        print(f"  파일 크기: {os.path.getsize(filepath)} bytes")
        
        # 통계 정보
        print(f"\n[통계 정보]")
        print(f"  최솟값: {data.min()}")
        print(f"  최댓값: {data.max()}")
        print(f"  평균값: {data.mean():.2f}")
        print(f"  표준편차: {data.std():.2f}")
        print(f"  중앙값: {np.median(data):.2f}")
        
        # 모터별 상세 정보
        print(f"\n[모터별 값]")
        motor_names = [
            "모터  1 (엄지 IP)",
            "모터  2 (엄지 MCP)", 
            "모터  3 (검지 DIP)",
            "모터  4 (검지 MCP)",
            "모터  5 (검지 PIP)",
            "모터  6 (중지 DIP)",
            "모터  7 (중지 MCP)",
            "모터  8 (약지 DIP)",
            "모터  9 (약지 MCP)",
            "모터 10 (소지 DIP)",
            "모터 11 (소지 MCP)"
        ]
        
        for i, value in enumerate(data):
            if i < len(motor_names):
                print(f"  {motor_names[i]}: {value:4d}")
            else:
                print(f"  모터 {i+1:2d}: {value:4d}")
        
        # 원본 배열 출력
        print(f"\n[원본 배열]")
        print(f"  {data}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")

def check_all_calibrations(base_dir="motor_limits"):
    """
    motor_limits 디렉토리의 모든 캘리브레이션 파일 확인

    Args:
        base_dir: 캘리브레이션 파일이 저장된 디렉토리
    """
    if not os.path.exists(base_dir):
        print(f"❌ 디렉토리를 찾을 수 없습니다: {base_dir}")
        return

    # .npy 파일 찾기
    npy_files = list(Path(base_dir).glob("*.npy"))

    if not npy_files:
        print(f"❌ {base_dir}에 .npy 파일이 없습니다.")
        return

    print(f"\n발견된 캘리브레이션 파일: {len(npy_files)}개\n")

    # 파일 정렬 (right -> left, curl -> tension 순서)
    npy_files = sorted(npy_files, key=lambda x: (
        "left" in x.name,  # right가 먼저
        "tension" in x.name  # curl이 먼저
    ))

    for filepath in npy_files:
        check_calibration_file(str(filepath))


def compare_calibrations(curl_file, tension_file):
    """
    curl과 tension 캘리브레이션 비교

    Args:
        curl_file: curl limits 파일 경로
        tension_file: tension limits 파일 경로
    """
    if not os.path.exists(curl_file):
        print(f"❌ Curl 파일을 찾을 수 없습니다: {curl_file}")
        return

    if not os.path.exists(tension_file):
        print(f"❌ Tension 파일을 찾을 수 없습니다: {tension_file}")
        return

    try:
        curl = np.load(curl_file)
        tension = np.load(tension_file)
        
        print("\n" + "="*70)
        print("Curl vs Tension 비교")
        print("="*70)
        
        motor_names = [
            "엄지 IP", "엄지 MCP", "검지 DIP", "검지 MCP", "검지 PIP",
            "중지 DIP", "중지 MCP", "약지 DIP", "약지 MCP", "소지 DIP", "소지 MCP"
        ]
        
        print(f"\n{'모터':<12} {'Curl':>6} {'Tension':>8} {'차이':>6} {'Range':>6}")
        print("-" * 70)
        
        for i in range(len(curl)):
            diff = abs(curl[i] - tension[i])
            motor_range = abs(curl[i] - tension[i])
            name = motor_names[i] if i < len(motor_names) else f"모터 {i+1}"
            print(f"{name:<12} {curl[i]:6d} {tension[i]:8d} {diff:6d} {motor_range:6d}")
        
        print("-" * 70)
        print(f"{'평균':<12} {curl.mean():6.1f} {tension.mean():8.1f} "
            f"{np.abs(curl - tension).mean():6.1f}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"❌ 비교 실패: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RUKA 모터 캘리브레이션 데이터 확인"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="확인할 .npy 파일 경로"
    )

    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="motor_limits 디렉토리의 모든 파일 확인"
    )

    parser.add_argument(
        "-c", "--compare",
        type=str,
        nargs=2,
        metavar=("CURL_FILE", "TENSION_FILE"),
        help="curl과 tension 파일 비교"
    )

    parser.add_argument(
        "-ht", "--hand-type",
        type=str,
        choices=["right", "left"],
        help="특정 손의 curl/tension 비교 (--all과 함께 사용)"
    )

    args = parser.parse_args()

    # 단일 파일 확인
    if args.file:
        check_calibration_file(args.file)

    # 두 파일 비교
    elif args.compare:
        compare_calibrations(args.compare[0], args.compare[1])

    # 특정 손 비교
    elif args.hand_type:
        curl_file = f"motor_limits/{args.hand_type}_curl_limits.npy"
        tension_file = f"motor_limits/{args.hand_type}_tension_limits.npy"
        compare_calibrations(curl_file, tension_file)

    # 전체 확인
    elif args.all:
        check_all_calibrations()

    # 기본: 전체 확인
    else:
        check_all_calibrations()