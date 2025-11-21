#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Tracking Module - MediaPipe 기반 손 추적 모듈

이 모듈은 MediaPipe 라이브러리를 사용하여 실시간 손 추적 및
제스처 인식 기능을 제공합니다.

주요 기능:

1. 손 검출 및 21개 랜드마크 추출
2. 손가락 펴짐/접힘 상태 판단
3. 엄지 회전 각도 계산
4. 손 바운딩 박스 및 중심점 계산
5. 두 점 사이 거리 측정

기술:

- MediaPipe Hands: Google의 손 추적 AI 모델
- OpenCV: 영상 처리 및 시각화

# 작성자: Computer Vision Zone
웹사이트: https://www.computervision.zone/
"""

# =============================================================================
# 라이브러리 임포트
# =============================================================================

import math              # 수학 연산 (각도, 거리 계산)
import cv2               # OpenCV: 영상 처리
import mediapipe as mp   # MediaPipe: 손 추적 AI 모델

# =============================================================================
# 유틸리티 함수
# =============================================================================

def clamp(n, smallest, largest):
    """
    값을 특정 범위로 제한하는 함수
    
    Parameters:
    -----------
    n : float
        제한할 값
    smallest : float
        최소값
    largest : float
        최대값
    
    Returns:
    --------
    float
        제한된 값
    
    예시:
    -----
    clamp(150, 0, 100)  → 100  (100 초과 시 100으로)
    clamp(-50, 0, 100)  → 0    (0 미만 시 0으로)
    clamp(50, 0, 100)   → 50   (범위 내)
    """
    return max(smallest, min(n, largest))


def interpolate(n, from_min, from_max, to_min, to_max):
    """
    값을 한 범위에서 다른 범위로 선형 매핑하는 함수
    
    Parameters:
    -----------
    n : float
        변환할 값
    from_min : float
        입력 범위의 최소값
    from_max : float
        입력 범위의 최대값
    to_min : float
        출력 범위의 최소값
    to_max : float
        출력 범위의 최대값
    
    Returns:
    --------
    float
        변환된 값
    
    수식:
    -----
    (n - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
    
    예시:
    -----
    interpolate(5, 0, 10, 0, 100)  → 50
    # 0~10 범위의 5를 0~100 범위로 변환 → 50
    
    interpolate(97, 97, 125, 0, 1)  → 0
    # 97~125 범위의 97을 0~1 범위로 변환 → 0
    
    interpolate(125, 97, 125, 0, 1)  → 1
    # 97~125 범위의 125를 0~1 범위로 변환 → 1
    """
    return (n - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


# =============================================================================
# 엄지 회전 각도 범위 상수
# =============================================================================
# 엄지 손가락과 검지-중지 사이의 각도 범위
# 이 각도로 엄지 루트의 회전 정도를 판단함

MAXTHUMBDEGREE = 125    # 최대 각도 (엄지가 완전히 벌어짐)
MINTHUMBDEGREE = 97     # 최소 각도 (엄지가 손바닥 쪽으로 붙음)

# =============================================================================
# HandDetector 클래스
# =============================================================================

class HandDetector:
    """
    MediaPipe 기반 손 검출 및 추적 클래스
    
    이 클래스는 카메라 영상에서 손을 검출하고,
    21개의 랜드마크(관절점)를 추출하여 다양한 정보를 제공합니다.
    
    MediaPipe 손 랜드마크 구조 (21개):
    -----------------------------------
    0: 손목 (Wrist)
    
    엄지 (Thumb): 1, 2, 3, 4
      1: 엄지 CMC 관절 (손바닥)
      2: 엄지 MCP 관절
      3: 엄지 IP 관절
      4: 엄지 끝 (Thumb Tip)
    
    검지 (Index): 5, 6, 7, 8
      5: 검지 MCP 관절 (손바닥)
      6: 검지 PIP 관절
      7: 검지 DIP 관절
      8: 검지 끝 (Index Tip)
    
    중지 (Middle): 9, 10, 11, 12
      9: 중지 MCP 관절
      10: 중지 PIP 관절
      11: 중지 DIP 관절
      12: 중지 끝 (Middle Tip)
    
    약지 (Ring): 13, 14, 15, 16
      13: 약지 MCP 관절
      14: 약지 PIP 관절
      15: 약지 DIP 관절
      16: 약지 끝 (Ring Tip)
    
    새끼 (Pinky): 17, 18, 19, 20
      17: 새끼 MCP 관절
      18: 새끼 PIP 관절
      19: 새끼 DIP 관절
      20: 새끼 끝 (Pinky Tip)
    
    주요 메서드:
    -----------
    - findHands(): 이미지에서 손 검출
    - fingersUp(): 각 손가락의 펴짐 상태 판단
    - findDistance(): 두 랜드마크 사이 거리 계산
    - calculate_angle(): 세 점으로 각도 계산
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, 
                 detectionCon=0.5, minTrackCon=0.5):
        """
        HandDetector 생성자
        
        Parameters:
        -----------
        staticMode : bool (기본값: False)
            - False: 비디오 모드 (추적 사용, 빠름)
            - True: 이미지 모드 (매 프레임 검출, 느림)
        
        maxHands : int (기본값: 2)
            - 검출할 최대 손 개수
            - 1: 한 손만 (성능 최적화)
            - 2: 양손
        
        modelComplexity : int (기본값: 1)
            - 0: Lite 모델 (빠르지만 덜 정확)
            - 1: Full 모델 (느리지만 정확)
        
        detectionCon : float (기본값: 0.5)
            - 손 검출 신뢰도 임계값 (0.0 ~ 1.0)
            - 높을수록: 정확하지만 검출 어려움
            - 낮을수록: 민감하지만 오검출 가능
        
        minTrackCon : float (기본값: 0.5)
            - 손 추적 신뢰도 임계값 (0.0 ~ 1.0)
            - 높을수록: 안정적 추적
            - 낮을수록: 불안정하지만 빠른 반응
        
        예시:
        -----
        # 한 손만 높은 정확도로 추적
        detector = HandDetector(maxHands=1, detectionCon=0.8)
        
        # 양손을 빠르게 추적
        detector = HandDetector(maxHands=2, modelComplexity=0)
        """
        
        # 파라미터 저장
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        
        # MediaPipe Hands 모듈 로드
        self.mpHands = mp.solutions.hands
        
        # MediaPipe Hands 객체 생성
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticMode,
            max_num_hands=self.maxHands,
            model_complexity=modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        
        # MediaPipe 그리기 유틸리티
        self.mpDraw = mp.solutions.drawing_utils
        
        # 손가락 끝 랜드마크 인덱스
        # [엄지, 검지, 중지, 약지, 새끼]
        self.tipIds = [4, 8, 12, 16, 20]
        
        # 결과 저장 변수
        self.fingers = []               # 손가락 펴짐 상태
        self.lmList = []                # 랜드마크 리스트
        self.previousThumbDegreeValue = 0  # 이전 엄지 회전 값 (스무딩용)
    
    def findHands(self, img, draw=True, flipType=True):
        """
        이미지에서 손을 검출하고 랜드마크를 추출하는 메서드
        
        이 메서드는 MediaPipe Hands 모델을 사용하여
        입력 이미지에서 손을 찾고 21개의 랜드마크를 추출합니다.
        
        Parameters:
        -----------
        img : numpy.ndarray
            입력 이미지 (BGR 포맷, OpenCV 형식)
        
        draw : bool (기본값: True)
            - True: 랜드마크와 연결선을 이미지에 그림
            - False: 그리지 않음
        
        flipType : bool (기본값: True)
            - True: 좌우 반전 (거울 모드)
            - False: 원본 그대로
        
        Returns:
        --------
        tuple: (allHands, img)
            allHands : list of dict
                검출된 각 손의 정보 리스트
                [
                    {
                        "lmList": [[x, y, z], ...],  # 21개 랜드마크 좌표
                        "bbox": (x, y, w, h),         # 바운딩 박스
                        "center": (cx, cy),           # 중심점
                        "type": "Left" or "Right"     # 손 타입
                    },
                    ...
                ]
            
            img : numpy.ndarray
                랜드마크가 그려진 이미지
        
        처리 과정:
        ----------
        1. BGR → RGB 변환 (MediaPipe는 RGB 사용)
        2. MediaPipe로 손 검출
        3. 랜드마크 픽셀 좌표로 변환
        4. 바운딩 박스 계산
        5. 손 타입 판별 (Left/Right)
        6. 랜드마크 그리기 (옵션)
        
        예시:
        -----
        success, img = cap.read()
        hands, img = detector.findHands(img, draw=True)
        
        if hands:
            hand1 = hands[0]
            lmList = hand1["lmList"]  # 랜드마크 좌표
            bbox = hand1["bbox"]      # (x, y, w, h)
            handType = hand1["type"]  # "Left" or "Right"
        """
        
        # =====================================================================
        # 1단계: BGR → RGB 변환
        # =====================================================================
        # MediaPipe는 RGB 이미지를 사용하므로 변환 필요
        # OpenCV는 기본적으로 BGR 포맷 사용
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # =====================================================================
        # 2단계: MediaPipe로 손 검출
        # =====================================================================
        # process(): 이미지에서 손 검출 및 랜드마크 추출
        self.results = self.hands.process(imgRGB)
        
        # 검출된 모든 손의 정보를 저장할 리스트
        allHands = []
        
        # 이미지 크기 가져오기
        h, w, c = img.shape  # height, width, channels
        
        # =====================================================================
        # 3단계: 검출된 손이 있는지 확인
        # =====================================================================
        if self.results.multi_hand_landmarks:
            # multi_handedness: 손 타입 정보 (Left/Right)
            # multi_hand_landmarks: 손 랜드마크 좌표 (0~1 정규화)
            
            # 각 손에 대해 반복 처리
            for handType, handLms in zip(self.results.multi_handedness, 
                                         self.results.multi_hand_landmarks):
                # 현재 손의 정보를 저장할 딕셔너리
                myHand = {}
                
                # =============================================================
                # 3.1 랜드마크 리스트 생성
                # =============================================================
                mylmList = []   # 랜드마크 좌표 [(x, y, z), ...]
                xList = []      # x 좌표만 모음 (바운딩 박스 계산용)
                yList = []      # y 좌표만 모음
                
                # 21개 랜드마크를 순회
                for id, lm in enumerate(handLms.landmark):
                    # 정규화된 좌표(0~1)를 픽셀 좌표로 변환
                    # lm.x, lm.y: 0~1 범위의 정규화된 좌표
                    # lm.z: 손목 기준 깊이 (작을수록 카메라에 가까움)
                    px = int(lm.x * w)  # x 픽셀 좌표
                    py = int(lm.y * h)  # y 픽셀 좌표
                    pz = int(lm.z * w)  # z 깊이 (w 기준으로 스케일링)
                    
                    # 랜드마크 좌표 추가
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                
                # =============================================================
                # 3.2 바운딩 박스 계산
                # =============================================================
                # 모든 랜드마크를 포함하는 최소 직사각형
                xmin, xmax = min(xList), max(xList)  # x 범위
                ymin, ymax = min(yList), max(yList)  # y 범위
                
                # 박스 크기 계산
                boxW = xmax - xmin  # 너비
                boxH = ymax - ymin  # 높이
                
                # 바운딩 박스: (x, y, w, h)
                bbox = xmin, ymin, boxW, boxH
                
                # =============================================================
                # 3.3 중심점 계산
                # =============================================================
                # 바운딩 박스의 중심 좌표
                cx = bbox[0] + (bbox[2] // 2)  # x + width/2
                cy = bbox[1] + (bbox[3] // 2)  # y + height/2
                
                # =============================================================
                # 3.4 손 정보 딕셔너리에 저장
                # =============================================================
                myHand["lmList"] = mylmList     # 랜드마크 좌표
                myHand["bbox"] = bbox           # 바운딩 박스
                myHand["center"] = (cx, cy)     # 중심점
                
                # =============================================================
                # 3.5 손 타입 판별 (Left/Right)
                # =============================================================
                if flipType:
                    # flipType=True: 거울 모드
                    # MediaPipe가 "Right"라고 판단하면 실제로는 오른손
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    # flipType=False: 원본 그대로
                    myHand["type"] = handType.classification[0].label
                
                # 손 정보를 전체 리스트에 추가
                allHands.append(myHand)
                
                # =============================================================
                # 3.6 랜드마크 그리기 (옵션)
                # =============================================================
                if draw:
                    # 랜드마크와 연결선 그리기
                    # - handLms: 랜드마크 좌표
                    # - HAND_CONNECTIONS: 손가락 뼈대 연결 정보
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS
                    )
                    
                    # 바운딩 박스 그리기
                    # rectangle(이미지, 좌상단, 우하단, 색상, 두께)
                    cv2.rectangle(
                        img,
                        (bbox[0] - 20, bbox[1] - 20),           # 좌상단 (여백 20px)
                        (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),  # 우하단
                        (255, 0, 255),                          # 색상 (보라색)
                        2                                       # 두께
                    )
                    
                    # 손 타입 텍스트 표시
                    # putText(이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)
                    cv2.putText(
                        img,
                        myHand["type"],                         # "Left" or "Right"
                        (bbox[0] - 30, bbox[1] - 30),          # 위치
                        cv2.FONT_HERSHEY_PLAIN,                 # 폰트
                        2,                                      # 크기
                        (255, 0, 255),                          # 색상
                        2                                       # 두께
                    )
        
        # 검출된 모든 손 정보와 이미지 반환
        return allHands, img
    
    def fingersUp(self, myHand):
        """
        각 손가락의 펴짐/접힘 상태를 판단하는 메서드
        
        이 메서드는 손가락 끝과 관절의 위치를 비교하여
        각 손가락이 펴져 있는지 접혀 있는지 판단합니다.
        
        Parameters:
        -----------
        myHand : dict
            findHands()에서 반환한 손 정보 딕셔너리
            {"lmList": [...], "type": "Left" or "Right", ...}
        
        Returns:
        --------
        list
            각 손가락의 상태 [엄지, 검지, 중지, 약지, 새끼, 엄지회전]
            - 0: 접혀 있음
            - 1: 펴져 있음
            - 0~1 (엄지회전): 회전 정도 (0=붙음, 1=벌어짐)
        
        판단 기준:
        ----------
        1. 엄지 (Thumb):
           - 좌/우 손에 따라 판단 방법 다름
           - 왼손: 끝(4번)이 관절(3번)보다 오른쪽 → 펴짐
           - 오른손: 끝(4번)이 관절(3번)보다 왼쪽 → 펴짐
        
        1. 나머지 4개 손가락:
           - 끝(tipIds[i])이 관절(tipIds[i]-2)보다 위쪽 → 펴짐
           - 아래쪽 → 접혀 있음
        
        1. 엄지 회전 (6번째 값):
           - 엄지-검지-중지 각도로 판단
           - 97도 ~ 125도 범위를 0 ~ 1로 매핑
           - 0: 엄지가 손바닥에 붙음
           - 1: 엄지가 완전히 벌어짐
        
        예시:
        -----
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            # fingers = [1, 1, 0, 0, 0, 0.5]
            # → 엄지와 검지는 펴짐, 나머지는 접힘, 엄지 회전 50%
        """
        
        # 결과 저장 리스트
        fingers = []
        
        # 손 타입과 랜드마크 가져오기
        myHandType = myHand["type"]      # "Left" or "Right"
        myLmList = myHand["lmList"]      # 21개 랜드마크 좌표
        
        # 손이 검출된 경우만 처리
        if self.results.multi_hand_landmarks:
            
            # =================================================================
            # 1단계: 엄지 상태 판단
            # =================================================================
            # 엄지는 다른 손가락과 판단 방법이 다름
            # x 좌표로 좌우 위치를 비교
            
            # tipIds[0] = 4 (엄지 끝)
            # tipIds[0] - 1 = 3 (엄지 IP 관절)
            
            if myHandType == "Left":
                # 왼손: 엄지 끝이 관절보다 오른쪽에 있으면 펴짐
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)  # 펴짐
                else:
                    fingers.append(0)  # 접힘
            else:
                # 오른손: 엄지 끝이 관절보다 왼쪽에 있으면 펴짐
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)  # 펴짐
                else:
                    fingers.append(0)  # 접힘
            
            # =================================================================
            # 2단계: 나머지 4개 손가락 상태 판단
            # =================================================================
            # 손가락 끝이 관절보다 위쪽에 있으면 펴짐
            # y 좌표가 작을수록 위쪽 (이미지 좌표계)
            
            # range(1, 5): 검지(1) ~ 새끼(4)
            for id in range(1, 5):
                # tipIds[id]: 손가락 끝 인덱스
                # tipIds[id] - 2: 해당 손가락의 PIP 관절 인덱스
                
                # 끝의 y 좌표 < 관절의 y 좌표 → 펴짐
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)  # 펴짐
                else:
                    fingers.append(0)  # 접힘
            
            # =================================================================
            # 3단계: 엄지 회전 각도 계산
            # =================================================================
            # 엄지 루트의 회전 정도를 0~1 범위로 계산
            
            # 필요한 랜드마크 가져오기
            thumb_base = myLmList[self.tipIds[0] - 2]   # 엄지 MCP (2번)
            index_base = myLmList[self.tipIds[1] - 3]   # 검지 MCP (5번)
            middle_base = myLmList[self.tipIds[2] - 3]  # 중지 MCP (9번)
            
            # 랜드마크가 모두 유효한지 확인
            if thumb_base is not None and index_base is not None and middle_base is not None:
                # 세 점으로 각도 계산
                # 검지를 중심으로 엄지-중지 각도
                degree = int(self.calculate_angle(thumb_base, index_base, middle_base))
                
                # 각도를 0~1 범위로 매핑
                # MINTHUMBDEGREE(97) ~ MAXTHUMBDEGREE(125) → 0 ~ 1
                degree2Interpolation = interpolate(
                    degree,
                    MINTHUMBDEGREE,  # 97
                    MAXTHUMBDEGREE,  # 125
                    0,               # 출력 최소값
                    1                # 출력 최대값
                )
                
                # 0~1 범위로 제한하고 소수점 2자리로 반올림
                degree2Interpolation = round(clamp(degree2Interpolation, 0, 1), 2)
                
                # 스무딩: 극단값만 사용, 중간값은 이전 값 유지
                # 0~0.2: 완전히 붙음 → 0
                if 0 <= degree2Interpolation <= 0.2:
                    degree2Interpolation = 0
                # 0.85~1: 완전히 벌어짐 → 1
                elif 0.85 <= degree2Interpolation <= 1:
                    degree2Interpolation = 1
                # 중간값: 이전 값 유지 (떨림 방지)
                else:
                    degree2Interpolation = self.previousThumbDegreeValue
                
                # 현재 값을 이전 값으로 저장
                self.previousThumbDegreeValue = degree2Interpolation
                
                # 엄지 회전 값 추가
                fingers.append(degree2Interpolation)
            else:
                # 랜드마크가 없으면 현재까지의 결과만 반환
                return fingers
        
        # 최종 결과 반환
        # [엄지, 검지, 중지, 약지, 새끼, 엄지회전]
        return fingers
    
    def calculate_angle(self, thumbVector, indexVector, middleVector) -> float:
        """
        세 점으로 각도를 계산하는 메서드
        
        검지를 중심으로 엄지와 중지가 이루는 각도를 계산합니다.
        이 각도로 엄지 루트의 회전 정도를 판단할 수 있습니다.
        
        Parameters:
        -----------
        thumbVector : list
            엄지 MCP 관절 좌표 [x, y, z]
        indexVector : list
            검지 MCP 관절 좌표 [x, y, z] (중심점)
        middleVector : list
            중지 MCP 관절 좌표 [x, y, z]
        
        Returns:
        --------
        float
            각도 (도 단위, 0~180)
        
        계산 방법:
        ----------
        1. 두 벡터 생성:
           - 검지 → 엄지 벡터
           - 검지 → 중지 벡터
        
        1. 벡터의 내적(dot product) 계산
        
        2. 코사인 법칙으로 각도 계산:
           cos(θ) = (v1 · v2) / (|v1| × |v2|)
           θ = arccos(cos(θ))
        
        3. 라디안을 도(degree)로 변환
        
        기하학적 의미:
        -------------
              엄지(2번)
                 ○
                /
               /
              / θ
             ○ ← 검지(5번, 중심)
              \\
               \\
                ○
              중지(9번)
        
        예시:
        -----
        angle = calculate_angle([100, 200], [150, 200], [200, 200])
        # 세 점이 일직선 → 180도
        
        angle = calculate_angle([150, 150], [150, 200], [200, 200])
        # 직각 → 90도
        """
        
        # =====================================================================
        # 1단계: 벡터 생성
        # =====================================================================
        # 검지를 중심으로 두 벡터 계산
        
        # 검지 → 엄지 벡터
        it_vector = (
            thumbVector[0] - indexVector[0],  # Δx
            thumbVector[1] - indexVector[1]   # Δy
        )
        
        # 검지 → 중지 벡터
        im_vector = (
            middleVector[0] - indexVector[0],  # Δx
            middleVector[1] - indexVector[1]   # Δy
        )
        
        # =====================================================================
        # 2단계: 벡터의 크기(길이) 계산
        # =====================================================================
        # 벡터 크기 = √(x² + y²)
        
        # 검지-엄지 벡터의 길이
        length_it_vector = round(
            math.sqrt(it_vector[0]**2 + it_vector[1]**2),
            0
        )
        
        # 검지-중지 벡터의 길이
        length_im_vector = round(
            math.sqrt(im_vector[0]**2 + im_vector[1]**2),
            0
        )
        
        # =====================================================================
        # 3단계: 내적(Dot Product) 계산
        # =====================================================================
        # 내적 = x1*x2 + y1*y2
        dot_product = it_vector[0] * im_vector[0] + it_vector[1] * im_vector[1]
        
        # =====================================================================
        # 4단계: 코사인 값 계산
        # =====================================================================
        # cos(θ) = 내적 / (벡터1 크기 × 벡터2 크기)
        cos_angle = dot_product / (length_it_vector * length_im_vector)
        
        # =====================================================================
        # 5단계: 아크코사인으로 각도 계산 (라디안)
        # =====================================================================
        try:
            # acos(): 역코사인 함수 (라디안 반환)
            radian_angle = math.acos(cos_angle)
        except Exception as e:
            # 예외 발생 시 (예: 0으로 나누기) 0 반환
            radian_angle = 0
        
        # =====================================================================
        # 6단계: 라디안을 도(degree)로 변환
        # =====================================================================
        # 1 라디안 = 180/π 도
        radian2degree = math.degrees(radian_angle)
        
        # 각도 반환
        return radian2degree
    
    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        두 랜드마크 사이의 거리를 계산하는 메서드
        
        Parameters:
        -----------
        p1 : tuple
            첫 번째 점 (x1, y1)
        p2 : tuple
            두 번째 점 (x2, y2)
        img : numpy.ndarray (옵션)
            거리를 시각화할 이미지
        color : tuple (기본값: (255, 0, 255))
            선 색상 (BGR)
        scale : int (기본값: 5)
            점 크기
        
        Returns:
        --------
        tuple: (length, info, img)
            length : float
                두 점 사이의 유클리드 거리
            info : tuple
                (x1, y1, x2, y2, cx, cy) - 중심점 포함
            img : numpy.ndarray
                시각화된 이미지 (img=None이면 None)
        
        계산 공식:
        ----------
        거리 = √((x2-x1)² + (y2-y1)²)
        
        예시:
        -----
        # 검지 끝(8번)과 중지 끝(12번) 사이 거리
        length, info, img = detector.findDistance(
            lmList[8][0:2],
            lmList[12][0:2],
            img
        )
        print(f"거리: {length} 픽셀")
        """
        
        # 점 좌표 언패킹
        x1, y1 = p1
        x2, y2 = p2
        
        # 중심점 계산
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # 유클리드 거리 계산
        # hypot(x, y) = √(x² + y²)
        length = math.hypot(x2 - x1, y2 - y1)
        
        # 정보 튜플
        info = (x1, y1, x2, y2, cx, cy)
        
        # 이미지에 시각화 (옵션)
        if img is not None:
            # 첫 번째 점 그리기
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            
            # 두 번째 점 그리기
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            
            # 두 점을 연결하는 선 그리기
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            
            # 중심점 그리기
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)
        
        return length, info, img

# =============================================================================
# 테스트 메인 함수
# =============================================================================

def main():
    """
    HandDetector 테스트 프로그램
    
    웹캠에서 영상을 받아 손을 추적하고
    손가락 개수와 거리를 화면에 표시합니다.
    
    종료: 아무 키나 누르면 종료
    """

    # 웹캠 초기화
    # 0: 기본 카메라
    cap = cv2.VideoCapture(0)
    
    # HandDetector 초기화
    # - staticMode=False: 비디오 모드
    # - maxHands=2: 양손 추적
    # - modelComplexity=1: Full 모델
    # - detectionCon=0.5: 검출 신뢰도 50%
    # - minTrackCon=0.5: 추적 신뢰도 50%
    detector = HandDetector(
        staticMode=False,
        maxHands=2,
        modelComplexity=1,
        detectionCon=0.5,
        minTrackCon=0.5
    )
    
    # 무한 루프
    while True:
        # 프레임 읽기
        success, img = cap.read()
        
        # 손 검출
        hands, img = detector.findHands(img, draw=True, flipType=True)
        
        # 손이 검출된 경우
        if hands:
            # 첫 번째 손 정보
            hand1 = hands[0]
            lmList1 = hand1["lmList"]    # 랜드마크 리스트
            bbox1 = hand1["bbox"]        # 바운딩 박스
            center1 = hand1['center']    # 중심점
            handType1 = hand1["type"]    # 손 타입
            
            # 손가락 개수 계산
            fingers1 = detector.fingersUp(hand1)
            print(f'H1 = {fingers1.count(1)}', end=" ")
            
            # 검지(8번)와 중지(12번) 끝 사이 거리 계산
            length, info, img = detector.findDistance(
                lmList1[8][0:2],     # 검지 끝 (x, y)
                lmList1[12][0:2],    # 중지 끝 (x, y)
                img,
                color=(255, 0, 255),
                scale=10
            )
            
            # 두 번째 손이 검출된 경우
            if len(hands) == 2:
                # 두 번째 손 정보
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                center2 = hand2['center']
                handType2 = hand2["type"]
                
                # 손가락 개수 계산
                fingers2 = detector.fingersUp(hand2)
                print(f'H2 = {fingers2.count(1)}', end=" ")
                
                # 양손의 검지 끝 사이 거리 계산
                length, info, img = detector.findDistance(
                    lmList1[8][0:2],  # 첫 번째 손 검지
                    lmList2[8][0:2],  # 두 번째 손 검지
                    img,
                    color=(255, 0, 0),
                    scale=10
                )
            
            print(" ")  # 줄바꿈
        
        # 화면에 표시
        cv2.imshow("Image", img)
        
        # 키 입력 대기 (1ms)
        cv2.waitKey(1)

# =============================================================================
# 프로그램 진입점
# =============================================================================

if __name__ == "__main__":
    """
    이 모듈을 직접 실행할 때만 main() 호출
    
    다른 파일에서 import 할 때는 main() 실행 안 됨
    """
    main()