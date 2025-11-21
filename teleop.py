import argparse

from ruka_hand.teleoperation.manus_teleoperator import ManusTeleoperator
from ruka_hand.teleoperation.oculus_teleoperator import OculusTeleoperator
from ruka_hand.teleoperation.webcam_teleoperator import WebCamTeleoperator  # 웹캠 텔레오퍼레이터 추가
from ruka_hand.utils.constants import HOST, OCULUS_LEFT_PORT, OCULUS_RIGHT_PORT

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Teleop robot hands.")
	parser.add_argument(
		"-ht",
		"–hand_type",
		type=str,
		help="Hand you’d like to teleoperate",
		default="",
	)

	parser.add_argument(
		"-m",
		"--mode",
		type=str,
		help="Mode you'd like to teleoperate (manus, oculus, webcam)",
		default="manus",
	)

	# 웹캠 모드 옵션 추가
	parser.add_argument(
		"-c",
		"--camera",
		type=int,
		help="Camera ID for webcam mode (default: 0)",
		default=0,
	)

	parser.add_argument(
		"-f",
		"--frequency",
		type=int,
		help="Control frequency in Hz (default: 30 for webcam)",
		default=None,
	)

	args = parser.parse_args()

	if args.mode == "manus":
		frequency = args.frequency if args.frequency else 50
		manus_teleoperator = ManusTeleoperator(
			hand_names=[args.hand_type],
			frequency=frequency,
			record=False,
		)
		manus_teleoperator.run()

	elif args.mode == "oculus":
		frequency = args.frequency if args.frequency else 90
		oculus_teleoperator = OculusTeleoperator(
			HOST,
			OCULUS_LEFT_PORT,
			OCULUS_RIGHT_PORT,
			frequency,
			hands=[args.hand_type],
		)
		oculus_teleoperator.run()

	elif args.mode == "webcam":
		frequency = args.frequency if args.frequency else 30
		hands = [args.hand_type] if args.hand_type else ["left", "right"]

		webcam_teleoperator = WebCamTeleoperator(
			camera_id=args.camera,
			frequency=frequency,
			moving_average_limit=10,
			hands=hands,
			detection_confidence=0.7,
			tracking_confidence=0.7,
		)
		webcam_teleoperator.run()

	else:
		print(f"[ERROR] Unknown mode: {args.mode}")
		print("Available modes: manus, oculus, webcam")