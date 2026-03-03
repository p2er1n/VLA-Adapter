#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import time
from typing import Any

import cv2
import requests

from piper_sdk import C_PiperInterface_V2


POS_SCALE_M = 1e-6
ANGLE_SCALE_RAD = math.pi / 180000.0
CAMERA_WARMUP_FRAMES = 50


def read_eef_state_m(piper: C_PiperInterface_V2) -> list[float]:
    raw_msg = piper.GetArmEndPoseMsgs()
    pose = raw_msg.end_pose

    return [
        float(pose.X_axis) * POS_SCALE_M,
        float(pose.Y_axis) * POS_SCALE_M,
        float(pose.Z_axis) * POS_SCALE_M,
        float(pose.RX_axis) * ANGLE_SCALE_RAD,
        float(pose.RY_axis) * ANGLE_SCALE_RAD,
        float(pose.RZ_axis) * ANGLE_SCALE_RAD,
    ]


def read_gripper_qpos_m(piper: C_PiperInterface_V2) -> list[float]:
    raw_msg = piper.GetArmGripperMsgs()
    stroke = raw_msg.gripper_state.grippers_angle

    stroke_m = float(stroke) * POS_SCALE_M
    half = stroke_m / 2.0
    return [half, -half]


def read_state(piper: C_PiperInterface_V2) -> list[float]:
    return read_eef_state_m(piper) + read_gripper_qpos_m(piper)


def read_camera_frame(cap: cv2.VideoCapture, camera_name: str) -> tuple[list, Any]:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from {camera_name}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb.tolist(), frame


def build_payload(
    piper: C_PiperInterface_V2,
    full_cam: cv2.VideoCapture,
    wrist_cam: cv2.VideoCapture,
    instruction: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_image, full_preview = read_camera_frame(full_cam, "full camera")
    wrist_image, wrist_preview = read_camera_frame(wrist_cam, "wrist camera")

    payload = {
        "instruction": instruction,
        "state": read_state(piper),
        "full_image": full_image,
        "wrist_image": wrist_image,
    }
    preview_frames = {
        "full": full_preview,
        "wrist": wrist_preview,
    }
    return payload, preview_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client for realworld_piper/server.py")
    parser.add_argument("instruction", help="Language instruction sent to the server")
    parser.add_argument("--server-endpoint", default="http://127.0.0.1:6006/act")
    parser.add_argument("--full-camera-index", type=int, default=0)
    parser.add_argument("--wrist-camera-index", type=int, default=1)
    parser.add_argument("--period", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--once", action="store_true", help="Send one request and exit")
    return parser.parse_args()


def configure_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(CAMERA_WARMUP_FRAMES):
        ok, _frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to warm up camera index {index}")

    return cap


def main() -> None:
    args = parse_args()

    piper = C_PiperInterface_V2()
    piper.ConnectPort()

    full_cam = configure_camera(args.full_camera_index, args.width, args.height)
    wrist_cam = configure_camera(args.wrist_camera_index, args.width, args.height)

    try:
        while True:
            payload, preview_frames = build_payload(piper, full_cam, wrist_cam, args.instruction)

            cv2.imshow("full_camera", preview_frames["full"])
            cv2.imshow("wrist_camera", preview_frames["wrist"])
            cv2.waitKey(1)

            response = requests.post(args.server_endpoint, json=payload, timeout=30)
            response.raise_for_status()

            actions = response.json()
            print("state:", payload["state"])
            print("actions:", actions)

            if args.once:
                break

            time.sleep(args.period)
    finally:
        full_cam.release()
        wrist_cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
