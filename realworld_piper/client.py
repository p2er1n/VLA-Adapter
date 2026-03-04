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
CAMERA_WARMUP_SECONDS = 10.0
GRIPPER_OPEN_STROKE_M = 0.03 * 2.0
GRIPPER_CLOSE_STROKE_M = 0.01 * 2.0
GRIPPER_EFFORT = 1000
SDK_ENABLE_CODE = 0x01


def pos_m_to_sdk(value_m: float) -> int:
    return round(value_m / POS_SCALE_M)


def angle_rad_to_sdk(value_rad: float) -> int:
    return round(value_rad / ANGLE_SCALE_RAD)


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


def target_gripper_qpos(gripper_action: float) -> list[float]:
    stroke_m = GRIPPER_OPEN_STROKE_M if gripper_action < 0 else GRIPPER_CLOSE_STROKE_M
    half = stroke_m / 2.0
    return [half, -half]


def send_action_sequence(
    piper: C_PiperInterface_V2,
    actions: list[list[float]],
    action_delay: float,
) -> None:
    for idx, action in enumerate(actions):
        if len(action) != 7:
            raise RuntimeError(f"Expected action length 7, got {len(action)} at index {idx}")

        current_state = read_state(piper)
        target_pose = [current_state[i] + float(action[i]) for i in range(6)]
        target_gripper = target_gripper_qpos(float(action[6]))
        current_x = pos_m_to_sdk(current_state[0])
        current_y = pos_m_to_sdk(current_state[1])
        current_z = pos_m_to_sdk(current_state[2])
        current_rx = angle_rad_to_sdk(current_state[3])
        current_ry = angle_rad_to_sdk(current_state[4])
        current_rz = angle_rad_to_sdk(current_state[5])
        current_gripper_stroke = pos_m_to_sdk(abs(current_state[6] - current_state[7]))
        target_x = pos_m_to_sdk(target_pose[0])
        target_y = pos_m_to_sdk(target_pose[1])
        target_z = pos_m_to_sdk(target_pose[2])
        target_rx = angle_rad_to_sdk(target_pose[3])
        target_ry = angle_rad_to_sdk(target_pose[4])
        target_rz = angle_rad_to_sdk(target_pose[5])
        gripper_stroke = pos_m_to_sdk(abs(target_gripper[0] - target_gripper[1]))

        piper.MotionCtrl_2(SDK_ENABLE_CODE, 0x00, 100, 0x00)
        piper.EndPoseCtrl(
            target_x,
            target_y,
            target_z,
            target_rx,
            target_ry,
            target_rz,
        )
        piper.GripperCtrl(gripper_stroke, GRIPPER_EFFORT, SDK_ENABLE_CODE, 0)

        print(f"executed action[{idx}]:", action)
        print(
            "previous_state_api_units:",
            (current_x, current_y, current_z, current_rx, current_ry, current_rz, current_gripper_stroke),
        )
        print(
            "target_state_api_units:",
            (target_x, target_y, target_z, target_rx, target_ry, target_rz, gripper_stroke),
        )
        print(
            "EndPoseCtrl args:",
            (target_x, target_y, target_z, target_rx, target_ry, target_rz),
        )
        print(
            "GripperCtrl args:",
            (gripper_stroke, GRIPPER_EFFORT, SDK_ENABLE_CODE, 0),
        )
        time.sleep(action_delay)


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
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--full-camera-index", type=int, default=0)
    parser.add_argument("--wrist-camera-index", type=int, default=1)
    parser.add_argument("--period", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--action-delay", type=float, default=0.01, help="Seconds between executing returned actions")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def configure_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def warmup_cameras(full_cam: cv2.VideoCapture, wrist_cam: cv2.VideoCapture) -> None:
    start_time = time.time()
    while time.time() - start_time < CAMERA_WARMUP_SECONDS:
        ok_full, full_frame = full_cam.read()
        ok_wrist, wrist_frame = wrist_cam.read()

        if not ok_full or full_frame is None:
            raise RuntimeError("Failed to warm up full camera")
        if not ok_wrist or wrist_frame is None:
            raise RuntimeError("Failed to warm up wrist camera")

        cv2.imshow("full_camera", full_frame)
        cv2.imshow("wrist_camera", wrist_frame)
        cv2.waitKey(1)


def main() -> None:
    args = parse_args()

    piper = C_PiperInterface_V2(args.can_port)
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    piper.GripperCtrl(0, GRIPPER_EFFORT, SDK_ENABLE_CODE, 0)

    full_cam = configure_camera(args.full_camera_index, args.width, args.height)
    wrist_cam = configure_camera(args.wrist_camera_index, args.width, args.height)
    warmup_cameras(full_cam, wrist_cam)

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
            send_action_sequence(piper, actions, args.action_delay)

            time.sleep(args.period)
    finally:
        full_cam.release()
        wrist_cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
