#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests

from piper_sdk import C_PiperInterface_V2


POS_SCALE_M = 1e-6
ANGLE_SCALE_RAD = math.pi / 180000.0
CAMERA_WARMUP_SECONDS = 10.0
GRIPPER_OPEN_STROKE_M = 0.03 * 2.0
GRIPPER_CLOSE_STROKE_M = 0.01 * 2.0
GRIPPER_EFFORT = 1000
SDK_ENABLE_CODE = 0x01
FRAME_LOCK = threading.Lock()
LATEST_FRAMES = {"full": None, "wrist": None}


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
    motion_speed_percent: int,
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

        piper.MotionCtrl_2(SDK_ENABLE_CODE, 0x00, motion_speed_percent, 0x00)
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


def compress_image_to_base64(image: np.ndarray, quality: int = 100) -> str:
    """Compress image to JPEG and encode as base64 string."""
    # Encode image as JPEG with specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', image, encode_param)

    # Convert to base64 string
    base64_string = base64.b64encode(encoded_img).decode('utf-8')
    return base64_string


def build_payload(
    piper: C_PiperInterface_V2,
    instruction: str,
) -> dict[str, Any]:
    with FRAME_LOCK:
        full_frame = None if LATEST_FRAMES["full"] is None else LATEST_FRAMES["full"].copy()
        wrist_frame = None if LATEST_FRAMES["wrist"] is None else LATEST_FRAMES["wrist"].copy()

    if full_frame is None or wrist_frame is None:
        raise RuntimeError("Latest camera frames are not ready yet")

    # Compress images to base64-encoded JPEG strings
    full_image_compressed = compress_image_to_base64(full_frame)
    wrist_image_compressed = compress_image_to_base64(wrist_frame)

    payload = {
        "instruction": instruction,
        "state": read_state(piper),
        "full_image": full_image_compressed,
        "wrist_image": wrist_image_compressed,
        "image_format": "jpeg",  # Indicate compression format
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Client for realworld_piper/server.py")
    parser.add_argument("instruction", help="Language instruction sent to the server")
    parser.add_argument("--server-endpoint", default="http://127.0.0.1:6006/act")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--full-camera-index", type=int, default=0)
    parser.add_argument("--wrist-camera-index", type=int, default=1)
    parser.add_argument("--period", type=float, default=0, help="Seconds between requests")
    parser.add_argument("--action-delay", type=float, default=0.2, help="Seconds between executing returned actions")
    parser.add_argument(
        "--motion-speed-percent",
        type=int,
        default=2,
        help="Motion speed percentage for MotionCtrl_2 (range depends on SDK, commonly 0-100)",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--save-request-dir",
        type=str,
        default=None,
        help="If set, save each request/response/images to a per-request subfolder in this directory",
    )
    return parser.parse_args()


def configure_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def capture_worker(
    full_cam: cv2.VideoCapture,
    wrist_cam: cv2.VideoCapture,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        ok_full, full_frame = full_cam.read()
        ok_wrist, wrist_frame = wrist_cam.read()

        if ok_full and full_frame is not None and ok_wrist and wrist_frame is not None:
            with FRAME_LOCK:
                LATEST_FRAMES["full"] = full_frame
                LATEST_FRAMES["wrist"] = wrist_frame

        time.sleep(0.001)


def warmup_cameras() -> None:
    start_time = time.time()
    while time.time() - start_time < CAMERA_WARMUP_SECONDS:
        with FRAME_LOCK:
            full_ready = LATEST_FRAMES["full"] is not None
            wrist_ready = LATEST_FRAMES["wrist"] is not None
        if not (full_ready and wrist_ready):
            time.sleep(0.01)
            continue

        time.sleep(0.005)


def save_request_record(
    save_root: Path,
    request_idx: int,
    payload: dict[str, Any],
    response_json: Any,
    response_text: str,
) -> None:
    request_dir = save_root / f"request_{request_idx:06d}"
    request_dir.mkdir(parents=True, exist_ok=False)

    with open(request_dir / "request.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with open(request_dir / "response.json", "w", encoding="utf-8") as f:
        json.dump(response_json, f, ensure_ascii=False)

    with open(request_dir / "response_raw.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    # Save images based on their format
    if payload.get("image_format") == "jpeg":
        # Decode compressed images and save as PNG for viewing
        full_jpg_data = base64.b64decode(payload["full_image"])
        wrist_jpg_data = base64.b64decode(payload["wrist_image"])

        full_img = cv2.imdecode(np.frombuffer(full_jpg_data, np.uint8), cv2.IMREAD_COLOR)
        wrist_img = cv2.imdecode(np.frombuffer(wrist_jpg_data, np.uint8), cv2.IMREAD_COLOR)

        cv2.imwrite(str(request_dir / "full_image.png"), full_img)
        cv2.imwrite(str(request_dir / "wrist_image.png"), wrist_img)
    else:
        # Legacy format - direct RGB arrays
        full_rgb = np.asarray(payload["full_image"], dtype=np.uint8)
        wrist_rgb = np.asarray(payload["wrist_image"], dtype=np.uint8)
        cv2.imwrite(str(request_dir / "full_image.png"), cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(request_dir / "wrist_image.png"), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_request_dir) if args.save_request_dir else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
    request_idx = 0
    stop_event = threading.Event()

    piper = C_PiperInterface_V2(args.can_port)
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    piper.GripperCtrl(0, GRIPPER_EFFORT, SDK_ENABLE_CODE, 0)

    full_cam = configure_camera(args.full_camera_index, args.width, args.height)
    wrist_cam = configure_camera(args.wrist_camera_index, args.width, args.height)
    capture_thread = threading.Thread(target=capture_worker, args=(full_cam, wrist_cam, stop_event), daemon=True)
    capture_thread.start()

    try:
        warmup_cameras()
        while True:
            payload = build_payload(piper, args.instruction)

            response = requests.post(args.server_endpoint, json=payload, timeout=30)
            response.raise_for_status()

            actions = response.json()
            if save_root is not None:
                request_idx += 1
                save_request_record(
                    save_root=save_root,
                    request_idx=request_idx,
                    payload=payload,
                    response_json=actions,
                    response_text=response.text,
                )
            print("state:", payload["state"])
            print("actions:", actions)
            send_action_sequence(
                piper,
                actions,
                args.action_delay,
                args.motion_speed_percent,
            )

            time.sleep(args.period)
    finally:
        stop_event.set()
        capture_thread.join(timeout=1.0)
        full_cam.release()
        wrist_cam.release()


if __name__ == "__main__":
    main()
