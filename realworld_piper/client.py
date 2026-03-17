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
JOINT_SCALE_RAD = math.pi / 180000.0  # Joint angle scale (0.001 degrees to radians)
CAMERA_WARMUP_SECONDS = 10.0
GRIPPER_OPEN_STROKE_M = 0.03 * 2.0
GRIPPER_CLOSE_STROKE_M = 0.01 * 2.0
SDK_JOINT_MODE_FLAG = 0x01  # Flag for joint mode in MotionCtrl_2
GRIPPER_EFFORT = 1000  # Default gripper effort (1N = 1000)
GRIPPER_EFFORT_MIN = 500  # Minimum gripper effort (0.5N)
GRIPPER_EFFORT_MAX = 3000  # Maximum gripper effort (3N)
SDK_ENABLE_CODE = 0x01
SDK_MOTION_CTRL_MODE = 0xAD  # Motion control mode for joint/endpose control
SDK_ENDPOSE_MODE_FLAG = 0x00  # Flag for endpose mode in MotionCtrl_2
SDK_MAX_SPEED_PERCENT = 100  # Maximum speed percentage
SDK_DEFAULT_SPEED_PERCENT = 50  # Default speed percentage when velocity not specified
GRIPPER_DEADZONE = 200  # Gripper deadzone threshold
GRIPPER_MAX_STROKE = 80000  # Maximum gripper stroke
GRIPPER_MIN_STROKE = 0  # Minimum gripper stroke
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


def read_joint_state_m(piper: C_PiperInterface_V2) -> list[float]:
    """Read joint angles in radians (6 joints + gripper)."""
    raw_msg = piper.GetArmJointMsgs()
    joint_state = raw_msg.joint_state

    # Convert from 0.001 degrees to radians
    return [
        float(joint_state.joint_1) * JOINT_SCALE_RAD,
        float(joint_state.joint_2) * JOINT_SCALE_RAD,
        float(joint_state.joint_3) * JOINT_SCALE_RAD,
        float(joint_state.joint_4) * JOINT_SCALE_RAD,
        float(joint_state.joint_5) * JOINT_SCALE_RAD,
        float(joint_state.joint_6) * JOINT_SCALE_RAD,
    ]


def read_gripper_qpos_m(piper: C_PiperInterface_V2) -> list[float]:
    raw_msg = piper.GetArmGripperMsgs()
    stroke = raw_msg.gripper_state.grippers_angle

    stroke_m = float(stroke) * POS_SCALE_M
    half = stroke_m / 2.0
    return [half, -half]


def read_state(piper: C_PiperInterface_V2, control_mode: str = "endpose") -> list[float]:
    """Read current state based on control mode.

    Args:
        piper: Piper interface instance
        control_mode: "endpose" for end-effector pose, "joint" for joint angles

    Returns:
        List of 8 floats: 6 pose/joint values + 2 gripper values
    """
    if control_mode == "joint":
        return read_joint_state_m(piper) + read_gripper_qpos_m(piper)
    else:
        return read_eef_state_m(piper) + read_gripper_qpos_m(piper)


def target_gripper_qpos(gripper_action: float, open_stroke: float = None, close_stroke: float = None) -> list[float]:
    if open_stroke is None:
        open_stroke = GRIPPER_OPEN_STROKE_M
    if close_stroke is None:
        close_stroke = GRIPPER_CLOSE_STROKE_M
    stroke_m = open_stroke if gripper_action < 0 else close_stroke
    half = stroke_m / 2.0
    return [half, -half]


def angle_rad_to_joint_sdk(value_rad: float) -> int:
    """Convert radians to joint SDK units (0.001 degrees)."""
    return round(value_rad / JOINT_SCALE_RAD)


def clamp_gripper_stroke(stroke: int) -> int:
    """Clamp gripper stroke to valid range and apply deadzone.
    
    Args:
        stroke: Gripper stroke value
        
    Returns:
        Clamped gripper stroke value
    """
    # Apply deadzone
    if abs(stroke) < GRIPPER_DEADZONE:
        stroke = 0
    # Clamp to valid range
    return max(GRIPPER_MIN_STROKE, min(abs(stroke), GRIPPER_MAX_STROKE))


def clamp_gripper_effort(effort: float) -> int:
    """Convert and clamp gripper effort to SDK units.
    
    Args:
        effort: Gripper effort in Newtons (typical range: 0.5-3.0N)
        
    Returns:
        Gripper effort in SDK units (1000 = 1N)
    """
    # Clamp effort to valid range
    clamped_effort = max(0.5, min(effort, 3.0))
    return round(clamped_effort * 1000)


def clamp_motion_speed(speed: int) -> int:
    """Clamp motion speed to valid range.
    
    Args:
        speed: Motion speed percentage
        
    Returns:
        Clamped speed value in range [0, 100]
    """
    return max(0, min(speed, SDK_MAX_SPEED_PERCENT))


def send_action_sequence(
    piper: C_PiperInterface_V2,
    actions: list[list[float]],
    action_delay: float,
    motion_speed_percent: int,
    control_mode: str = "endpose",
    gripper_open_stroke: float = None,
    gripper_close_stroke: float = None,
    gripper_effort: float = None,
) -> None:
    """Send action sequence to Piper arm.

    Args:
        piper: Piper interface instance
        actions: List of actions, each action is [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        action_delay: Delay between actions in seconds
        motion_speed_percent: Motion speed percentage (will be clamped to 0-100)
        control_mode: "endpose" or "joint"
        gripper_open_stroke: Gripper open stroke in meters
        gripper_close_stroke: Gripper close stroke in meters
        gripper_effort: Gripper effort in Newtons (range: 0.5-3.0N, default: 1.0N)
    """
    for idx, action in enumerate(actions):
        if len(action) != 7:
            raise RuntimeError(f"Expected action length 7, got {len(action)} at index {idx}")

        current_state = read_state(piper, control_mode)
        target_values = [current_state[i] + float(action[i]) for i in range(6)]
        target_gripper = target_gripper_qpos(float(action[6]), gripper_open_stroke, gripper_close_stroke)
        current_gripper_stroke = pos_m_to_sdk(abs(current_state[6] - current_state[7]))
        gripper_stroke = pos_m_to_sdk(abs(target_gripper[0] - target_gripper[1]))

        if control_mode == "joint":
            # Joint mode control
            joint_1 = angle_rad_to_joint_sdk(target_values[0])
            joint_2 = angle_rad_to_joint_sdk(target_values[1])
            joint_3 = angle_rad_to_joint_sdk(target_values[2])
            joint_4 = angle_rad_to_joint_sdk(target_values[3])
            joint_5 = angle_rad_to_joint_sdk(target_values[4])
            joint_6 = angle_rad_to_joint_sdk(target_values[5])

            # Clamp speed to valid range [0, 100]
            clamped_speed = clamp_motion_speed(motion_speed_percent)
            
            # MotionCtrl_1: Initialize motion control
            piper.MotionCtrl_1(0x00, 0x00, 0x00)
            
            # MotionCtrl_2(enable, joint_mode_flag, speed, mode)
            # Using 0xAD mode for joint control as per ROS callback implementation
            piper.MotionCtrl_2(SDK_ENABLE_CODE, SDK_JOINT_MODE_FLAG, clamped_speed, SDK_MOTION_CTRL_MODE)
            piper.JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)

            print(f"executed action[{idx}]:", action)
            print(
                "previous_state_api_units (joint):",
                tuple(angle_rad_to_joint_sdk(current_state[i]) for i in range(6)),
            )
            print(
                "target_state_api_units (joint):",
                (joint_1, joint_2, joint_3, joint_4, joint_5, joint_6),
            )
            print(
                "JointCtrl args:",
                (joint_1, joint_2, joint_3, joint_4, joint_5, joint_6),
            )
        else:
            # End pose mode control
            current_x = pos_m_to_sdk(current_state[0])
            current_y = pos_m_to_sdk(current_state[1])
            current_z = pos_m_to_sdk(current_state[2])
            current_rx = angle_rad_to_sdk(current_state[3])
            current_ry = angle_rad_to_sdk(current_state[4])
            current_rz = angle_rad_to_sdk(current_state[5])
            target_x = pos_m_to_sdk(target_values[0])
            target_y = pos_m_to_sdk(target_values[1])
            target_z = pos_m_to_sdk(target_values[2])
            target_rx = angle_rad_to_sdk(target_values[3])
            target_ry = angle_rad_to_sdk(target_values[4])
            target_rz = angle_rad_to_sdk(target_values[5])

            # Clamp speed to valid range [0, 100]
            clamped_speed = clamp_motion_speed(motion_speed_percent)

            # MotionCtrl_1: Initialize motion control
            piper.MotionCtrl_1(0x00, 0x00, 0x00)
            
            # MotionCtrl_2(enable, endpose_mode_flag, speed, mode)
            # Using 0xAD mode for endpose control as per ROS callback implementation
            piper.MotionCtrl_2(SDK_ENABLE_CODE, SDK_ENDPOSE_MODE_FLAG, clamped_speed, SDK_MOTION_CTRL_MODE)
            piper.EndPoseCtrl(
                target_x,
                target_y,
                target_z,
                target_rx,
                target_ry,
                target_rz,
            )

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

        # Gripper control with deadzone and range limits
        gripper_stroke = pos_m_to_sdk(abs(target_gripper[0] - target_gripper[1]))
        gripper_stroke = clamp_gripper_stroke(gripper_stroke)
        
        # Use provided gripper effort or default
        effort_sdk = clamp_gripper_effort(gripper_effort) if gripper_effort else GRIPPER_EFFORT
        
        piper.GripperCtrl(gripper_stroke, effort_sdk, SDK_ENABLE_CODE, 0)
        print(
            "GripperCtrl args:",
            (gripper_stroke, effort_sdk, SDK_ENABLE_CODE, 0),
        )
        
        # Final MotionCtrl_2 call for endpose mode (matching ROS pos_callback behavior)
        # Note: Joint mode does not have this final call, matching ROS joint_callback behavior
        if control_mode != "joint":
            piper.MotionCtrl_2(SDK_ENABLE_CODE, SDK_ENDPOSE_MODE_FLAG, clamped_speed, SDK_MOTION_CTRL_MODE)
        
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
    control_mode: str = "endpose",
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
        "state": read_state(piper, control_mode),
        "control_mode": control_mode,
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
    parser.add_argument(
        "--steps-per-inference",
        type=int,
        default=1,
        help="Number of actions to execute before starting next inference (async mode only)",
    )
    parser.add_argument(
        "--async",
        action="store_true",
        dest="async_mode",
        help="Enable async pipeline mode (overlap inference and execution)",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument(
        "--save-request-dir",
        type=str,
        default=None,
        help="If set, save each request/response/images to a per-request subfolder in this directory",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="endpose",
        choices=["endpose", "joint"],
        help="Control mode: endpose (end-effector pose) or joint (joint angles)",
    )
    parser.add_argument(
        "--gripper-open-stroke",
        type=float,
        default=0.06,
        help="Gripper open stroke in meters (default: 0.06)",
    )
    parser.add_argument(
        "--gripper-close-stroke",
        type=float,
        default=0.02,
        help="Gripper close stroke in meters (default: 0.02)",
    )
    parser.add_argument(
        "--gripper-effort",
        type=float,
        default=None,
        help="Gripper effort in Newtons (range: 0.5-3.0N, default: 1.0N)",
    )
    return parser.parse_args()


def configure_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width != width or actual_height != height:
        print(f"Warning: Camera {index} requested {width}x{height}, got {actual_width}x{actual_height}")

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


class InferenceThread(threading.Thread):
    """Thread for running inference in background."""

    def __init__(self, session: requests.Session, endpoint: str, payload: dict[str, Any]):
        super().__init__(daemon=True)
        self.session = session
        self.endpoint = endpoint
        self.payload = payload
        self.result = None
        self.error = None

    def run(self):
        try:
            response = self.session.post(self.endpoint, json=self.payload, timeout=30)
            response.raise_for_status()
            self.result = response.json()
        except Exception as e:
            self.error = e

    def is_done(self) -> bool:
        return not self.is_alive() and self.result is not None

    def get_result(self) -> list[list[float]]:
        if self.error:
            raise self.error
        return self.result


def run_async_pipeline(
    piper: C_PiperInterface_V2,
    session: requests.Session,
    server_endpoint: str,
    actions: list[list[float]],
    n: int,
    action_delay: float,
    motion_speed_percent: int,
    save_root: Path | None,
    request_idx: int,
    instruction: str,
    control_mode: str = "endpose",
    gripper_open_stroke: float = None,
    gripper_close_stroke: float = None,
    gripper_effort: float = None,
) -> int:
    """Run async pipeline mode (overlapping inference and execution)."""
    actions_executed = 0
    pending_inference: InferenceThread | None = None
    actions_executed_at_inference_start = 0

    while True:
        # Execute first chunk (n steps)
        chunk_end = min(actions_executed + n, len(actions))
        chunk = actions[actions_executed:chunk_end]
        if not chunk:
            break

        send_action_sequence(piper, chunk, action_delay, motion_speed_percent,
                            control_mode, gripper_open_stroke, gripper_close_stroke, gripper_effort)
        actions_executed += len(chunk)

        # After executing first chunk, START next inference in background thread
        payload = build_payload(piper, instruction, control_mode)
        actions_executed_at_inference_start = actions_executed
        pending_inference = InferenceThread(session, server_endpoint, payload)
        pending_inference.start()

        # Execute remaining actions one by one, while inference runs in background
        while actions_executed < len(actions):
            single = [actions[actions_executed]]
            send_action_sequence(piper, single, action_delay, motion_speed_percent,
                                control_mode, gripper_open_stroke, gripper_close_stroke)
            actions_executed += 1

            if pending_inference is not None and pending_inference.is_done():
                raw_actions = pending_inference.get_result()
                request_idx += 1

                skip_steps = actions_executed - actions_executed_at_inference_start
                new_actions = raw_actions[skip_steps:] if skip_steps > 0 else raw_actions

                if save_root is not None:
                    save_request_record(save_root, request_idx, payload, new_actions, "")

                print(f"[Inference {request_idx}] state: {payload['state']}")
                print(f"[Inference {request_idx}] actions: {new_actions}")
                print(f"[Inference {request_idx}] skipped first {skip_steps} actions")

                actions = new_actions
                actions_executed = 0

                chunk = actions[:n]
                if chunk:
                    send_action_sequence(piper, chunk, action_delay, motion_speed_percent,
                            control_mode, gripper_open_stroke, gripper_close_stroke, gripper_effort)
                    actions_executed = len(chunk)

                    payload = build_payload(piper, instruction, control_mode)
                    actions_executed_at_inference_start = actions_executed
                    pending_inference = InferenceThread(session, server_endpoint, payload)
                    pending_inference.start()
                    break
                else:
                    pending_inference = None
                    break

        # If we finished executing all actions but have a pending inference
        if actions_executed >= len(actions) and pending_inference is not None:
            pending_inference.join()
            if pending_inference.error:
                raise pending_inference.error

            raw_actions = pending_inference.get_result()
            request_idx += 1

            skip_steps = actions_executed - actions_executed_at_inference_start
            new_actions = raw_actions[skip_steps:] if skip_steps > 0 else raw_actions

            if save_root is not None:
                save_request_record(save_root, request_idx, payload, new_actions, "")

            print(f"[Inference {request_idx}] state: {payload['state']}")
            print(f"[Inference {request_idx}] actions: {new_actions}")
            print(f"[Inference {request_idx}] skipped first {skip_steps} actions")

            actions = new_actions
            actions_executed = 0
            pending_inference = None
        elif actions_executed >= len(actions):
            break

    return request_idx


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_request_dir) if args.save_request_dir else None
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
    request_idx = 0
    stop_event = threading.Event()

    # Initialize Piper arm
    piper = C_PiperInterface_V2(args.can_port)
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    piper.GripperCtrl(0, GRIPPER_EFFORT, SDK_ENABLE_CODE, 0)

    # Initialize cameras
    full_cam = configure_camera(args.full_camera_index, args.width, args.height)
    wrist_cam = configure_camera(args.wrist_camera_index, args.width, args.height)
    capture_thread = threading.Thread(target=capture_worker, args=(full_cam, wrist_cam, stop_event), daemon=True)
    capture_thread.start()

    try:
        warmup_cameras()

        n = args.steps_per_inference  # steps to execute before next inference (0 = sync mode)
        action_delay = args.action_delay
        motion_speed_percent = args.motion_speed_percent

        # Use sessions for connection pooling
        session = requests.Session()

        # === Main loop ===
        while True:
            # === First inference (blocking) ===
            payload = build_payload(piper, args.instruction, args.control_mode)
            response = session.post(args.server_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            actions = response.json()
            request_idx += 1

            if save_root is not None:
                save_request_record(save_root, request_idx, payload, actions, response.text)

            print(f"[Inference {request_idx}] state: {payload['state']}")
            print(f"[Inference {request_idx}] actions: {actions}")

            if args.async_mode:
                # === Async pipeline mode ===
                run_async_pipeline(piper, session, args.server_endpoint, actions, n, action_delay,
                                   motion_speed_percent, save_root, request_idx, args.instruction,
                                   args.control_mode, args.gripper_open_stroke, args.gripper_close_stroke,
                                   args.gripper_effort)
            else:
                # === Sync mode (original) ===
                send_action_sequence(piper, actions, action_delay, motion_speed_percent,
                            args.control_mode, args.gripper_open_stroke, args.gripper_close_stroke,
                            args.gripper_effort)

            time.sleep(args.period)

    finally:
        stop_event.set()
        capture_thread.join(timeout=1.0)
        full_cam.release()
        wrist_cam.release()


if __name__ == "__main__":
    main()