"""
Convert teleoperation HDF5 episodes into a LIBERO-style RLDS dataset.

Expected input file format:
    episode<n>.hdf5

Expected HDF5 keys:
    /arm/endPose/piper_end                    (T, 6)
    /gripper/encoderDistance/pika             (T,)
    /camera/color/pikaDepthCamera             (T, 480, 640, 3)  BGR
    /camera/color/pikaFisheyeCamera           (T, 480, 640, 3)  BGR

Output step format (per trajectory step):
    action                (7,)    first 6 dims are EEF deltas; last dim is binary gripper action
    observation.state     (8,)    first 6 dims are absolute EEF state; last 2 dims are gripper distances
    observation.image     (256, 256, 3) third-person RGB image
    observation.wrist_image
                          (256, 256, 3) wrist RGB image
    language_instruction          task instruction (stored as tf.string; read back as bytes)

Notes:
    - The generated dataset is TFDS/RLDS-style. It can be loaded by TensorFlow Datasets as a prepared dataset.
    - The script only implements the conversion logic described in the file header. It does not try to infer any
      additional robot semantics beyond those rules.
"""

# 本脚本将多个hdf5文件，转换为libero rlds格式的文件，方便后续训练使用
# 从指定文件夹读取hdf5文件，文件的命名有固定格式，episode<n>.hdf5，例如：episode0.hdf5, episode1.hdf5, episode40.hdf5等等
# 每个episode hdf5文件代表最终的rlds数据集的一个episode（即一个轨迹）
# 原来hdf5文件中的数据含义如下：
## /arm/endPose/piper_end：（len， 6），表示机械臂的EEF state，绝对值
## /gripper/encoderDistance/pika：（len， 1）,表示两个夹爪之间的距离，绝对值
## /camera/color/pikaDepthCamera: (len，480，640，3)，表示第三人称摄像头图片，BGR格式，480*640分辨率
## /camera/color/pikaFisheyeCamera: (len，480，640，3)，表示腕部摄像头图片，BGR格式，480*640分辨率
# 转换后的rlds数据集中的数据含义如下（每个轨迹中的一步）：
## action：（7，），前六维：相对值，EEF格式，表示下一个EEF状态和当前EEF状态的差值；最后一维：绝对值，二值化，-1表示夹爪打开，1表示夹爪闭合
## obeservation.state：（8，），前六维：绝对值，EEF state，表示机械臂EEF当前的状态（也就是hdf5的piper_end）；后两维：绝对值，第七维表示左夹爪到闭合处的距离（正值），第八维表示右夹爪到闭合处的距离的绝对值的负数（负值），比如：两个夹爪的距离为0.08，第七维为0.04，第八维为-0.04
## observation.image：（256, 256, 3），第三人称摄像头图片，RGB格式，分辨率为256*256，对应hdf5的pikaDepthCamera
## observation.wrist_image：（256, 256, 3），腕部摄像头图片，RGB格式，分辨率为256*256，对应hdf5的pikaFisheyeCamera
## language_instruction：bytes类型数据，表示当前轨迹的任务的指令，从命令行参数中获取
# 转换步骤：
## 1. 命令行参数指定文件夹，该文件夹下存放有episode<n>.hdf5命名的文件若干
## 2. 命令行参数指定遥操作的hdf5文件的范围，例如，0-49，表示按照顺序（按照<n>数值大小从小到大的顺序）将episode0.hdf5到episode49.hdf5转换为rlds数据集中的50个轨迹
## 3. 命令行参数指定每个轨迹的任务指令，格式为：<language_instruction>-<len>,<language_instruction2>-<len2>, ...，保证所有len相加等于上面指定的范围的长度
## 4. 命令行指定求差值的间隔，delta，默认为1
## 5. 命令行指定判定夹爪
## 6. 加载一个hdf5文件，
### 这个轨迹的第一步的observation.state[:6] = piper_end[0]，
### 这个轨迹的第一步的observation.state[-2:] = "encoderDistance/pika"[0]/2，-"encoderDistance/pika"[0]/2
### 这个轨迹的第一步的observation.image = bgr_to_rgb(resize(pikaDepthCamera[0], (256, 256)))
### 这个轨迹的第一步的observation.wrist_image = bgr_to_rgb(resize(pikaFisheyeCamera[0], (256, 256)))
### 这个轨迹的第一步的action[:6] = piper_end[delta] - piper_end[0]
### 这个轨迹的第一步的action[6]：
#### 计算两步之间的夹爪距离的差值，设定一个阈值，如果是负值并且绝对值超过阈值，就认为夹爪是“闭合”状态，如果是正值并且绝对值超过阈值，就认为夹爪是“打开”状态，如果绝对值没有超过阈值，就认为夹爪停在当前状态（之前是“闭合”就是“闭合”，之前是“打开”就是“打开”）。

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image

try:
    import tensorflow_datasets as tfds
except ImportError as exc:  # pragma: no cover - import guard for script usage
    raise ImportError(
        "This script requires `tensorflow-datasets`. Install it before running the converter."
    ) from exc


EPISODE_RE = re.compile(r"^episode(\d+)\.hdf5$")
IMAGE_SIZE = (256, 256)
DATASET_VERSION = "1.0.0"


@dataclass(frozen=True)
class InstructionSpan:
    instruction: str
    count: int


def parse_episode_range(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"(\d+)-(\d+)", value.strip())
    if match is None:
        raise ValueError(f"Invalid episode range `{value}`. Expected format like `0-49`.")

    start, end = int(match.group(1)), int(match.group(2))
    if end < start:
        raise ValueError(f"Invalid episode range `{value}`. End index must be >= start index.")

    return start, end


def parse_instruction_spans(spec: str, expected_count: int) -> List[InstructionSpan]:
    spans: List[InstructionSpan] = []
    total = 0

    for raw_chunk in [chunk.strip() for chunk in spec.split(",") if chunk.strip()]:
        if "-" not in raw_chunk:
            raise ValueError(
                f"Invalid instruction chunk `{raw_chunk}`. Expected format `<instruction>-<count>`."
            )
        instruction, count_text = raw_chunk.rsplit("-", 1)
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Instruction text cannot be empty.")
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid instruction count `{count_text}` in chunk `{raw_chunk}`."
            ) from exc
        if count <= 0:
            raise ValueError(f"Instruction count must be positive in chunk `{raw_chunk}`.")

        spans.append(InstructionSpan(instruction=instruction, count=count))
        total += count

    if total != expected_count:
        raise ValueError(
            f"Instruction counts sum to {total}, but the selected episode range contains {expected_count} episodes."
        )

    return spans


def expand_instruction_spans(spans: Sequence[InstructionSpan]) -> List[str]:
    instructions: List[str] = []
    for span in spans:
        instructions.extend([span.instruction] * span.count)
    return instructions


def list_episode_files(input_dir: Path) -> Dict[int, Path]:
    indexed_files: Dict[int, Path] = {}
    for entry in input_dir.iterdir():
        if not entry.is_file():
            continue
        match = EPISODE_RE.match(entry.name)
        if match is None:
            continue
        indexed_files[int(match.group(1))] = entry
    return indexed_files


def bgr_to_rgb_resized(image_bgr: np.ndarray, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image_bgr.shape}.")

    image_rgb = image_bgr[..., ::-1]
    pil_image = Image.fromarray(image_rgb.astype(np.uint8))
    pil_image = pil_image.resize(image_size, Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def build_gripper_state(distance: float) -> np.ndarray:
    half = float(distance) / 2.0
    return np.asarray([half, -half], dtype=np.float32)


def infer_initial_gripper_action(
    distances: np.ndarray,
    delta: int,
    threshold: float,
    mode: str,
) -> float:
    if mode == "open":
        return -1.0
    if mode == "closed":
        return 1.0

    # "auto": look for the first significant change, otherwise default to open.
    for idx in range(len(distances)):
        next_idx = min(idx + delta, len(distances) - 1)
        diff = float(distances[next_idx] - distances[idx])
        if diff < -threshold:
            return 1.0
        if diff > threshold:
            return -1.0
    return -1.0


def infer_gripper_actions(
    distances: np.ndarray,
    delta: int,
    threshold: float,
    initial_state: str,
) -> np.ndarray:
    if threshold < 0:
        raise ValueError("Gripper threshold must be non-negative.")

    actions = np.empty((len(distances),), dtype=np.float32)
    current_state = infer_initial_gripper_action(distances, delta, threshold, initial_state)

    for idx in range(len(distances)):
        next_idx = min(idx + delta, len(distances) - 1)
        diff = float(distances[next_idx] - distances[idx])

        if diff < -threshold:
            current_state = 1.0   # close
        elif diff > threshold:
            current_state = -1.0  # open

        actions[idx] = current_state

    return actions


def validate_lengths(*arrays: np.ndarray) -> int:
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched sequence lengths detected: {lengths}")
    return lengths[0]


def load_episode_steps(
    hdf5_path: Path,
    instruction: str,
    delta: int,
    gripper_threshold: float,
    initial_gripper_state: str,
) -> List[Dict[str, object]]:
    with h5py.File(hdf5_path, "r") as handle:
        eef_state = np.asarray(handle["/arm/endPose/piper_end"], dtype=np.float32)
        gripper_distance = np.asarray(handle["/gripper/encoderDistance/pika"], dtype=np.float32)
        third_person_bgr = np.asarray(handle["/camera/color/pikaDepthCamera"], dtype=np.uint8)
        wrist_bgr = np.asarray(handle["/camera/color/pikaFisheyeCamera"], dtype=np.uint8)

    if gripper_distance.ndim != 1:
        raise ValueError(
            f"Expected `/gripper/encoderDistance/pika` to have shape (T,), got {gripper_distance.shape}."
        )

    length = validate_lengths(eef_state, gripper_distance, third_person_bgr, wrist_bgr)
    if length == 0:
        raise ValueError(f"Episode `{hdf5_path}` is empty.")
    if delta <= 0:
        raise ValueError("Delta must be a positive integer.")

    gripper_actions = infer_gripper_actions(
        gripper_distance,
        delta=delta,
        threshold=gripper_threshold,
        initial_state=initial_gripper_state,
    )

    steps: List[Dict[str, object]] = []
    for idx in range(length):
        next_idx = min(idx + delta, length - 1)

        action = np.empty((7,), dtype=np.float32)
        action[:6] = eef_state[next_idx] - eef_state[idx]
        action[6] = gripper_actions[idx]

        state = np.empty((8,), dtype=np.float32)
        state[:6] = eef_state[idx]
        state[6:] = build_gripper_state(gripper_distance[idx])

        step = {
            "observation": {
                "image": bgr_to_rgb_resized(third_person_bgr[idx]),
                "wrist_image": bgr_to_rgb_resized(wrist_bgr[idx]),
                "state": state,
            },
            "action": action,
            "discount": np.float32(1.0),
            "reward": np.float32(1.0 if idx == length - 1 else 0.0),
            "is_first": bool(idx == 0),
            "is_last": bool(idx == length - 1),
            "is_terminal": bool(idx == length - 1),
            "language_instruction": instruction,
        }
        steps.append(step)

    return steps


class PikaLiberoRldsBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(DATASET_VERSION)
    RELEASE_NOTES = {
        DATASET_VERSION: "Initial conversion from Pika teleoperation HDF5 to LIBERO-style RLDS."
    }

    def __init__(self, *args, dataset_name: str, episodes: Sequence[List[Dict[str, object]]], **kwargs):
        self._dataset_name = dataset_name
        self._episodes = list(episodes)
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:  # pragma: no cover - exercised by tfds at runtime
        return self._dataset_name

    def _info(self) -> tfds.core.DatasetInfo:
        features = tfds.features.FeaturesDict(
            {
                "steps": tfds.features.Dataset(
                    tfds.features.FeaturesDict(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(shape=(256, 256, 3)),
                                    "wrist_image": tfds.features.Image(shape=(256, 256, 3)),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    )
                )
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description="Pika teleoperation dataset converted to LIBERO-style RLDS.",
            features=features,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for episode_idx, steps in enumerate(self._episodes):
            yield episode_idx, {"steps": steps}


def write_manifest(
    output_dir: Path,
    dataset_name: str,
    source_files: Sequence[Path],
    instructions: Sequence[str],
    delta: int,
    gripper_threshold: float,
    initial_gripper_state: str,
) -> None:
    manifest = {
        "dataset_name": dataset_name,
        "num_episodes": len(source_files),
        "version": DATASET_VERSION,
        "delta": delta,
        "gripper_delta_threshold": gripper_threshold,
        "initial_gripper_state": initial_gripper_state,
        "episodes": [
            {
                "source_file": str(path),
                "language_instruction": instruction,
            }
            for path, instruction in zip(source_files, instructions)
        ],
    }
    with (output_dir / "conversion_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing episode<n>.hdf5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the generated TFDS/RLDS dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pika_libero_rlds",
        help="Name of the generated dataset inside the output directory.",
    )
    parser.add_argument(
        "--episode_range",
        type=str,
        required=True,
        help="Episode range in the form `start-end`, inclusive.",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="Instruction spec in the form `<instruction>-<count>,<instruction2>-<count2>,...`.",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=1,
        help="Future index offset used to compute EEF delta actions.",
    )
    parser.add_argument(
        "--gripper_delta_threshold",
        type=float,
        default=1e-4,
        help="Absolute threshold for deciding whether the gripper changed state.",
    )
    parser.add_argument(
        "--initial_gripper_state",
        type=str,
        choices=["auto", "open", "closed"],
        default="auto",
        help="Initial binary gripper state used before any significant gripper delta is observed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    start_idx, end_idx = parse_episode_range(args.episode_range)
    episode_count = end_idx - start_idx + 1
    instruction_spans = parse_instruction_spans(args.instructions, episode_count)
    instructions = expand_instruction_spans(instruction_spans)

    available_files = list_episode_files(input_dir)
    selected_files: List[Path] = []
    for episode_idx in range(start_idx, end_idx + 1):
        if episode_idx not in available_files:
            raise FileNotFoundError(
                f"Missing required file `episode{episode_idx}.hdf5` under {input_dir}."
            )
        selected_files.append(available_files[episode_idx])

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes: List[List[Dict[str, object]]] = []
    for hdf5_path, instruction in zip(selected_files, instructions):
        print(f"Converting {hdf5_path.name} -> instruction `{instruction}`")
        steps = load_episode_steps(
            hdf5_path=hdf5_path,
            instruction=instruction,
            delta=args.delta,
            gripper_threshold=args.gripper_delta_threshold,
            initial_gripper_state=args.initial_gripper_state,
        )
        episodes.append(steps)

    builder = PikaLiberoRldsBuilder(
        data_dir=str(output_dir),
        dataset_name=args.dataset_name,
        episodes=episodes,
    )
    builder.download_and_prepare()

    write_manifest(
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        source_files=selected_files,
        instructions=instructions,
        delta=args.delta,
        gripper_threshold=args.gripper_delta_threshold,
        initial_gripper_state=args.initial_gripper_state,
    )

    print(f"Saved dataset `{args.dataset_name}` to {output_dir}")


if __name__ == "__main__":
    main()
