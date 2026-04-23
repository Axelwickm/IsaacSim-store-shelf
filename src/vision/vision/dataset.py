import json
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
TRAIN_SPLIT_THRESHOLD = 0.75
BACKGROUND_CLASS_ID = 0
UNKNOWN_CLASS_ID = 1
DEFAULT_IMAGE_SIZE = 384


def _stable_sample_float(sample_id: str) -> float:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def _split_matches(sample_id: str, split: str, train_split_threshold: float) -> bool:
    sample_value = _stable_sample_float(sample_id)
    if split == "train":
        return sample_value < train_split_threshold
    if split == "test":
        return sample_value >= train_split_threshold
    raise ValueError(f"Unsupported split {split!r}; expected 'train' or 'test'")


def _decode_instance_segmentation_ids(segmentation: np.ndarray) -> np.ndarray:
    return (
        segmentation[..., 0].astype(np.int64)
        + 256 * segmentation[..., 1].astype(np.int64)
        + 65536 * segmentation[..., 2].astype(np.int64)
    )


def _semantic_name_to_class_id(name: str | None) -> int:
    if not name:
        return UNKNOWN_CLASS_ID
    normalized = name.strip().lower()
    if normalized in {"background", "bg"}:
        return BACKGROUND_CLASS_ID
    if normalized.startswith("item_group_"):
        return 2
    if normalized == "cart":
        return 3
    if normalized == "robot":
        return 4
    if normalized == "shelf":
        return 5
    return UNKNOWN_CLASS_ID


class StoreShelfVisionDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        train_split_threshold: float = TRAIN_SPLIT_THRESHOLD,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split = split
        self.train_split_threshold = train_split_threshold
        self.image_size = image_size
        self.sample_ids = self._discover_sample_ids()
        if not self.sample_ids:
            raise ValueError(
                f"No vision samples found under {self.dataset_dir} for split {self.split!r}"
            )

    def _discover_sample_ids(self) -> list[str]:
        sample_ids = []
        for path in sorted(self.dataset_dir.glob("rgb_*.png")):
            sample_id = path.stem.split("_")[-1]
            if self._sample_exists(sample_id) and _split_matches(
                sample_id,
                self.split,
                self.train_split_threshold,
            ):
                sample_ids.append(sample_id)
        return sample_ids

    def _sample_exists(self, sample_id: str) -> bool:
        required_paths = [
            self.dataset_dir / f"rgb_{sample_id}.png",
            self.dataset_dir / f"instance_segmentation_{sample_id}.png",
            self.dataset_dir / f"distance_to_camera_{sample_id}.npy",
            self.dataset_dir / f"instance_segmentation_mapping_{sample_id}.json",
            self.dataset_dir / f"instance_segmentation_semantics_mapping_{sample_id}.json",
        ]
        return all(path.exists() for path in required_paths)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict | str]:
        sample_id = self.sample_ids[index]

        rgb = cv2.imread(
            str(self.dataset_dir / f"rgb_{sample_id}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        segmentation = cv2.imread(
            str(self.dataset_dir / f"instance_segmentation_{sample_id}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        depth = np.load(self.dataset_dir / f"distance_to_camera_{sample_id}.npy")

        if rgb is None or segmentation is None:
            raise ValueError(f"Failed to load sample {sample_id} from {self.dataset_dir}")

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGBA)[..., :3]
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGRA2RGBA)
        depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0).astype(np.float32)
        rgb = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        segmentation = cv2.resize(
            segmentation,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = cv2.resize(
            depth,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        normalized_rgb = (rgb_tensor - RGB_MEAN) / RGB_STD

        with open(
            self.dataset_dir / f"instance_segmentation_mapping_{sample_id}.json",
            "r",
            encoding="utf-8",
        ) as handle:
            instance_mapping = json.load(handle)
        with open(
            self.dataset_dir
            / f"instance_segmentation_semantics_mapping_{sample_id}.json",
            "r",
            encoding="utf-8",
        ) as handle:
            semantics_mapping = json.load(handle)

        segmentation_ids = _decode_instance_segmentation_ids(segmentation)
        return {
            "sample_id": sample_id,
            "split": self.split,
            "rgb": rgb_tensor,
            "rgb_normalized": normalized_rgb,
            "instance_segmentation": torch.from_numpy(segmentation_ids),
            "distance_to_camera": torch.from_numpy(depth).unsqueeze(0),
            "instance_mapping": instance_mapping,
            "semantics_mapping": semantics_mapping,
        }


def collate_vision_samples(batch: list[dict]) -> dict:
    return {
        "sample_id": [sample["sample_id"] for sample in batch],
        "split": [sample["split"] for sample in batch],
        "rgb": torch.stack([sample["rgb"] for sample in batch]),
        "rgb_normalized": torch.stack([sample["rgb_normalized"] for sample in batch]),
        "instance_segmentation": torch.stack(
            [sample["instance_segmentation"] for sample in batch]
        ),
        "distance_to_camera": torch.stack(
            [sample["distance_to_camera"] for sample in batch]
        ),
        "instance_mapping": [sample["instance_mapping"] for sample in batch],
        "semantics_mapping": [sample["semantics_mapping"] for sample in batch],
    }


def build_mask2former_batch_targets(batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mask_labels = []
    class_labels = []

    for instance_segmentation, semantics_mapping in zip(
        batch["instance_segmentation"],
        batch["semantics_mapping"],
    ):
        sample_masks = []
        sample_classes = []
        raw_labels = semantics_mapping.get("idToLabels", semantics_mapping)
        for raw_instance_id, labels in raw_labels.items():
            semantic_name = None
            if isinstance(labels, dict):
                semantic_name = labels.get("class") or labels.get("label")
            elif isinstance(labels, list) and labels:
                semantic_name = labels[0]
            elif isinstance(labels, str):
                semantic_name = labels

            raw_numbers = re.findall(r"\d+", str(raw_instance_id)) + ["0", "0", "0"]
            instance_id = (
                int(raw_numbers[0])
                + 256 * int(raw_numbers[1])
                + 65536 * int(raw_numbers[2])
            )
            instance_mask = instance_segmentation == instance_id
            if torch.any(instance_mask):
                sample_masks.append(instance_mask.to(torch.float32))
                sample_classes.append(_semantic_name_to_class_id(semantic_name))

        if not sample_masks:
            sample_masks.append(
                torch.zeros_like(instance_segmentation, dtype=torch.float32)
            )
            sample_classes.append(BACKGROUND_CLASS_ID)

        mask_labels.append(torch.stack(sample_masks))
        class_labels.append(torch.tensor(sample_classes, dtype=torch.int64))

    return mask_labels, class_labels
