from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ThingsEEGSplit:
    eeg: np.ndarray
    features: torch.Tensor
    labels: np.ndarray
    image_paths: np.ndarray
    texts: np.ndarray

    def __post_init__(self) -> None:
        size = self.eeg.shape[0]
        if not all(
            len(value) == size
            for value in (self.features, self.labels, self.image_paths, self.texts)
        ):
            raise ValueError("EEG data and metadata have inconsistent sample counts")
        if self.eeg.ndim != 4:
            raise ValueError(
                f"Expected EEG with shape [images, repetitions, channels, time], got {self.eeg.shape}"
            )
        if self.features.ndim != 2:
            raise ValueError(
                f"Expected image features with shape [images, dimensions], got {self.features.shape}"
            )


@dataclass(frozen=True)
class VisualSplit:
    features: torch.Tensor
    labels: np.ndarray
    image_paths: np.ndarray
    texts: np.ndarray


@dataclass(frozen=True)
class RawEEGSplit:
    eeg: np.ndarray
    labels: np.ndarray
    image_paths: np.ndarray
    texts: np.ndarray


def _first_repetition(values: Any) -> np.ndarray:
    array = np.asarray(values)
    return array[:, 0] if array.ndim > 1 else array


def load_things_eeg_split(
    dataset_root: Path, subject: int, split: str
) -> ThingsEEGSplit:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    preprocessed = dataset_root / "Preprocessed_data_250Hz_whiten"
    eeg_path = preprocessed / f"sub-{subject:02d}" / f"{split}.pt"
    feature_path = preprocessed / f"ViT-B-32_features_{split}.pt"
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEG data not found: {eeg_path}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Image features not found: {feature_path}")

    eeg_data = torch.load(eeg_path, map_location="cpu", weights_only=False)
    feature_data = torch.load(feature_path, map_location="cpu", weights_only=False)
    required = {"eeg", "label", "img", "text"}
    missing = required.difference(eeg_data)
    if missing:
        raise ValueError(f"{eeg_path} is missing keys: {sorted(missing)}")
    if "img_features" not in feature_data:
        raise ValueError(f"{feature_path} does not contain img_features")

    image_paths = _first_repetition(eeg_data["img"]).astype(str)
    labels = _first_repetition(eeg_data["label"]).astype(np.int64)
    texts = _first_repetition(eeg_data["text"]).astype(str)
    feature_map = feature_data["img_features"]

    missing_features = [path for path in image_paths if path not in feature_map]
    if missing_features:
        examples = ", ".join(missing_features[:3])
        raise ValueError(
            f"{len(missing_features)} EEG images have no matching ViT feature; examples: {examples}"
        )

    features = torch.stack(
        [torch.as_tensor(feature_map[path], dtype=torch.float32) for path in image_paths]
    )
    return ThingsEEGSplit(
        eeg=np.asarray(eeg_data["eeg"]),
        features=torch.nn.functional.normalize(features, dim=1),
        labels=labels,
        image_paths=image_paths,
        texts=texts,
    )


def load_raw_eeg_split(
    dataset_root: Path, subject: int, split: str
) -> RawEEGSplit:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    eeg_path = (
        dataset_root
        / "Preprocessed_data_250Hz_whiten"
        / f"sub-{subject:02d}"
        / f"{split}.pt"
    )
    if not eeg_path.is_file():
        raise FileNotFoundError(f"EEG data not found: {eeg_path}")
    eeg_data = torch.load(eeg_path, map_location="cpu", weights_only=False)
    required = {"eeg", "label", "img", "text"}
    missing = required.difference(eeg_data)
    if missing:
        raise ValueError(f"{eeg_path} is missing keys: {sorted(missing)}")
    return RawEEGSplit(
        eeg=np.asarray(eeg_data["eeg"]),
        labels=_first_repetition(eeg_data["label"]).astype(np.int64),
        image_paths=_first_repetition(eeg_data["img"]).astype(str),
        texts=_first_repetition(eeg_data["text"]).astype(str),
    )


def load_visual_split(dataset_root: Path, split: str = "train") -> VisualSplit:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    feature_path = (
        dataset_root
        / "Preprocessed_data_250Hz_whiten"
        / f"ViT-B-32_features_{split}.pt"
    )
    if not feature_path.is_file():
        raise FileNotFoundError(f"Image features not found: {feature_path}")
    feature_data = torch.load(feature_path, map_location="cpu", weights_only=False)
    if "img_features" not in feature_data:
        raise ValueError(f"{feature_path} does not contain img_features")

    feature_map = feature_data["img_features"]
    image_paths = np.asarray([str(path) for path in feature_map])
    parent_names = [Path(path).parent.name for path in image_paths]
    try:
        labels = np.asarray(
            [int(parent.split("_", 1)[0]) - 1 for parent in parent_names],
            dtype=np.int64,
        )
    except (ValueError, IndexError) as error:
        raise ValueError("Image directories must start with a numeric concept ID") from error
    texts = np.asarray(
        [parent.split("_", 1)[1] for parent in parent_names], dtype=str
    )
    features = torch.stack(
        [torch.as_tensor(feature_map[path], dtype=torch.float32) for path in feature_map]
    )
    return VisualSplit(
        features=torch.nn.functional.normalize(features, dim=1),
        labels=labels,
        image_paths=image_paths,
        texts=texts,
    )


def concept_split_indices(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    concepts = np.unique(labels)
    if len(concepts) < 2:
        raise ValueError("At least two concepts are required for a train/validation split")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(concepts)
    validation_count = max(1, int(round(len(concepts) * validation_fraction)))
    validation_labels = shuffled[:validation_count]
    validation_mask = np.isin(labels, validation_labels)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def limit_indices(indices: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False))


class AveragedEEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]):
    def __init__(
        self,
        split: ThingsEEGSplit,
        indices: np.ndarray | None = None,
        repetitions: int | None = None,
    ) -> None:
        self.split = split
        self.indices = (
            np.arange(split.eeg.shape[0], dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        available = split.eeg.shape[1]
        if repetitions is not None and not 1 <= repetitions <= available:
            raise ValueError(
                f"repetitions must be between 1 and {available}, got {repetitions}"
            )
        self.repetitions = repetitions or available

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        index = int(self.indices[item])
        trials = np.asarray(
            self.split.eeg[index, : self.repetitions], dtype=np.float32
        )
        eeg = torch.from_numpy(trials.mean(axis=0))
        return (
            eeg,
            self.split.features[index],
            int(self.split.labels[index]),
            index,
        )


class ImageFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        split: VisualSplit,
        dataset_root: Path,
        indices: np.ndarray,
        image_size: int = 64,
    ) -> None:
        self.split = split
        self.dataset_root = dataset_root
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        path = resolve_image_path(
            self.dataset_root, str(self.split.image_paths[index])
        )
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = torch.from_numpy(pixels.copy()).permute(2, 0, 1)
        return self.split.features[index], pixels


class EEGImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]):
    def __init__(
        self,
        split: RawEEGSplit,
        dataset_root: Path,
        indices: np.ndarray,
        repetitions: int | None = None,
        image_size: int = 64,
        cache_images: bool = False,
        shared_image_cache: np.ndarray | None = None,
    ) -> None:
        self.split = split
        self.dataset_root = dataset_root
        self.indices = np.asarray(indices, dtype=np.int64)
        available = split.eeg.shape[1]
        if repetitions is not None and not 1 <= repetitions <= available:
            raise ValueError(
                f"repetitions must be between 1 and {available}, got {repetitions}"
            )
        self.repetitions = repetitions or available
        self.image_size = image_size
        self.image_cache = shared_image_cache
        self.cache_uses_global_indices = shared_image_cache is not None
        if shared_image_cache is not None and len(shared_image_cache) != len(
            split.image_paths
        ):
            raise ValueError("Shared image cache must contain every image in the split")
        if cache_images and shared_image_cache is None:
            self.image_cache = np.empty(
                (len(self.indices), 3, image_size, image_size), dtype=np.uint8
            )
            self.cache_uses_global_indices = False
            for item, index in enumerate(self.indices):
                self.image_cache[item] = self._load_image(int(index))

    def _load_image(self, index: int) -> np.ndarray:
        path = resolve_image_path(
            self.dataset_root, str(self.split.image_paths[index])
        )
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        index = int(self.indices[item])
        trials = np.asarray(
            self.split.eeg[index, : self.repetitions], dtype=np.float32
        )
        eeg = torch.from_numpy(trials.mean(axis=0))
        pixels = (
            self.image_cache[index if self.cache_uses_global_indices else item]
            if self.image_cache is not None
            else self._load_image(index)
        )
        target = torch.from_numpy(pixels.astype(np.float32) / 127.5 - 1.0)
        return eeg, target, int(self.split.labels[index]), index


def build_image_cache(
    split: RawEEGSplit, dataset_root: Path, image_size: int = 64
) -> np.ndarray:
    cache = np.empty(
        (len(split.image_paths), 3, image_size, image_size), dtype=np.uint8
    )
    for index, relative_path in enumerate(split.image_paths):
        path = resolve_image_path(dataset_root, str(relative_path))
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            )
            cache[index] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
    return cache


def resolve_image_path(dataset_root: Path, relative_path: str) -> Path:
    direct = dataset_root / "Image_set" / relative_path
    if direct.is_file():
        return direct
    if relative_path.startswith("train_images/"):
        fallback = dataset_root / "Image_set" / relative_path.replace(
            "train_images/", "training_images/", 1
        )
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"Stimulus image not found: {relative_path}")
