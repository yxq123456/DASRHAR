from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from dasr_har.outages.generator import OnlineOutageGenerator, derive_events


def load_split(path: str | Path, fold: int) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        folds = yaml.safe_load(handle)["folds"]
    value = folds.get(fold, folds.get(str(fold)))
    return {key: [str(item) for item in items] for key, items in value.items()}


def fit_normalizer(path: str | Path, train_subjects: list[str]) -> dict[str, dict[str, list[float]]]:
    result = {}
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        selected = np.isin(data["subject_id"], np.asarray(train_subjects))
        for stream in metadata["streams"]:
            values = data[f"x__{stream}"][selected]
            finite = np.isfinite(values)
            count = finite.sum(axis=(0, 1)).clip(min=1)
            total = np.where(finite, values, 0.0).sum(axis=(0, 1))
            mean = total / count
            centered = np.where(finite, values - mean, 0.0)
            std = np.sqrt((centered * centered).sum(axis=(0, 1)) / count).clip(min=1e-6)
            result[stream] = {"mean": mean.tolist(), "std": std.tolist()}
    return result


class WindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: str | Path,
        subjects: list[str],
        normalizer: dict[str, dict[str, list[float]]],
        sample_rate: float,
        outage_transform: Callable[[str, int], dict[str, np.ndarray]] | None = None,
    ):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)
        self.metadata = json.loads(str(self.data["metadata"]))
        self.streams = list(self.metadata["streams"])
        self.indices = np.flatnonzero(np.isin(self.data["subject_id"], np.asarray(subjects)))
        self.normalizer = normalizer
        self.sample_rate = float(sample_rate)
        self.outage_transform = outage_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self.indices[index])
        sample_id = str(self.data["sample_id"][source_index])
        x = {}
        native = {}
        length = 0
        for stream in self.streams:
            values = self.data[f"x__{stream}"][source_index].astype(np.float32)
            mean = np.asarray(self.normalizer[stream]["mean"], dtype=np.float32)
            std = np.asarray(self.normalizer[stream]["std"], dtype=np.float32)
            values = np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
            x[stream] = values
            native[stream] = self.data[f"native__{stream}"][source_index].astype(bool)
            length = len(values)
        synthetic = (
            self.outage_transform(sample_id, length)
            if self.outage_transform is not None
            else {stream: np.ones(length, dtype=bool) for stream in self.streams}
        )
        available = {stream: native[stream] & synthetic[stream] for stream in self.streams}
        gap_age, reentry, duration = derive_events(synthetic, 1.0 / self.sample_rate)
        for stream in self.streams:
            reentry[stream] &= available[stream]
            duration[stream] *= reentry[stream]
        return {
            "x": x,
            "native_mask": native,
            "synthetic_mask": synthetic,
            "available": available,
            "gap_age": gap_age,
            "reentry": reentry,
            "duration": duration,
            "label": int(self.data["label"][source_index]),
            "label_sequence": self.data["label_sequence"][source_index].astype(np.int64),
            "subject_id": str(self.data["subject_id"][source_index]),
            "sample_id": sample_id,
            "session_id": str(self.data["session_id"][source_index]),
            "start_index": int(self.data["start_index"][source_index]),
        }

    def close(self) -> None:
        self.data.close()


def collate_windows(items: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_groups = [
        "x",
        "native_mask",
        "synthetic_mask",
        "available",
        "gap_age",
        "reentry",
        "duration",
    ]
    batch: dict[str, Any] = {}
    for group in tensor_groups:
        batch[group] = {
            stream: torch.as_tensor(np.stack([item[group][stream] for item in items]))
            for stream in items[0][group]
        }
    batch["label"] = torch.as_tensor([item["label"] for item in items], dtype=torch.long)
    batch["label_sequence"] = torch.as_tensor(
        np.stack([item["label_sequence"] for item in items]), dtype=torch.long
    )
    for key in ["subject_id", "sample_id", "session_id", "start_index"]:
        batch[key] = [item[key] for item in items]
    return batch


def build_dataloaders(
    dataset_config: dict[str, Any],
    fold: int,
    batch_size: int,
    seed: int,
    workers: int = 0,
    train_outages: bool = True,
    manifest_transforms: dict[str, Callable[[str, int], dict[str, np.ndarray]]] | None = None,
) -> tuple[dict[str, DataLoader], dict[str, dict[str, list[float]]]]:
    split = load_split(dataset_config["split_path"], fold)
    normalizer = fit_normalizer(dataset_config["processed_path"], split["train"])
    train_transform = None
    if train_outages:
        train_transform = OnlineOutageGenerator(
            list(dataset_config["streams"]), seed=seed, clean_probability=0.4
        )
    manifest_transforms = manifest_transforms or {}
    datasets = {
        name: WindowDataset(
            dataset_config["processed_path"],
            subjects,
            normalizer,
            dataset_config["sample_rate"],
            train_transform if name == "train" else manifest_transforms.get(name),
        )
        for name, subjects in split.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_windows,
            generator=generator if name == "train" else None,
        )
        for name, dataset in datasets.items()
    }
    return loaders, normalizer
