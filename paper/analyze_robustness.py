from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain2image.cli import evaluate_loader, make_loader, model_from_checkpoint, select_device
from brain2image.data import load_things_eeg_split


class BalancedSessionDataset(Dataset):
    def __init__(self, split, repetitions_per_session: int) -> None:
        if not 1 <= repetitions_per_session <= 20:
            raise ValueError("repetitions_per_session must be between 1 and 20")
        self.split = split
        self.repetition_indices = np.concatenate(
            [
                np.arange(session * 20, session * 20 + repetitions_per_session)
                for session in range(4)
            ]
        )

    def __len__(self) -> int:
        return len(self.split.image_paths)

    def __getitem__(self, index: int):
        trials = np.asarray(
            self.split.eeg[index, self.repetition_indices], dtype=np.float32
        )
        return (
            torch.from_numpy(trials.mean(axis=0)),
            self.split.features[index],
            int(self.split.labels[index]),
            index,
        )


def evaluate(
    checkpoint: dict,
    dataset_root: Path,
    subject: int,
    repetitions: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    split = load_things_eeg_split(dataset_root, subject, "test")
    loader = make_loader(
        split,
        np.arange(len(split.image_paths)),
        batch_size,
        repetitions,
        False,
        0,
    )
    model = model_from_checkpoint(checkpoint, device)
    metrics, _, _ = evaluate_loader(
        model, loader, device, int(checkpoint.get("seed", 42))
    )
    del model, loader, split
    gc.collect()
    return metrics


def evaluate_balanced_repetitions(
    checkpoint: dict,
    dataset_root: Path,
    repetitions_per_session: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    split = load_things_eeg_split(dataset_root, 1, "test")
    loader = DataLoader(
        BalancedSessionDataset(split, repetitions_per_session),
        batch_size=batch_size,
        shuffle=False,
    )
    model = model_from_checkpoint(checkpoint, device)
    metrics, _, _ = evaluate_loader(
        model, loader, device, int(checkpoint.get("seed", 42))
    )
    del model, loader, split
    gc.collect()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/sub-01/best.pt")
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("things-eeg"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper-analysis/robustness.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    repetitions_per_session = (1, 2, 5, 10, 20)
    repetition_ablation = {
        str(per_session * 4): evaluate_balanced_repetitions(
            checkpoint,
            args.dataset_root,
            per_session,
            device,
            args.batch_size,
        )
        for per_session in repetitions_per_session
    }
    cross_subject = {
        str(subject): evaluate(
            checkpoint,
            args.dataset_root,
            subject,
            80,
            device,
            args.batch_size,
        )
        for subject in range(1, 11)
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_seed": checkpoint.get("seed"),
        "repetition_ablation_subject": 1,
        "repetition_ablation_sessions": 4,
        "repetition_selection": "equal number of earliest repetitions per session",
        "repetition_ablation": repetition_ablation,
        "cross_subject_training_subject": 1,
        "cross_subject_repetitions": 80,
        "cross_subject": cross_subject,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
