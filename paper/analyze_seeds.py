from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain2image.cli import evaluate_loader, make_loader, model_from_checkpoint, select_device
from brain2image.data import load_things_eeg_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/sub-01/best.pt"),
            Path("outputs/sub-01-seed-7/best.pt"),
            Path("outputs/sub-01-seed-21/best.pt"),
        ],
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("things-eeg"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper-analysis/seed-study.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = select_device(args.device)
    split = load_things_eeg_split(args.dataset_root, 1, "test")
    loader = make_loader(
        split,
        np.arange(len(split.image_paths)),
        args.batch_size,
        80,
        False,
        0,
    )
    runs = []
    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model = model_from_checkpoint(checkpoint, device)
        metrics, _, _ = evaluate_loader(
            model, loader, device, int(checkpoint.get("seed", 42))
        )
        runs.append(
            {
                "seed": int(checkpoint.get("seed", -1)),
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get(
                    "best_validation_metrics", {}
                ).get("epoch"),
                "metrics": metrics,
            }
        )
        del model
        gc.collect()

    metric_names = (
        "image_top1",
        "image_top5",
        "image_top10",
        "median_rank",
        "mean_rank",
        "mean_cosine",
    )
    aggregate = {}
    for name in metric_names:
        values = np.asarray([run["metrics"][name] for run in runs], dtype=float)
        aggregate[name] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "values": values.tolist(),
        }
    result = {
        "subject": 1,
        "split": "test",
        "repetitions_averaged": 80,
        "runs": runs,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
