from __future__ import annotations

import torch
from torch.nn import functional as F


def _concept_chance(labels: torch.Tensor, k: int) -> float:
    total = len(labels)
    _, counts = torch.unique(labels, return_counts=True)
    probabilities = []
    for count in counts.tolist():
        miss = 1.0
        for draw in range(k):
            miss *= max(total - count - draw, 0) / (total - draw)
        probabilities.extend([1.0 - miss] * count)
    return float(torch.tensor(probabilities).mean())


def retrieval_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target embeddings must have identical shapes")
    if len(labels) != len(predicted):
        raise ValueError("A label is required for every embedding")

    predicted = F.normalize(predicted.float(), dim=1)
    target = F.normalize(target.float(), dim=1)
    similarities = predicted @ target.T
    diagonal = similarities.diagonal()
    ranks = 1 + (similarities > diagonal[:, None]).sum(dim=1)
    ranking = similarities.argsort(dim=1, descending=True)

    result = {
        "samples": float(len(predicted)),
        "mean_cosine": float(diagonal.mean()),
        "median_rank": float(ranks.float().median()),
        "mean_rank": float(ranks.float().mean()),
    }
    for k in topk:
        if k > len(predicted):
            continue
        exact = (ranks <= k).float().mean()
        retrieved_labels = labels[ranking[:, :k]]
        concept = (retrieved_labels == labels[:, None]).any(dim=1).float().mean()
        result[f"image_top{k}"] = float(exact)
        result[f"concept_top{k}"] = float(concept)
        result[f"chance_image_top{k}"] = k / len(predicted)
        result[f"chance_concept_top{k}"] = _concept_chance(labels, k)

    return result


def shuffled_baseline(
    predicted: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    if len(predicted) < 2:
        return {}
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(predicted), generator=generator)
    baseline = retrieval_metrics(predicted[permutation], target, labels, topk=topk)
    return {
        f"shuffled_{key}": value
        for key, value in baseline.items()
        if key.startswith(("image_top", "concept_top"))
    }
