from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EEGEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 63,
        time_points: int = 250,
        embedding_dim: int = 512,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.time_points = time_points
        self.embedding_dim = embedding_dim

        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(32),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=(channels, 1),
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=64,
                bias=False,
            ),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        pooled_time = (time_points // 4) // 4
        if pooled_time < 1:
            raise ValueError("time_points must be at least 16")
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pooled_time, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(
                f"Expected EEG with shape [batch, channels, time], got {tuple(eeg.shape)}"
            )
        if eeg.shape[1:] != (self.channels, self.time_points):
            raise ValueError(
                f"Expected {self.channels} channels and {self.time_points} time points, "
                f"got {tuple(eeg.shape[1:])}"
            )
        features = self.temporal(eeg.unsqueeze(1))
        features = self.spatial(features)
        features = self.separable(features)
        return F.normalize(self.projection(features), dim=1)


class MultiSubjectEEGEncoder(nn.Module):
    def __init__(
        self,
        subjects: int,
        channels: int = 63,
        time_points: int = 250,
        embedding_dim: int = 256,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if subjects < 1:
            raise ValueError("At least one subject is required")
        self.base = EEGEncoder(channels, time_points, embedding_dim, dropout)
        self.subject_embedding = nn.Embedding(subjects, embedding_dim)
        nn.init.normal_(self.subject_embedding.weight, std=0.02)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, eeg: torch.Tensor, subject_indices: torch.Tensor
    ) -> torch.Tensor:
        shared = self.base(eeg)
        subject_offset = self.subject_embedding(subject_indices)
        return F.normalize(self.output_norm(shared + subject_offset), dim=1)


class ImageDecoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, image_size: int = 64) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ImageDecoder currently supports 64x64 output")
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, 512 * 4 * 4),
            nn.GELU(),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings with shape [batch, {self.embedding_dim}], "
                f"got {tuple(embedding.shape)}"
            )
        features = self.project(embedding).reshape(-1, 512, 4, 4)
        return self.decode(features)


class ScratchGenerator(nn.Module):
    def __init__(
        self, condition_dim: int = 256, noise_dim: int = 64, image_size: int = 64
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ScratchGenerator currently supports 64x64 output")
        self.condition_dim = condition_dim
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.project = nn.Sequential(
            nn.Linear(condition_dim + noise_dim, 256 * 4 * 4),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(
        self, condition: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.zeros(
                len(condition),
                self.noise_dim,
                device=condition.device,
                dtype=condition.dtype,
            )
        if noise.shape != (len(condition), self.noise_dim):
            raise ValueError(
                f"Expected noise with shape {(len(condition), self.noise_dim)}, "
                f"got {tuple(noise.shape)}"
            )
        projected = self.project(torch.cat((condition, noise), dim=1))
        projected = projected.reshape(-1, 256, 4, 4)
        return self.blocks(projected)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, condition_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.unconditional = nn.Linear(256, 1)
        self.condition_projection = nn.Linear(condition_dim, 256)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = self.features(image).mean(dim=(2, 3))
        unconditional = self.unconditional(features).squeeze(1)
        projected = self.condition_projection(condition)
        conditional = (features * projected).sum(dim=1) / features.shape[1] ** 0.5
        return unconditional + conditional


def discriminator_hinge_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    mismatched_scores: torch.Tensor,
) -> torch.Tensor:
    return (
        F.relu(1 - real_scores).mean()
        + 0.5 * F.relu(1 + fake_scores).mean()
        + 0.5 * F.relu(1 + mismatched_scores).mean()
    )


def multiscale_reconstruction_loss(
    generated: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if generated.shape != target.shape:
        raise ValueError(
            f"Generated and target images must match, got {generated.shape} and {target.shape}"
        )
    loss = 0.7 * F.l1_loss(generated, target)
    for weight, scale in ((0.2, 2), (0.1, 4)):
        generated_scaled = F.avg_pool2d(generated, scale)
        target_scaled = F.avg_pool2d(target, scale)
        loss = loss + weight * F.l1_loss(generated_scaled, target_scaled)
    return loss


def contrastive_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 0.07,
    cosine_weight: float = 0.25,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(
            f"Predicted and target embeddings must match, got {predicted.shape} and {target.shape}"
        )
    target = F.normalize(target, dim=1)
    logits = predicted @ target.T / temperature
    identity = torch.arange(len(predicted), device=predicted.device)
    if labels is not None:
        labels = labels.to(predicted.device)
        false_negatives = labels[:, None].eq(labels[None, :])
        false_negatives.fill_diagonal_(False)
        logits = logits.masked_fill(false_negatives, torch.finfo(logits.dtype).min)
    contrastive = (
        F.cross_entropy(logits, identity) + F.cross_entropy(logits.T, identity)
    ) / 2
    cosine = 1 - (predicted * target).sum(dim=1).mean()
    return contrastive + cosine_weight * cosine
