from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (
    AveragedEEGDataset,
    EEGImageDataset,
    ImageFeatureDataset,
    ThingsEEGSplit,
    build_image_cache,
    concept_split_indices,
    limit_indices,
    load_things_eeg_split,
    load_raw_eeg_split,
    load_visual_split,
    resolve_image_path,
)
from .metrics import retrieval_metrics, shuffled_baseline
from .model import (
    ConditionalDiscriminator,
    EEGEncoder,
    ImageDecoder,
    MultiSubjectEEGEncoder,
    ScratchGenerator,
    contrastive_alignment_loss,
    discriminator_hinge_loss,
    multiscale_reconstruction_loss,
)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    split: ThingsEEGSplit,
    indices: np.ndarray,
    batch_size: int,
    repetitions: int | None,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    dataset = AveragedEEGDataset(split, indices=indices, repetitions=repetitions)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) > batch_size and len(dataset) % batch_size == 1,
    )


def collect_predictions(
    model: EEGEncoder, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for eeg, target, label, index in loader:
            predictions.append(model(eeg.to(device)).cpu())
            targets.append(F.normalize(target.float(), dim=1))
            labels.append(label.long())
            indices.append(index.long())
    return tuple(
        torch.cat(parts)
        for parts in (predictions, targets, labels, indices)
    )  # type: ignore[return-value]


def evaluate_loader(
    model: EEGEncoder, loader: DataLoader, device: torch.device, seed: int
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    predicted, target, labels, indices = collect_predictions(model, loader, device)
    metrics = retrieval_metrics(predicted, target, labels)
    metrics.update(shuffled_baseline(predicted, target, labels, seed))
    return metrics, predicted, indices


def format_metrics(metrics: dict[str, float]) -> str:
    percentages = {
        key
        for key in metrics
        if "top" in key or key.startswith("chance_") or key.startswith("shuffled_")
    }
    parts = []
    for key, value in metrics.items():
        rendered = f"{value * 100:.2f}%" if key in percentages else f"{value:.4f}"
        parts.append(f"{key}={rendered}")
    return " | ".join(parts)


def model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> EEGEncoder:
    config = checkpoint["model_config"]
    model = EEGEncoder(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def command_inspect(args: argparse.Namespace) -> None:
    split = load_things_eeg_split(args.dataset_root, args.subject, args.split)
    print(f"subject=sub-{args.subject:02d} split={args.split}")
    print(
        f"eeg={split.eeg.shape} dtype={split.eeg.dtype} "
        f"features={tuple(split.features.shape)}"
    )
    print(
        f"images={len(split.image_paths)} concepts={len(np.unique(split.labels))} "
        f"repetitions={split.eeg.shape[1]}"
    )


def command_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    split = load_things_eeg_split(args.dataset_root, args.subject, "train")
    train_indices, validation_indices = concept_split_indices(
        split.labels, args.validation_fraction, args.seed
    )
    train_indices = limit_indices(train_indices, args.max_train_images, args.seed)
    validation_indices = limit_indices(
        validation_indices, args.max_validation_images, args.seed + 1
    )
    if len(train_indices) < 2 or len(validation_indices) < 2:
        raise ValueError("Training and validation each require at least two images")
    train_loader = make_loader(
        split,
        train_indices,
        args.batch_size,
        args.repetitions,
        True,
        args.workers,
    )
    validation_loader = make_loader(
        split,
        validation_indices,
        args.batch_size,
        args.repetitions,
        False,
        args.workers,
    )

    channels, time_points = split.eeg.shape[-2:]
    model_config = {
        "channels": channels,
        "time_points": time_points,
        "embedding_dim": split.features.shape[1],
        "dropout": args.dropout,
    }
    model = EEGEncoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(
        f"device={device} train_images={len(train_indices)} "
        f"validation_images={len(validation_indices)}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    best_score = -1.0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for eeg, target, labels, _ in train_loader:
            eeg = eeg.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(eeg)
            loss = contrastive_alignment_loss(
                predicted,
                target,
                labels=labels,
                temperature=args.temperature,
                cosine_weight=args.cosine_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running_loss += loss.detach().item() * len(eeg)
            seen += len(eeg)

        metrics, _, _ = evaluate_loader(
            model, validation_loader, device, args.seed + epoch
        )
        train_loss = running_loss / seen
        metrics["train_loss"] = train_loss
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        print(f"epoch={epoch} train_loss={train_loss:.4f} | {format_metrics(metrics)}")

        score = metrics.get("concept_top5", metrics["concept_top1"])
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "subject": args.subject,
                    "seed": args.seed,
                    "validation_fraction": args.validation_fraction,
                    "repetitions": args.repetitions,
                    "best_validation_metrics": metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stopping epoch={epoch}")
                break

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint_path}")


def command_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    subject = args.subject or int(checkpoint["subject"])
    split = load_things_eeg_split(args.dataset_root, subject, args.split)
    indices = np.arange(len(split.image_paths))
    indices = limit_indices(indices, args.max_images, args.seed)
    loader = make_loader(
        split, indices, args.batch_size, args.repetitions, False, args.workers
    )
    model = model_from_checkpoint(checkpoint, device)
    metrics, _, _ = evaluate_loader(model, loader, device, args.seed)
    output = {
        "subject": subject,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "metrics": metrics,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def build_contact_sheet(
    split: ThingsEEGSplit,
    dataset_root: Path,
    query_indices: torch.Tensor,
    retrieved_indices: torch.Tensor,
    output: Path,
) -> None:
    width, image_size, label_height = 2, 192, 28
    rows = len(query_indices)
    canvas = Image.new(
        "RGB", (width * image_size, rows * (image_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for row, (query, retrieved) in enumerate(
        zip(query_indices.tolist(), retrieved_indices.tolist())
    ):
        for column, (index, title) in enumerate(
            ((query, "Target"), (retrieved, "EEG retrieval"))
        ):
            image = Image.open(
                resolve_image_path(dataset_root, str(split.image_paths[index]))
            ).convert("RGB")
            image.thumbnail((image_size, image_size))
            x = column * image_size + (image_size - image.width) // 2
            y = row * (image_size + label_height) + label_height
            canvas.paste(image, (x, y))
            draw.text(
                (column * image_size + 4, row * (image_size + label_height) + 5),
                f"{title}: {split.texts[index]}",
                fill="black",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def command_reconstruct(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    subject = args.subject or int(checkpoint["subject"])
    split = load_things_eeg_split(args.dataset_root, subject, args.split)
    count = min(args.count, len(split.image_paths))
    generator = np.random.default_rng(args.seed)
    indices = np.sort(generator.choice(len(split.image_paths), count, replace=False))
    loader = make_loader(
        split, indices, args.batch_size, args.repetitions, False, args.workers
    )
    model = model_from_checkpoint(checkpoint, device)
    predicted, _, _, query_indices = collect_predictions(model, loader, device)
    similarities = predicted @ split.features.T
    retrieved_indices = similarities.argmax(dim=1)
    build_contact_sheet(
        split,
        args.dataset_root,
        query_indices,
        retrieved_indices,
        args.output,
    )
    records = [
        {
            "target": str(split.image_paths[query]),
            "retrieved": str(split.image_paths[retrieved]),
            "target_concept": str(split.texts[query]),
            "retrieved_concept": str(split.texts[retrieved]),
            "cosine_similarity": float(similarities[row, retrieved]),
        }
        for row, (query, retrieved) in enumerate(
            zip(query_indices.tolist(), retrieved_indices.tolist())
        )
    ]
    args.output.with_suffix(".json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"contact_sheet={args.output}")


def command_train_decoder(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    split = load_visual_split(args.dataset_root, "train")
    train_indices, validation_indices = concept_split_indices(
        split.labels, args.validation_fraction, args.seed
    )
    train_indices = limit_indices(train_indices, args.max_train_images, args.seed)
    validation_indices = limit_indices(
        validation_indices, args.max_validation_images, args.seed + 1
    )
    train_dataset = ImageFeatureDataset(
        split, args.dataset_root, train_indices, args.image_size
    )
    validation_dataset = ImageFeatureDataset(
        split, args.dataset_root, validation_indices, args.image_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) > args.batch_size
        and len(train_dataset) % args.batch_size == 1,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    model_config = {
        "embedding_dim": split.features.shape[1],
        "image_size": args.image_size,
    }
    model = ImageDecoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale_epochs = 0
    print(
        f"device={device} train_images={len(train_dataset)} "
        f"validation_images={len(validation_dataset)}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_seen = 0
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            generated = model(features)
            loss = multiscale_reconstruction_loss(generated, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_total += loss.detach().item() * len(features)
            train_seen += len(features)

        model.eval()
        validation_total = 0.0
        validation_seen = 0
        with torch.inference_mode():
            for features, target in validation_loader:
                features, target = features.to(device), target.to(device)
                loss = multiscale_reconstruction_loss(model(features), target)
                validation_total += loss.item() * len(features)
                validation_seen += len(features)
        train_loss = train_total / train_seen
        validation_loss = validation_total / validation_seen
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "seed": args.seed,
                    "validation_fraction": args.validation_fraction,
                    "best_validation_loss": best_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stopping epoch={epoch}")
                break

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint_path}")


def build_generated_sheet(
    split: ThingsEEGSplit,
    dataset_root: Path,
    query_indices: torch.Tensor,
    generated: torch.Tensor,
    output: Path,
) -> None:
    image_size, display_size, label_height = 64, 256, 28
    canvas = Image.new(
        "RGB",
        (display_size * 2, len(query_indices) * (display_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, query in enumerate(query_indices.tolist()):
        target = Image.open(
            resolve_image_path(dataset_root, str(split.image_paths[query]))
        ).convert("RGB")
        target = target.resize((display_size, display_size), Image.Resampling.LANCZOS)
        pixels = (
            generated[row]
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        synthesized = Image.fromarray(pixels).resize(
            (display_size, display_size), Image.Resampling.NEAREST
        )
        y = row * (display_size + label_height)
        canvas.paste(target, (0, y + label_height))
        canvas.paste(synthesized, (display_size, y + label_height))
        draw.text((4, y + 5), f"Target: {split.texts[query]}", fill="black")
        draw.text((display_size + 4, y + 5), "EEG generated", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def command_generate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    eeg_checkpoint = torch.load(
        args.eeg_checkpoint, map_location="cpu", weights_only=False
    )
    decoder_checkpoint = torch.load(
        args.decoder_checkpoint, map_location="cpu", weights_only=False
    )
    subject = args.subject or int(eeg_checkpoint["subject"])
    split = load_things_eeg_split(args.dataset_root, subject, args.split)
    count = min(args.count, len(split.image_paths))
    generator = np.random.default_rng(args.seed)
    indices = np.sort(generator.choice(len(split.image_paths), count, replace=False))
    loader = make_loader(
        split, indices, args.batch_size, args.repetitions, False, args.workers
    )
    eeg_model = model_from_checkpoint(eeg_checkpoint, device)
    decoder = ImageDecoder(**decoder_checkpoint["model_config"]).to(device)
    decoder.load_state_dict(decoder_checkpoint["model_state"])
    decoder.eval()
    predicted, _, _, query_indices = collect_predictions(eeg_model, loader, device)
    with torch.inference_mode():
        generated = decoder(predicted.to(device)).cpu()
    build_generated_sheet(
        split, args.dataset_root, query_indices, generated, args.output
    )
    torch.save(
        {
            "query_indices": query_indices,
            "image_paths": [str(split.image_paths[i]) for i in query_indices.tolist()],
            "generated": generated,
        },
        args.output.with_suffix(".pt"),
    )
    print(f"generated_sheet={args.output}")


def command_generate_diffusion(args: argparse.Namespace) -> None:
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as error:
        raise RuntimeError(
            "Diffusion dependencies are missing; run: "
            "python3 -m pip install -r requirements-diffusion.txt"
        ) from error

    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(
        args.eeg_checkpoint, map_location="cpu", weights_only=False
    )
    subject = args.subject or int(checkpoint["subject"])
    split = load_things_eeg_split(args.dataset_root, subject, args.split)
    count = min(args.count, len(split.image_paths))
    random_generator = np.random.default_rng(args.seed)
    indices = np.sort(
        random_generator.choice(len(split.image_paths), count, replace=False)
    )
    loader = make_loader(
        split, indices, args.batch_size, args.repetitions, False, args.workers
    )
    eeg_model = model_from_checkpoint(checkpoint, device)
    predicted, _, _, query_indices = collect_predictions(eeg_model, loader, device)
    similarities = predicted @ split.features.T
    retrieved_indices = similarities.argmax(dim=1)
    predicted_concepts = [
        str(split.texts[index]).replace("_", " ")
        for index in retrieved_indices.tolist()
    ]
    prompts = [
        (
            f"a clear high quality realistic photograph showing {concept}, "
            "centered subject, detailed, natural lighting"
        )
        for concept in predicted_concepts
    ]

    # SD-Turbo's VAE can produce NaNs with float16 on MPS; CUDA supports it safely.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing()

    generated_images = []
    for offset, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + offset)
        result = pipeline(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
            height=args.image_size,
            width=args.image_size,
        )
        generated_images.append(result.images[0])

    display_size, label_height = args.image_size, 32
    canvas = Image.new(
        "RGB",
        (display_size * 2, count * (display_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    records = []
    for row, (query, retrieved, generated) in enumerate(
        zip(
            query_indices.tolist(),
            retrieved_indices.tolist(),
            generated_images,
        )
    ):
        target = Image.open(
            resolve_image_path(args.dataset_root, str(split.image_paths[query]))
        ).convert("RGB")
        target = target.resize(
            (display_size, display_size), Image.Resampling.LANCZOS
        )
        generated = generated.resize(
            (display_size, display_size), Image.Resampling.LANCZOS
        )
        y = row * (display_size + label_height)
        canvas.paste(target, (0, y + label_height))
        canvas.paste(generated, (display_size, y + label_height))
        draw.text((4, y + 7), f"Target: {split.texts[query]}", fill="black")
        draw.text(
            (display_size + 4, y + 7),
            f"EEG prediction: {split.texts[retrieved]}",
            fill="black",
        )
        records.append(
            {
                "target": str(split.image_paths[query]),
                "target_concept": str(split.texts[query]),
                "eeg_predicted_concept": str(split.texts[retrieved]),
                "prompt": prompts[row],
                "seed": args.seed + row,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"diffusion_sheet={args.output}")


def command_train_scratch(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    split = load_raw_eeg_split(args.dataset_root, args.subject, "train")
    train_indices, validation_indices = concept_split_indices(
        split.labels, args.validation_fraction, args.seed
    )
    train_indices = limit_indices(train_indices, args.max_train_images, args.seed)
    validation_indices = limit_indices(
        validation_indices, args.max_validation_images, args.seed + 1
    )
    if len(train_indices) < 2 or len(validation_indices) < 2:
        raise ValueError("Training and validation each require at least two images")

    def make_image_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = EEGImageDataset(
            split,
            args.dataset_root,
            indices,
            repetitions=args.repetitions,
            image_size=args.image_size,
            cache_images=True,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle
            and len(dataset) > args.batch_size
            and len(dataset) % args.batch_size == 1,
        )

    train_loader = make_image_loader(train_indices, True)
    validation_loader = make_image_loader(validation_indices, False)
    channels, time_points = split.eeg.shape[-2:]
    encoder_config = {
        "channels": channels,
        "time_points": time_points,
        "embedding_dim": args.condition_dim,
        "dropout": args.dropout,
    }
    generator_config = {
        "condition_dim": args.condition_dim,
        "image_size": args.image_size,
    }
    encoder = EEGEncoder(**encoder_config).to(device)
    generator = ScratchGenerator(**generator_config).to(device)
    discriminator = ConditionalDiscriminator(args.condition_dim).to(device)
    generator_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=args.generator_learning_rate,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        betas=(0.5, 0.999),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    stale_epochs = 0
    print(
        f"device={device} train_images={len(train_indices)} "
        f"validation_images={len(validation_indices)} pretrained_weights=none"
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        generator.train()
        discriminator.train()
        generator_total = 0.0
        discriminator_total = 0.0
        seen = 0
        for eeg, real, _, _ in train_loader:
            eeg, real = eeg.to(device), real.to(device)
            condition = encoder(eeg)

            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = generator(condition)
            real_scores = discriminator(real, condition.detach())
            fake_scores = discriminator(fake, condition.detach())
            mismatched = discriminator(real.roll(1, dims=0), condition.detach())
            discriminator_loss = discriminator_hinge_loss(
                real_scores, fake_scores, mismatched
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            condition = encoder(eeg)
            fake = generator(condition)
            adversarial_loss = -discriminator(fake, condition).mean()
            reconstruction_loss = F.l1_loss(fake, real)
            generator_loss = (
                adversarial_loss
                + args.reconstruction_weight * reconstruction_loss
            )
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(generator.parameters()),
                args.gradient_clip,
            )
            generator_optimizer.step()

            batch_size = len(eeg)
            generator_total += generator_loss.detach().item() * batch_size
            discriminator_total += discriminator_loss.detach().item() * batch_size
            seen += batch_size

        encoder.eval()
        generator.eval()
        validation_total = 0.0
        validation_seen = 0
        with torch.inference_mode():
            for eeg, target, _, _ in validation_loader:
                generated = generator(encoder(eeg.to(device)))
                validation_total += (
                    F.l1_loss(generated, target.to(device)).item() * len(eeg)
                )
                validation_seen += len(eeg)
        validation_loss = validation_total / validation_seen
        record = {
            "epoch": float(epoch),
            "generator_loss": generator_total / seen,
            "discriminator_loss": discriminator_total / seen,
            "validation_l1": validation_loss,
        }
        history.append(record)
        print(
            f"epoch={epoch} generator_loss={record['generator_loss']:.4f} "
            f"discriminator_loss={record['discriminator_loss']:.4f} "
            f"validation_l1={validation_loss:.4f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "generator_state": generator.state_dict(),
                    "encoder_config": encoder_config,
                    "generator_config": generator_config,
                    "subject": args.subject,
                    "seed": args.seed,
                    "repetitions": args.repetitions,
                    "best_validation_l1": best_validation,
                    "pretrained_weights": False,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stopping epoch={epoch}")
                break
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint_path}")


def command_generate_scratch(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("pretrained_weights") is not False:
        raise ValueError("Checkpoint is not marked as a from-scratch model")
    subject = args.subject or int(checkpoint["subject"])
    split = load_raw_eeg_split(args.dataset_root, subject, args.split)
    count = min(args.count, len(split.image_paths))
    generator_rng = np.random.default_rng(args.seed)
    indices = np.sort(generator_rng.choice(len(split.image_paths), count, replace=False))
    dataset = EEGImageDataset(
        split,
        args.dataset_root,
        indices,
        repetitions=args.repetitions,
        image_size=checkpoint["generator_config"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder = EEGEncoder(**checkpoint["encoder_config"]).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    image_generator = ScratchGenerator(**checkpoint["generator_config"]).to(device)
    image_generator.load_state_dict(checkpoint["generator_state"])
    encoder.eval()
    image_generator.eval()

    images: list[torch.Tensor] = []
    query_indices: list[torch.Tensor] = []
    with torch.inference_mode():
        for eeg, _, _, index in loader:
            generated = image_generator(encoder(eeg.to(device)))
            images.append(((generated.cpu() + 1) / 2).clamp(0, 1))
            query_indices.append(index)
    generated_tensor = torch.cat(images)
    query_tensor = torch.cat(query_indices)
    build_generated_sheet(
        ThingsEEGSplit(
            eeg=split.eeg,
            features=torch.empty(len(split.labels), 0),
            labels=split.labels,
            image_paths=split.image_paths,
            texts=split.texts,
        ),
        args.dataset_root,
        query_tensor,
        generated_tensor,
        args.output,
    )
    torch.save(
        {
            "query_indices": query_tensor,
            "image_paths": [
                str(split.image_paths[i]) for i in query_tensor.tolist()
            ],
            "generated": generated_tensor,
            "pretrained_weights": False,
        },
        args.output.with_suffix(".pt"),
    )
    print(f"scratch_generated_sheet={args.output}")


def command_train_multisubject(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    subjects = list(dict.fromkeys(args.subjects))
    reference = load_raw_eeg_split(args.dataset_root, subjects[0], "train")
    train_indices, validation_indices = concept_split_indices(
        reference.labels, args.validation_fraction, args.seed
    )
    validation_indices = limit_indices(
        validation_indices, args.max_validation_images, args.seed + 1
    )
    image_cache = build_image_cache(reference, args.dataset_root, args.image_size)
    channels, time_points = reference.eeg.shape[-2:]
    encoder_config = {
        "subjects": len(subjects),
        "channels": channels,
        "time_points": time_points,
        "embedding_dim": args.condition_dim,
        "dropout": args.dropout,
    }
    generator_config = {
        "condition_dim": args.condition_dim,
        "noise_dim": args.noise_dim,
        "image_size": args.image_size,
    }
    encoder = MultiSubjectEEGEncoder(**encoder_config).to(device)
    generator = ScratchGenerator(**generator_config).to(device)
    discriminator = ConditionalDiscriminator(args.condition_dim).to(device)
    generator_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=args.generator_learning_rate,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        betas=(0.5, 0.999),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    stale_epochs = 0
    print(
        f"device={device} subjects={subjects} images_per_subject="
        f"{args.images_per_subject or len(train_indices)} pretrained_weights=none"
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        generator.train()
        discriminator.train()
        generator_total = 0.0
        discriminator_total = 0.0
        seen = 0
        for subject_position, subject in enumerate(subjects):
            subject_split = (
                reference
                if subject == subjects[0]
                else load_raw_eeg_split(args.dataset_root, subject, "train")
            )
            if not np.array_equal(subject_split.image_paths, reference.image_paths):
                raise ValueError(f"Image ordering differs for subject {subject}")
            subject_indices = limit_indices(
                train_indices,
                args.images_per_subject,
                args.seed + epoch * 100 + subject,
            )
            dataset = EEGImageDataset(
                subject_split,
                args.dataset_root,
                subject_indices,
                repetitions=args.repetitions,
                image_size=args.image_size,
                shared_image_cache=image_cache,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=len(dataset) > args.batch_size
                and len(dataset) % args.batch_size == 1,
            )
            for eeg, real, _, _ in loader:
                eeg, real = eeg.to(device), real.to(device)
                subject_ids = torch.full(
                    (len(eeg),),
                    subject_position,
                    dtype=torch.long,
                    device=device,
                )
                condition = encoder(eeg, subject_ids)
                noise = torch.randn(
                    len(eeg), args.noise_dim, device=device
                )

                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = generator(condition, noise)
                real_scores = discriminator(real, condition.detach())
                fake_scores = discriminator(fake, condition.detach())
                mismatched = discriminator(real.roll(1, dims=0), condition.detach())
                discriminator_loss = discriminator_hinge_loss(
                    real_scores, fake_scores, mismatched
                )
                discriminator_loss.backward()
                discriminator_optimizer.step()

                generator_optimizer.zero_grad(set_to_none=True)
                condition = encoder(eeg, subject_ids)
                fake = generator(condition, noise)
                adversarial_loss = -discriminator(fake, condition).mean()
                reconstruction_loss = F.l1_loss(fake, real)
                generator_loss = (
                    adversarial_loss
                    + args.reconstruction_weight * reconstruction_loss
                )
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(generator.parameters()),
                    args.gradient_clip,
                )
                generator_optimizer.step()
                size = len(eeg)
                generator_total += generator_loss.detach().item() * size
                discriminator_total += discriminator_loss.detach().item() * size
                seen += size
            if subject_split is not reference:
                del subject_split, dataset, loader
                gc.collect()

        validation_dataset = EEGImageDataset(
            reference,
            args.dataset_root,
            validation_indices,
            repetitions=args.repetitions,
            image_size=args.image_size,
            shared_image_cache=image_cache,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=False
        )
        encoder.eval()
        generator.eval()
        validation_total = 0.0
        validation_seen = 0
        with torch.inference_mode():
            for eeg, target, _, _ in validation_loader:
                eeg = eeg.to(device)
                subject_ids = torch.zeros(
                    len(eeg), dtype=torch.long, device=device
                )
                condition = encoder(eeg, subject_ids)
                generated = generator(condition)
                validation_total += (
                    F.l1_loss(generated, target.to(device)).item() * len(eeg)
                )
                validation_seen += len(eeg)
        validation_loss = validation_total / validation_seen
        record = {
            "epoch": float(epoch),
            "samples": float(seen),
            "generator_loss": generator_total / seen,
            "discriminator_loss": discriminator_total / seen,
            "validation_l1": validation_loss,
        }
        history.append(record)
        print(
            f"epoch={epoch} samples={seen} "
            f"generator_loss={record['generator_loss']:.4f} "
            f"discriminator_loss={record['discriminator_loss']:.4f} "
            f"validation_l1={validation_loss:.4f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "generator_state": generator.state_dict(),
                    "encoder_config": encoder_config,
                    "generator_config": generator_config,
                    "subjects": subjects,
                    "seed": args.seed,
                    "best_validation_l1": best_validation,
                    "pretrained_weights": False,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stopping epoch={epoch}")
                break
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={checkpoint_path}")


def command_generate_multisubject(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("pretrained_weights") is not False:
        raise ValueError("Checkpoint is not marked as a from-scratch model")
    subjects = checkpoint["subjects"]
    if args.subject not in subjects:
        raise ValueError(f"Subject {args.subject} was not used during training")
    subject_position = subjects.index(args.subject)
    split = load_raw_eeg_split(args.dataset_root, args.subject, args.split)
    count = min(args.count, len(split.image_paths))
    random_generator = np.random.default_rng(args.seed)
    indices = np.sort(
        random_generator.choice(len(split.image_paths), count, replace=False)
    )
    dataset = EEGImageDataset(
        split,
        args.dataset_root,
        indices,
        repetitions=args.repetitions,
        image_size=checkpoint["generator_config"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    encoder = MultiSubjectEEGEncoder(**checkpoint["encoder_config"]).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    generator = ScratchGenerator(**checkpoint["generator_config"]).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    encoder.eval()
    generator.eval()
    generated_parts: list[torch.Tensor] = []
    query_parts: list[torch.Tensor] = []
    torch_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.inference_mode():
        for eeg, _, _, index in loader:
            eeg = eeg.to(device)
            subject_ids = torch.full(
                (len(eeg),), subject_position, dtype=torch.long, device=device
            )
            noise = torch.randn(
                len(eeg),
                checkpoint["generator_config"]["noise_dim"],
                generator=torch_generator,
            ).to(device)
            generated = generator(encoder(eeg, subject_ids), noise)
            generated_parts.append(((generated.cpu() + 1) / 2).clamp(0, 1))
            query_parts.append(index)
    generated_tensor = torch.cat(generated_parts)
    query_tensor = torch.cat(query_parts)
    build_generated_sheet(
        ThingsEEGSplit(
            eeg=split.eeg,
            features=torch.empty(len(split.labels), 0),
            labels=split.labels,
            image_paths=split.image_paths,
            texts=split.texts,
        ),
        args.dataset_root,
        query_tensor,
        generated_tensor,
        args.output,
    )
    torch.save(
        {
            "query_indices": query_tensor,
            "generated": generated_tensor,
            "subjects": subjects,
            "pretrained_weights": False,
        },
        args.output.with_suffix(".pt"),
    )
    print(f"multisubject_generated_sheet={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an EEG-to-image retrieval model on THINGS-EEG."
    )
    parser.set_defaults(func=None)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset-root", type=Path, default=Path("things-eeg")
    )
    common.add_argument("--subject", type=int, default=None)

    subparsers = parser.add_subparsers(dest="command")
    inspect_parser = subparsers.add_parser("inspect", parents=[common])
    inspect_parser.add_argument("--split", choices=("train", "test"), default="train")
    inspect_parser.set_defaults(func=command_inspect, subject=1)

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.set_defaults(func=command_train, subject=1)
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/sub-01"))
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-3)
    train_parser.add_argument("--dropout", type=float, default=0.35)
    train_parser.add_argument("--temperature", type=float, default=0.07)
    train_parser.add_argument("--cosine-weight", type=float, default=0.25)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--repetitions", type=int)
    train_parser.add_argument("--max-train-images", type=int)
    train_parser.add_argument("--max-validation-images", type=int)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate", parents=[common])
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("train", "test"), default="test")
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument("--batch-size", type=int, default=128)
    evaluate_parser.add_argument("--repetitions", type=int)
    evaluate_parser.add_argument("--max-images", type=int)
    evaluate_parser.add_argument("--workers", type=int, default=0)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.set_defaults(func=command_evaluate)

    reconstruct_parser = subparsers.add_parser("reconstruct", parents=[common])
    reconstruct_parser.add_argument("--checkpoint", type=Path, required=True)
    reconstruct_parser.add_argument("--split", choices=("train", "test"), default="test")
    reconstruct_parser.add_argument("--count", type=int, default=8)
    reconstruct_parser.add_argument(
        "--output", type=Path, default=Path("outputs/retrievals.png")
    )
    reconstruct_parser.add_argument("--batch-size", type=int, default=128)
    reconstruct_parser.add_argument("--repetitions", type=int)
    reconstruct_parser.add_argument("--workers", type=int, default=0)
    reconstruct_parser.add_argument("--device", default="auto")
    reconstruct_parser.add_argument("--seed", type=int, default=42)
    reconstruct_parser.set_defaults(func=command_reconstruct)

    decoder_parser = subparsers.add_parser("train-decoder", parents=[common])
    decoder_parser.set_defaults(func=command_train_decoder, subject=1)
    decoder_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/image-decoder")
    )
    decoder_parser.add_argument("--epochs", type=int, default=30)
    decoder_parser.add_argument("--batch-size", type=int, default=128)
    decoder_parser.add_argument("--learning-rate", type=float, default=2e-4)
    decoder_parser.add_argument("--weight-decay", type=float, default=1e-4)
    decoder_parser.add_argument("--gradient-clip", type=float, default=1.0)
    decoder_parser.add_argument("--validation-fraction", type=float, default=0.1)
    decoder_parser.add_argument("--patience", type=int, default=6)
    decoder_parser.add_argument("--image-size", type=int, default=64)
    decoder_parser.add_argument("--max-train-images", type=int)
    decoder_parser.add_argument("--max-validation-images", type=int)
    decoder_parser.add_argument("--workers", type=int, default=0)
    decoder_parser.add_argument("--device", default="auto")
    decoder_parser.add_argument("--seed", type=int, default=42)

    generate_parser = subparsers.add_parser("generate", parents=[common])
    generate_parser.add_argument("--eeg-checkpoint", type=Path, required=True)
    generate_parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    generate_parser.add_argument("--split", choices=("train", "test"), default="test")
    generate_parser.add_argument("--count", type=int, default=8)
    generate_parser.add_argument(
        "--output", type=Path, default=Path("outputs/generated.png")
    )
    generate_parser.add_argument("--batch-size", type=int, default=128)
    generate_parser.add_argument("--repetitions", type=int)
    generate_parser.add_argument("--workers", type=int, default=0)
    generate_parser.add_argument("--device", default="auto")
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.set_defaults(func=command_generate)

    diffusion_parser = subparsers.add_parser(
        "generate-diffusion", parents=[common]
    )
    diffusion_parser.add_argument("--eeg-checkpoint", type=Path, required=True)
    diffusion_parser.add_argument("--split", choices=("train", "test"), default="test")
    diffusion_parser.add_argument("--count", type=int, default=4)
    diffusion_parser.add_argument(
        "--output", type=Path, default=Path("outputs/diffusion-generated.png")
    )
    diffusion_parser.add_argument(
        "--model", default="stabilityai/sd-turbo"
    )
    diffusion_parser.add_argument("--steps", type=int, default=2)
    diffusion_parser.add_argument("--image-size", type=int, default=512)
    diffusion_parser.add_argument("--batch-size", type=int, default=64)
    diffusion_parser.add_argument("--repetitions", type=int)
    diffusion_parser.add_argument("--workers", type=int, default=0)
    diffusion_parser.add_argument("--device", default="auto")
    diffusion_parser.add_argument("--seed", type=int, default=42)
    diffusion_parser.set_defaults(func=command_generate_diffusion)

    scratch_parser = subparsers.add_parser("train-scratch", parents=[common])
    scratch_parser.set_defaults(func=command_train_scratch, subject=1)
    scratch_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/from-scratch")
    )
    scratch_parser.add_argument("--epochs", type=int, default=30)
    scratch_parser.add_argument("--batch-size", type=int, default=128)
    scratch_parser.add_argument("--condition-dim", type=int, default=256)
    scratch_parser.add_argument("--dropout", type=float, default=0.35)
    scratch_parser.add_argument("--generator-learning-rate", type=float, default=2e-4)
    scratch_parser.add_argument(
        "--discriminator-learning-rate", type=float, default=1e-4
    )
    scratch_parser.add_argument("--reconstruction-weight", type=float, default=10.0)
    scratch_parser.add_argument("--gradient-clip", type=float, default=1.0)
    scratch_parser.add_argument("--validation-fraction", type=float, default=0.1)
    scratch_parser.add_argument("--patience", type=int, default=6)
    scratch_parser.add_argument("--image-size", type=int, default=64)
    scratch_parser.add_argument("--repetitions", type=int)
    scratch_parser.add_argument("--max-train-images", type=int)
    scratch_parser.add_argument("--max-validation-images", type=int)
    scratch_parser.add_argument("--workers", type=int, default=0)
    scratch_parser.add_argument("--device", default="auto")
    scratch_parser.add_argument("--seed", type=int, default=42)

    scratch_generate_parser = subparsers.add_parser(
        "generate-scratch", parents=[common]
    )
    scratch_generate_parser.add_argument("--checkpoint", type=Path, required=True)
    scratch_generate_parser.add_argument(
        "--split", choices=("train", "test"), default="test"
    )
    scratch_generate_parser.add_argument("--count", type=int, default=8)
    scratch_generate_parser.add_argument(
        "--output", type=Path, default=Path("outputs/from-scratch/generated.png")
    )
    scratch_generate_parser.add_argument("--batch-size", type=int, default=64)
    scratch_generate_parser.add_argument("--repetitions", type=int)
    scratch_generate_parser.add_argument("--device", default="auto")
    scratch_generate_parser.add_argument("--seed", type=int, default=42)
    scratch_generate_parser.set_defaults(func=command_generate_scratch)

    multisubject_parser = subparsers.add_parser(
        "train-multisubject-scratch", parents=[common]
    )
    multisubject_parser.set_defaults(func=command_train_multisubject)
    multisubject_parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(range(1, 11))
    )
    multisubject_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/multisubject-scratch")
    )
    multisubject_parser.add_argument("--epochs", type=int, default=8)
    multisubject_parser.add_argument("--images-per-subject", type=int, default=2000)
    multisubject_parser.add_argument("--batch-size", type=int, default=256)
    multisubject_parser.add_argument("--condition-dim", type=int, default=256)
    multisubject_parser.add_argument("--noise-dim", type=int, default=64)
    multisubject_parser.add_argument("--dropout", type=float, default=0.35)
    multisubject_parser.add_argument(
        "--generator-learning-rate", type=float, default=2e-4
    )
    multisubject_parser.add_argument(
        "--discriminator-learning-rate", type=float, default=1e-4
    )
    multisubject_parser.add_argument(
        "--reconstruction-weight", type=float, default=5.0
    )
    multisubject_parser.add_argument("--gradient-clip", type=float, default=1.0)
    multisubject_parser.add_argument("--validation-fraction", type=float, default=0.1)
    multisubject_parser.add_argument("--max-validation-images", type=int)
    multisubject_parser.add_argument("--patience", type=int, default=3)
    multisubject_parser.add_argument("--image-size", type=int, default=64)
    multisubject_parser.add_argument("--repetitions", type=int)
    multisubject_parser.add_argument("--device", default="auto")
    multisubject_parser.add_argument("--seed", type=int, default=42)

    multisubject_generate_parser = subparsers.add_parser(
        "generate-multisubject-scratch", parents=[common]
    )
    multisubject_generate_parser.set_defaults(
        func=command_generate_multisubject, subject=1
    )
    multisubject_generate_parser.add_argument(
        "--checkpoint", type=Path, required=True
    )
    multisubject_generate_parser.add_argument(
        "--split", choices=("train", "test"), default="test"
    )
    multisubject_generate_parser.add_argument("--count", type=int, default=8)
    multisubject_generate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multisubject-scratch/generated.png"),
    )
    multisubject_generate_parser.add_argument("--batch-size", type=int, default=64)
    multisubject_generate_parser.add_argument("--repetitions", type=int)
    multisubject_generate_parser.add_argument("--device", default="auto")
    multisubject_generate_parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        return
    args.func(args)
