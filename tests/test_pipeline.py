import unittest

import numpy as np
import torch

from brain2image.data import AveragedEEGDataset, ThingsEEGSplit, concept_split_indices
from brain2image.metrics import retrieval_metrics, shuffled_baseline
from brain2image.model import (
    ConditionalDiscriminator,
    EEGEncoder,
    ImageDecoder,
    MultiSubjectEEGEncoder,
    ScratchGenerator,
    contrastive_alignment_loss,
    discriminator_hinge_loss,
    multiscale_reconstruction_loss,
)


class PipelineTests(unittest.TestCase):
    def test_concept_split_has_no_label_leakage(self) -> None:
        labels = np.repeat(np.arange(20), 10)
        train, validation = concept_split_indices(labels, 0.2, seed=7)
        self.assertTrue(set(labels[train]).isdisjoint(set(labels[validation])))
        self.assertEqual(len(validation), 40)

    def test_dataset_averages_repetitions(self) -> None:
        eeg = np.arange(2 * 4 * 3 * 5, dtype=np.float16).reshape(2, 4, 3, 5)
        split = ThingsEEGSplit(
            eeg=eeg,
            features=torch.randn(2, 8),
            labels=np.array([0, 1]),
            image_paths=np.array(["a.jpg", "b.jpg"]),
            texts=np.array(["a", "b"]),
        )
        dataset = AveragedEEGDataset(split, repetitions=2)
        sample, _, label, index = dataset[1]
        expected = torch.from_numpy(eeg[1, :2].astype(np.float32).mean(axis=0))
        self.assertTrue(torch.equal(sample, expected))
        self.assertEqual((label, index), (1, 1))

    def test_encoder_and_loss(self) -> None:
        model = EEGEncoder(channels=4, time_points=64, embedding_dim=16)
        predicted = model(torch.randn(5, 4, 64))
        target = torch.randn(5, 16)
        loss = contrastive_alignment_loss(
            predicted, target, labels=torch.tensor([0, 0, 1, 2, 3])
        )
        self.assertEqual(predicted.shape, (5, 16))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(predicted.norm(dim=1), torch.ones(5), atol=1e-5))

    def test_perfect_retrieval(self) -> None:
        target = torch.eye(6)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        metrics = retrieval_metrics(target, target, labels)
        self.assertEqual(metrics["image_top1"], 1.0)
        self.assertEqual(metrics["concept_top1"], 1.0)
        self.assertEqual(metrics["median_rank"], 1.0)
        self.assertAlmostEqual(metrics["chance_image_top5"], 5 / 6)

    def test_baselines_cover_each_top_k(self) -> None:
        generator = torch.Generator().manual_seed(2)
        predicted = torch.randn(20, 8, generator=generator)
        target = torch.randn(20, 8, generator=generator)
        labels = torch.repeat_interleave(torch.arange(5), 4)
        metrics = retrieval_metrics(predicted, target, labels)
        shuffled = shuffled_baseline(predicted, target, labels, seed=3)
        for k in (1, 5, 10):
            self.assertIn(f"chance_image_top{k}", metrics)
            self.assertIn(f"chance_concept_top{k}", metrics)
            self.assertIn(f"shuffled_image_top{k}", shuffled)
            self.assertIn(f"shuffled_concept_top{k}", shuffled)

    def test_image_decoder_generates_pixels(self) -> None:
        decoder = ImageDecoder(embedding_dim=16)
        generated = decoder(torch.randn(2, 16))
        target = torch.rand(2, 3, 64, 64)
        loss = multiscale_reconstruction_loss(generated, target)
        self.assertEqual(generated.shape, (2, 3, 64, 64))
        self.assertGreaterEqual(generated.min().item(), 0.0)
        self.assertLessEqual(generated.max().item(), 1.0)
        self.assertTrue(torch.isfinite(loss))

    def test_from_scratch_gan_shapes_and_loss(self) -> None:
        generator = ScratchGenerator(condition_dim=16)
        discriminator = ConditionalDiscriminator(condition_dim=16)
        condition = torch.randn(3, 16)
        images = generator(condition)
        scores = discriminator(images, condition)
        loss = discriminator_hinge_loss(scores, scores - 1, scores.roll(1))
        self.assertEqual(images.shape, (3, 3, 64, 64))
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(torch.isfinite(loss))

    def test_multisubject_encoder(self) -> None:
        encoder = MultiSubjectEEGEncoder(
            subjects=3, channels=4, time_points=64, embedding_dim=16
        )
        output = encoder(torch.randn(3, 4, 64), torch.tensor([0, 1, 2]))
        self.assertEqual(output.shape, (3, 16))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(3), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
