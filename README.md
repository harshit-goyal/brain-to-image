# Brain-to-Image from THINGS-EEG

This repository implements an EEG-to-image **retrieval baseline** for the included
THINGS-EEG data. It replaces the earlier DREAMER autoencoder experiments, which did not
contain paired visual stimuli and therefore could not validate brain-to-image decoding.

The pipeline averages repeated EEG presentations to improve signal-to-noise ratio, uses a
compact EEGNet-style encoder to predict the provided 512-dimensional ViT-B/32 image
features, and evaluates image and concept retrieval against chance and shuffled-pair
baselines. The validation split is made by concept, preventing images from the same object
category from appearing in both training and validation.

## Data layout

The included data is expected at:

```text
things-eeg/
├── Image_set/
│   ├── training_images/
│   └── test_images/
└── Preprocessed_data_250Hz_whiten/
    ├── sub-01/train.pt
    ├── sub-01/test.pt
    ├── ViT-B-32_features_train.pt
    └── ViT-B-32_features_test.pt
```

Each training item has four EEG repetitions. Each official test stimulus has 80
repetitions. Repetitions are averaged lazily when a batch is loaded, so the approximately
2 GB subject files are not duplicated in memory. Keep the default `--workers 0` on macOS:
worker processes use spawn and can otherwise copy the in-memory subject array.

## Usage

Inspect the installed tensors:

```bash
python3 main.py inspect --subject 1 --split train
```

Run a quick smoke training:

```bash
python3 main.py train \
  --subject 1 \
  --epochs 2 \
  --max-train-images 256 \
  --max-validation-images 128 \
  --output-dir outputs/smoke
```

Train on all Subject 01 images:

```bash
python3 main.py train --subject 1 --epochs 50 --output-dir outputs/sub-01
```

Evaluate on the official 200-concept test set:

```bash
python3 main.py evaluate \
  --checkpoint outputs/sub-01/best.pt \
  --split test \
  --output outputs/sub-01/test_metrics.json
```

Render targets alongside the images retrieved from EEG:

```bash
python3 main.py reconstruct \
  --checkpoint outputs/sub-01/best.pt \
  --split test \
  --count 8 \
  --output outputs/sub-01/retrievals.png
```

Train the 64×64 visual-feature decoder and generate new pixels from EEG:

```bash
python3 main.py train-decoder --epochs 30 --output-dir outputs/image-decoder
python3 main.py generate \
  --eeg-checkpoint outputs/sub-01/best.pt \
  --decoder-checkpoint outputs/image-decoder/best.pt \
  --output outputs/sub-01/generated.png
```

For sharp, photorealistic output, use EEG retrieval as a semantic prediction and render
that prediction with SD-Turbo:

```bash
python3 -m pip install -r requirements-diffusion.txt
python3 main.py generate-diffusion \
  --eeg-checkpoint outputs/sub-01/best.pt \
  --output outputs/sub-01/diffusion-generated.png
```

This is a two-stage semantic reconstruction: EEG predicts the object concept and diffusion
generates a new image from that concept. It produces clear images, but it does not recover
the target's exact pose, background, color, or pixel layout.

To train every component locally without any pretrained visual model or external weights:

```bash
python3 main.py train-scratch --epochs 30 --output-dir outputs/from-scratch
python3 main.py generate-scratch \
  --checkpoint outputs/from-scratch/best.pt \
  --output outputs/from-scratch/generated.png
```

The from-scratch path trains an EEG encoder, conditional generator, and projection
discriminator using only the local Subject 01 EEG and stimulus images. Its output is the
most direct experiment, but it is expected to be substantially less detailed than
pretrained diffusion because the local training set is small.

Use all ten local subjects with subject-specific embeddings and no pretrained weights:

```bash
python3 main.py train-multisubject-scratch \
  --subjects 1 2 3 4 5 6 7 8 9 10 \
  --epochs 8 \
  --images-per-subject 2000 \
  --output-dir outputs/multisubject-scratch

python3 main.py generate-multisubject-scratch \
  --checkpoint outputs/multisubject-scratch/best.pt \
  --subject 1 \
  --output outputs/multisubject-scratch/generated.png
```

Subject files are loaded sequentially to avoid holding the complete 24 GB EEG dataset in
memory. A different deterministic image subset is sampled for each subject and epoch.

On Apple Silicon, `--device auto` uses MPS when available. Use `--device cpu` to force
CPU execution.

## Subject 01 baseline

The checked-in pipeline was trained for 12 epochs on all 14,890 training images in the
concept-disjoint training partition. Its best checkpoint is at `outputs/sub-01/best.pt`.

| Official test metric | Model | Chance | Shuffled EEG |
|---|---:|---:|---:|
| Top-1 image/concept retrieval | 13.5% | 0.5% | 0.5% |
| Top-5 image/concept retrieval | 41.0% | 2.5% | 2.0% |
| Top-10 image/concept retrieval | 59.0% | 5.0% | 3.5% |

The official test gallery contains one image per concept, so image and concept accuracy
are identical for this split. The median target rank is 8 out of 200. Full precision values
are stored in `outputs/sub-01/test_metrics.json`.

## Interpreting results

The key acceptance criterion is that held-out **concept retrieval** is consistently above
the analytical and shuffled-pair baselines reported at the same value of `k`. Image top-k
retrieval is stricter:
training concepts contain ten different images, while the official test set contains one
image for each unseen concept.

Nearest-neighbor retrieval is intentionally used before generative reconstruction. If the
model cannot identify visual semantics above chance, attaching a GAN or diffusion decoder
would produce visually plausible but scientifically unsupported images. Once retrieval is
validated, the predicted embeddings can condition a pretrained diffusion decoder.
