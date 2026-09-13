from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"
TRAIN_ROOT = ROOT / "things-eeg" / "Image_set_Resize" / "train_images"

TRAIN_CONCEPTS = (
    "00001_aardvark",
    "00237_cape",
    "00473_earwig",
    "00709_hot_tub",
    "00944_notebook",
    "01181_rope",
    "01418_sweatsuit",
    "01654_zucchini",
)

OUTPUTS = (
    ("Target", ROOT / "outputs" / "sub-01" / "generated.png", 256, 28),
    ("Direct decoder", ROOT / "outputs" / "sub-01" / "generated.png", 256, 28),
    (
        "Scratch, 1 subject",
        ROOT / "outputs" / "from-scratch" / "generated.png",
        256,
        28,
    ),
    (
        "Scratch, 10 subjects",
        ROOT / "outputs" / "multisubject-scratch" / "generated.png",
        256,
        28,
    ),
    (
        "Pretrained rendering",
        ROOT / "outputs" / "sub-01" / "diffusion-generated.png",
        512,
        32,
    ),
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def square_thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.LANCZOS)


def make_training_montage() -> None:
    tile = 250
    label_height = 34
    margin = 12
    canvas = Image.new(
        "RGB",
        (4 * tile + 5 * margin, 2 * (tile + label_height) + 3 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(22)

    for index, concept in enumerate(TRAIN_CONCEPTS):
        image_path = sorted((TRAIN_ROOT / concept).glob("*.jpg"))[0]
        row, column = divmod(index, 4)
        x = margin + column * (tile + margin)
        y = margin + row * (tile + label_height + margin)
        canvas.paste(square_thumbnail(image_path, tile), (x, y))
        label = concept.split("_", 1)[1].replace("_", " ")
        draw.text((x, y + tile + 5), label, fill="black", font=label_font)

    canvas.save(FIGURES / "training_samples.png", optimize=True)


def extract_output(
    path: Path, row: int, source_size: int, header_height: int, target: bool
) -> Image.Image:
    with Image.open(path) as image:
        x0 = 0 if target else source_size
        y0 = row * (source_size + header_height) + header_height
        crop = image.crop((x0, y0, x0 + source_size, y0 + source_size)).convert("RGB")
    return crop


def make_generation_comparison() -> None:
    selected_rows = ((0, "bok choy"), (2, "gopher"), (4, "popcorn"), (6, "spoon"))
    tile = 190
    top = 48
    row_label_width = 88
    margin = 8
    canvas = Image.new(
        "RGB",
        (
            row_label_width + len(OUTPUTS) * (tile + margin) + margin,
            top + len(selected_rows) * (tile + margin) + margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    header_font = font(17)
    row_font = font(19)

    for column, (label, _, _, _) in enumerate(OUTPUTS):
        x = row_label_width + column * (tile + margin)
        draw.text((x, 12), label, fill="black", font=header_font)

    for output_row, (source_row, concept) in enumerate(selected_rows):
        y = top + output_row * (tile + margin)
        draw.text((8, y + tile // 2 - 10), concept, fill="black", font=row_font)
        for column, (_, path, source_size, header_height) in enumerate(OUTPUTS):
            crop = extract_output(
                path,
                source_row,
                source_size,
                header_height,
                target=column == 0,
            )
            crop = crop.resize((tile, tile), Image.Resampling.LANCZOS)
            x = row_label_width + column * (tile + margin)
            canvas.paste(crop, (x, y))

    canvas.save(FIGURES / "generation_comparison.png", optimize=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_training_montage()
    make_generation_comparison()


if __name__ == "__main__":
    main()
