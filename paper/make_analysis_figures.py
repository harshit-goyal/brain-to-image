from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "paper-analysis" / "robustness.json"
FIGURES = ROOT / "paper" / "figures"


def line_chart(
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    x_label: str,
    output: Path,
) -> None:
    width, height = 1200, 650
    left, top, right, bottom = 105, 80, 1140, 555
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=22)
    font = ImageFont.load_default(size=18)
    draw.text((45, 25), title, fill="black", font=title_font)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    for tick in range(0, 61, 10):
        y = bottom - int(tick / 60 * (bottom - top))
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((42, y - 10), f"{tick}%", fill="black", font=font)
    count = len(labels)
    xs = [
        left + int(index * (right - left) / max(count - 1, 1))
        for index in range(count)
    ]
    for x, label in zip(xs, labels):
        draw.text((x - 12, bottom + 15), label, fill="black", font=font)
    draw.text(
        ((left + right) // 2 - 70, bottom + 55),
        x_label,
        fill="black",
        font=font,
    )
    for series_index, (name, values, color) in enumerate(series):
        points = [
            (x, bottom - int(value / 60 * (bottom - top)))
            for x, value in zip(xs, values)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        legend_x = 760 + series_index * 135
        draw.line((legend_x, 57, legend_x + 28, 57), fill=color, width=5)
        draw.text((legend_x + 35, 48), name, fill="black", font=font)
    image.save(output)


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)

    repetitions = data["repetition_ablation"]
    repetition_labels = list(repetitions)
    line_chart(
        repetition_labels,
        [
            (
                f"R@{k}",
                [repetitions[label][f"image_top{k}"] * 100 for label in repetition_labels],
                color,
            )
            for k, color in ((1, "#2563eb"), (5, "#f59e0b"), (10, "#16a34a"))
        ],
        "Retrieval versus session-balanced EEG repetitions (Subject 01)",
        "Repetitions averaged",
        FIGURES / "repetition_ablation.png",
    )

    subjects = data["cross_subject"]
    subject_labels = list(subjects)
    line_chart(
        subject_labels,
        [
            (
                f"R@{k}",
                [subjects[label][f"image_top{k}"] * 100 for label in subject_labels],
                color,
            )
            for k, color in ((1, "#2563eb"), (5, "#f59e0b"), (10, "#16a34a"))
        ],
        "Subject 01 model transferred without adaptation",
        "Test subject",
        FIGURES / "cross_subject.png",
    )


if __name__ == "__main__":
    main()
