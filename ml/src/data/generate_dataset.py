import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = ["blur", "dark", "low_contrast", "noise", "skew", "low_resolution"]
VARIANTS = [
    ("clean", {}),
    ("blur", {"blur": 1}),
    ("dark", {"dark": 1}),
    ("low_contrast", {"low_contrast": 1}),
    ("noise", {"noise": 1}),
    ("skew", {"skew": 1}),
    ("low_resolution", {"low_resolution": 1}),
    ("composed", {"blur": 1, "dark": 1, "noise": 1}),
]


def build_page(source_index: int, rng: random.Random) -> Image.Image:
    width, height = 768, 1024
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "C:/Windows/Fonts/arial.ttf"
    title = ImageFont.truetype(font_path, 34) if Path(font_path).exists() else ImageFont.load_default()
    body = ImageFont.truetype(font_path, 20) if Path(font_path).exists() else ImageFont.load_default()
    draw.text((60, 50), "DOCLEAR AI", fill=(20, 40, 80), font=title)
    draw.text((60, 105), "DOCUMENT QUALITY SAMPLE", fill=(30, 30, 30), font=body)
    draw.line((60, 145, width - 60, 145), fill=(50, 80, 130), width=3)
    fields = [
        "Name: Test User",
        f"Application ID: TEST-{2026 + source_index % 3:04d}-{source_index:04d}",
        "Department: Computer Science",
        "Document Type: Synthetic Quality Sample",
    ]
    for index, text in enumerate(fields):
        draw.text((75, 190 + index * 42), text, fill=(35, 35, 35), font=body)
    draw.rectangle((60, 390, width - 60, 610), outline=(80, 80, 80), width=2)
    for row in range(1, 4):
        y = 390 + row * 55
        draw.line((60, y, width - 60, y), fill=(130, 130, 130), width=1)
    for column in range(1, 4):
        x = 60 + column * 162
        draw.line((x, 390, x, 610), fill=(130, 130, 130), width=1)
    for index in range(5):
        draw.text((75, 680 + index * 42), f"This is synthetic document paragraph {index + 1} for sample {source_index:04d}.", fill=(45, 45, 45), font=body)
    return image


def apply_degradations(image: Image.Image, labels: dict[str, int], rng: random.Random) -> Image.Image:
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if labels.get("blur"):
        array = cv2.GaussianBlur(array, (0, 0), 2.5)
    if labels.get("dark"):
        array = np.clip(array.astype(np.float32) * 0.42, 0, 255).astype(np.uint8)
    if labels.get("low_contrast"):
        array = np.clip((array.astype(np.float32) - 128) * 0.38 + 128, 0, 255).astype(np.uint8)
    if labels.get("noise"):
        noise = rng.normalvariate(0, 1)
        rng_state = np.random.default_rng(int(abs(noise) * 1_000_000) + rng.randint(0, 1_000_000))
        array = np.clip(array.astype(np.float32) + rng_state.normal(0, 18, array.shape), 0, 255).astype(np.uint8)
    output = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    if labels.get("skew"):
        angle = rng.uniform(-6, 6)
        output = output.rotate(angle, expand=True, fillcolor="white", resample=Image.Resampling.BICUBIC)
    if labels.get("low_resolution"):
        small = output.resize((192, 256), Image.Resampling.BILINEAR)
        output = small.resize(output.size, Image.Resampling.BICUBIC)
    return output


def generate_dataset(ml_root: Path, source_count: int = 500, seed: int = 42) -> dict:
    rng = random.Random(seed)
    dataset_root = ml_root / "datasets"
    raw_root = dataset_root / "raw" / "clean"
    processed_root = dataset_root / "processed"
    splits_root = dataset_root / "splits"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)
    ids = list(range(source_count))
    rng.shuffle(ids)
    boundaries = (int(source_count * 0.70), int(source_count * 0.85))
    split_by_id = {source_id: "train" if index < boundaries[0] else "val" if index < boundaries[1] else "test" for index, source_id in enumerate(ids)}
    rows = []
    for source_id in range(source_count):
        source_key = f"src_{source_id:05d}"
        clean = build_page(source_id, rng)
        raw_path = raw_root / f"{source_key}.png"
        clean.save(raw_path)
        split = split_by_id[source_id]
        for variant_name, labels in VARIANTS:
            variant = apply_degradations(clean, labels, rng) if labels else clean.copy()
            relative_path = Path("datasets") / "processed" / split / f"{source_key}_{variant_name}.png"
            target = ml_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            variant.save(target)
            rows.append({"image_path": relative_path.as_posix(), "source_id": source_key, "split": split, **{label: labels.get(label, 0) for label in LABELS}})
    metadata_path = dataset_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "source_id", "split", *LABELS])
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        with (splits_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "source_id", "split", *LABELS])
            writer.writeheader()
            writer.writerows(split_rows)
    (splits_root / "source_split.json").write_text(__import__("json").dumps({"seed": seed, "source_count": source_count, "split_by_source": split_by_id}, indent=2), encoding="utf-8")
    return {"source_count": source_count, "image_count": len(rows), "split_counts": {split: sum(1 for row in rows if row["split"] == split) for split in ("train", "val", "test")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sources", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(generate_dataset(args.root, args.sources, args.seed))
