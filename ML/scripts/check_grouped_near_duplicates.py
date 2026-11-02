from pathlib import Path
from itertools import combinations

from PIL import Image
import imagehash
from matplotlib import image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "grouped"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
THRESHOLD = 4


def get_hash(path):
    try:
        with Image.open(path) as image:
            image = image.convert("L")
            image.thumbnail((256, 256), Image.Resampling.BILINEAR)
            return imagehash.phash(image)
    except Exception as error:
        print("Cannot read:", path, error)
        return None


print("Calculating perceptual hashes...")

import csv

total_pairs = 0
examples = []

all_pairs = []

classes = sorted(
    p.name for p in (DATA_DIR / "train").iterdir()
    if p.is_dir()
)

for plant_number, plant in enumerate(classes, start=1):
    print(
        f"Checking {plant_number}/{len(classes)}: {plant}",
        flush=True
    )
    images = []

    for split in ("train", "val", "test"):
        folder = DATA_DIR / split / plant

        for path in folder.iterdir():
            if path.suffix.lower() in EXTENSIONS:
                image_hash = get_hash(path)

                if image_hash is not None:
                    images.append((split, path, image_hash))

    for (split_a, path_a, hash_a), (split_b, path_b, hash_b) in combinations(images, 2):
        if split_a == split_b:
            continue

        distance = hash_a - hash_b

        if distance <= THRESHOLD:
            total_pairs += 1
            all_pairs.append({
        "class": plant,
        "split_a": split_a,
        "path_a": str(path_a),
        "split_b": split_b,
        "path_b": str(path_b),
        "distance": int(distance)
    })

            if len(examples) < 20:
                examples.append(
                    (plant, split_a, path_a.name,
                     split_b, path_b.name, distance)
                )

print("\nCross-split near-duplicate candidate pairs:", total_pairs)

for item in examples:
    print(item)

print("\nNear-duplicate check complete.")
output_path = ROOT / "ML" / "near_duplicate_candidates.csv"

with output_path.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "split_a",
            "path_a",
            "split_b",
            "path_b",
            "distance"
        ]
    )

    writer.writeheader()
    writer.writerows(all_pairs)

print("Saved candidate pairs to:", output_path)