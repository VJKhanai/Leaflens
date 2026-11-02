
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "cleaned"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
THRESHOLD = 4

records = []

for split in SPLITS:
    for path in sorted((DATA_DIR / split).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        with Image.open(path) as img:
            phash = imagehash.phash(img.convert("RGB"))

        records.append((split, path, phash))

print("Images analyzed:", len(records))

# Compare candidate pairs across different splits only.
# Bucket by class to reduce comparisons and prioritize
# leakage involving images of the same plant.
by_class = defaultdict(list)

for split, path, phash in records:
    by_class[path.parent.name].append((split, path, phash))

matches = []

for plant, images in by_class.items():
    for i in range(len(images)):
        split_a, path_a, hash_a = images[i]

        for j in range(i + 1, len(images)):
            split_b, path_b, hash_b = images[j]

            if split_a == split_b:
                continue

            distance = hash_a - hash_b

            if distance <= THRESHOLD:
                matches.append(
                    (distance, plant, split_a, path_a,
                     split_b, path_b)
                )

matches.sort(key=lambda x: x[0])

import csv

REPORT_PATH = ROOT / "ML" / "near_duplicates_report.csv"

with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "distance",
        "plant",
        "split_a",
        "image_a",
        "split_b",
        "image_b"
    ])

    for distance, plant, split_a, path_a, split_b, path_b in matches:
        writer.writerow([
            distance,
            plant,
            split_a,
            str(path_a),
            split_b,
            str(path_b)
        ])

print("Report saved:", REPORT_PATH)

print("\n--- POSSIBLE NEAR-DUPLICATES ---")
print("Cross-split candidate pairs:", len(matches))

for distance, plant, split_a, path_a, split_b, path_b in matches[:30]:
    print(f"\nPlant: {plant}")
    print(f"pHash distance: {distance}")
    print(f"[{split_a}] {path_a}")
    print(f"[{split_b}] {path_b}")

print("\nReview matches visually before making changes.")
