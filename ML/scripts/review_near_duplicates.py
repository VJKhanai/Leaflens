
from pathlib import Path
from collections import defaultdict
import math

from PIL import Image, ImageDraw, ImageFont
import imagehash

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "cleaned"
OUTPUT = ROOT / "ML" / "near_duplicate_review.jpg"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
THRESHOLD = 4

by_class = defaultdict(list)

for split in SPLITS:
    for path in sorted((DATA_DIR / split).rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            with Image.open(path) as img:
                phash = imagehash.phash(img.convert("RGB"))

            by_class[path.parent.name].append(
                (split, path, phash)
            )

matches = []

for plant, records in by_class.items():
    for i in range(len(records)):
        split_a, path_a, hash_a = records[i]

        for j in range(i + 1, len(records)):
            split_b, path_b, hash_b = records[j]

            if split_a == split_b:
                continue

            distance = hash_a - hash_b

            if distance <= THRESHOLD:
                matches.append(
                    (distance, plant, split_a,
                     path_a, split_b, path_b)
                )

matches.sort(key=lambda item: item[0])

# Show the 30 most similar pairs.
matches = matches[:30]

CELL_WIDTH = 300
CELL_HEIGHT = 220
PAIR_HEIGHT = 270

sheet = Image.new(
    "RGB",
    (CELL_WIDTH * 2, PAIR_HEIGHT * len(matches)),
    "white",
)

draw = ImageDraw.Draw(sheet)

for index, match in enumerate(matches):
    distance, plant, split_a, path_a, split_b, path_b = match

    y = index * PAIR_HEIGHT

    draw.text(
        (10, y + 5),
        f"{index + 1}. {plant} | pHash distance: {distance}",
        fill="black",
    )

    for col, (split, path) in enumerate([
        (split_a, path_a),
        (split_b, path_b)
    ]):
        with Image.open(path) as original:
            img = original.convert("RGB")
            img.thumbnail((280, 190))

        x = col * CELL_WIDTH + 10

        sheet.paste(img, (x, y + 30))

        draw.text(
            (x, y + 225),
            f"{split}: {path.name[:32]}",
            fill="black",
        )

if not matches:
    print("No candidate pairs found.")
else:
    sheet.save(OUTPUT, quality=90)
    print("Review sheet created:", OUTPUT)
    print("Pairs displayed:", len(matches))
