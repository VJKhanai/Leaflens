import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "ML" / "near_duplicate_candidates.csv"
OUTPUT_DIR = ROOT / "ML" / "near_duplicate_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_WIDTH = 760
PAIR_HEIGHT = 240
IMAGE_SIZE = (300, 185)
PAIRS_PER_SHEET = 12

with CSV_PATH.open(encoding="utf-8", newline="") as file:
    pairs = list(csv.DictReader(file))

def draw_image(canvas, path, x, y):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(IMAGE_SIZE)

        canvas.paste(image, (x, y))

sheet_count = math.ceil(len(pairs) / PAIRS_PER_SHEET)

for sheet_index in range(sheet_count):
    subset = pairs[
        sheet_index * PAIRS_PER_SHEET:
        (sheet_index + 1) * PAIRS_PER_SHEET
    ]

    sheet = Image.new(
        "RGB",
        (PAIR_WIDTH, PAIR_HEIGHT * len(subset)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, pair in enumerate(subset):
        y = index * PAIR_HEIGHT

        draw.text(
            (15, y + 8),
            f"Pair {sheet_index * PAIRS_PER_SHEET + index + 1}: "
            f"{pair['class']} | pHash distance: {pair['distance']}",
            fill="black"
        )

        draw_image(sheet, pair["path_a"], 15, y + 35)
        draw_image(sheet, pair["path_b"], 395, y + 35)

        draw.text(
            (15, y + 220),
            pair["split_a"],
            fill="black"
        )

        draw.text(
            (395, y + 220),
            pair["split_b"],
            fill="black"
        )

    output_path = OUTPUT_DIR / f"review_{sheet_index + 1:02d}.jpg"
    sheet.save(output_path, quality=90)

    print("Saved:", output_path)

print(f"\nGenerated {sheet_count} review sheets.")