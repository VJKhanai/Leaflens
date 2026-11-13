import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "ML" / "hybrid_near_duplicates.csv"

OUTPUT = ROOT / "ML" / "hybrid_duplicate_review.jpg"

with CSV_PATH.open(encoding="utf-8", newline="") as f:
    pairs = list(csv.DictReader(f))

WIDTH = 1000
IMAGE_WIDTH = 460
IMAGE_HEIGHT = 260
ROW_HEIGHT = 330

canvas = Image.new(
    "RGB",
    (WIDTH, max(1, len(pairs)) * ROW_HEIGHT),
    "white"
)

draw = ImageDraw.Draw(canvas)

for i, pair in enumerate(pairs):

    y = i * ROW_HEIGHT

    title = (
        f"PAIR {i+1} | {pair['class']} | "
        f"pHash distance: {pair['distance']}"
    )

    draw.text((20, y + 10), title, fill="black")

    paths = [
        pair["image1"],
        pair["image2"]
    ]

    labels = [
        f"{pair['split1']} / {pair['modality1']}",
        f"{pair['split2']} / {pair['modality2']}"
    ]

    for j, path in enumerate(paths):

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT))

            x = 20 + j * 500
            image_y = y + 50

            canvas.paste(img, (x, image_y))

        draw.text(
            (x, y + 315),
            labels[j],
            fill="black"
        )

canvas.save(OUTPUT, quality=92)

print("\nComparison image created successfully!")
print("Total pairs:", len(pairs))
print("Saved:", OUTPUT)