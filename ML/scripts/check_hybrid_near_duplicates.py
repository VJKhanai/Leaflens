import csv
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"

OUTPUT = ROOT / "ML" / "hybrid_near_duplicates.csv"

THRESHOLD = 4

with MANIFEST.open(
    encoding="utf-8",
    newline=""
) as f:
    records = list(csv.DictReader(f))

classes = defaultdict(list)

print("Calculating perceptual hashes...")

for record in records:

    path = Path(record["destination"])

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((256, 256))
            phash = imagehash.phash(img)

        classes[record["class"]].append(
            (record, phash)
        )

    except Exception as error:
        print("Error:", path, error)


candidates = []

print("Checking images across different splits...")

for plant, images in classes.items():

    for i in range(len(images)):
        record1, hash1 = images[i]

        for j in range(i + 1, len(images)):
            record2, hash2 = images[j]

            if record1["split"] == record2["split"]:
                continue

            distance = hash1 - hash2

            if distance <= THRESHOLD:
                candidates.append({
                    "class": plant,
                    "split1": record1["split"],
                    "split2": record2["split"],
                    "modality1": record1["modality"],
                    "modality2": record2["modality"],
                    "distance": distance,
                    "image1": record1["destination"],
                    "image2": record2["destination"],
                })


with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "split1",
            "split2",
            "modality1",
            "modality2",
            "distance",
            "image1",
            "image2"
        ]
    )

    writer.writeheader()
    writer.writerows(candidates)


print("\n===== HYBRID NEAR-DUPLICATE CHECK =====")
print("Total images:", len(records))
print("Plant classes:", len(classes))
print("Near-duplicate candidate pairs:", len(candidates))
print("Saved report:", OUTPUT)

print("\nFirst 10 candidates:")

for item in candidates[:10]:
    print(
        item["class"],
        item["split1"],
        item["split2"],
        item["distance"]
    )