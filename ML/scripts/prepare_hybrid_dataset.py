import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

LEAF_DIR = ROOT / "ML" / "data" / "grouped"

PLANT_DIR = Path(
    "/media/vijay/New Volume1/"
    "Indian Medicinal Leaves Image Datasets/"
    "Indian Medicinal Leaves Image Datasets/"
    "Medicinal plant dataset"
)

OUTPUT = ROOT / "ML" / "data" / "hybrid"
MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"

MAPPING = {
    "Tulasi": "Tulsi",
    "Curry_Leaf": "Curry",
    "Gauva": "Guava",
    "Pappaya": "Papaya",
    "Amruta_Balli": "Amruthaballi",
    "Brahmi": "Bhrami",
    "Doddapatre": "Doddpathre",
    "Pomegranate": "Pomoegranate",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
SEED = 42

random.seed(SEED)

if OUTPUT.exists():
    raise SystemExit(
        "Hybrid folder already exists. "
        "No files changed. Rename it before regenerating."
    )

classes = sorted(
    p.name for p in (LEAF_DIR / "train").iterdir()
    if p.is_dir()
)

class_set = set(classes)

records = []

def copy_image(source, split, plant, modality, group_id):
    target_dir = OUTPUT / split / plant
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{modality}_{group_id}_{source.name}"
    target = target_dir / filename

    shutil.copy2(source, target)

    records.append({
        "class": plant,
        "modality": modality,
        "split": split,
        "group_id": group_id,
        "source": str(source),
        "destination": str(target),
    })


print("Copying existing leaf dataset...")

for split in SPLITS:
    for plant in classes:
        folder = LEAF_DIR / split / plant

        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                # Existing filenames already include grouped IDs.
                # Keep their original split unchanged.
                group_id = path.name.split("_")[1]

                copy_image(
                    path, split, plant,
                    "leaf", group_id
                )


print("Processing whole-plant images...")

for plant_folder in sorted(PLANT_DIR.iterdir()):

    if not plant_folder.is_dir():
        continue

    plant = MAPPING.get(
        plant_folder.name,
        plant_folder.name
    )

    if plant not in class_set:
        print("Skipping unmatched:", plant_folder.name)
        continue

    paths = sorted(
        p for p in plant_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    # Compute perceptual hashes.
    hashes = []

    for path in paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((256, 256))
                hashes.append(imagehash.phash(image))
        except Exception as error:
            raise RuntimeError(
                f"Cannot process {path}: {error}"
            )

    # Group visually similar images.
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a = find(a)
        b = find(b)

        if a != b:
            parent[b] = a

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[i] - hashes[j] <= 6:
                union(i, j)

    groups = defaultdict(list)

    for i, path in enumerate(paths):
        groups[find(i)].append(path)

    grouped_images = list(groups.values())
    random.shuffle(grouped_images)

    total = sum(len(group) for group in grouped_images)

    targets = {
        "train": round(total * 0.70),
        "val": round(total * 0.15),
    }

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for group_number, group in enumerate(grouped_images):

        if counts["train"] < targets["train"]:
            split = "train"

        elif counts["val"] < targets["val"]:
            split = "val"

        else:
            split = "test"

        group_id = f"{plant_folder.name}_{group_number:04d}"

        for path in group:
            copy_image(
                path,
                split,
                plant,
                "plant",
                group_id
            )

            counts[split] += 1

    print(
        f"{plant_folder.name} -> {plant}: "
        f"{counts}"
    )


with MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "class",
            "modality",
            "split",
            "group_id",
            "source",
            "destination"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


print("\n================================")
print("HYBRID DATASET PREPARED")
print("================================")

print("Total images:", len(records))
print("Classes:", len(classes))

for split in SPLITS:
    count = sum(
        1 for record in records
        if record["split"] == split
    )

    print(f"{split}: {count}")

print("\nSaved:", OUTPUT)
print("Manifest:", MANIFEST)