import csv
import hashlib
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "ML" / "data" / "cleaned"
DEST = ROOT / "ML" / "data" / "grouped"
MANIFEST = ROOT / "ML" / "grouped_manifest.csv"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

rng = random.Random(SEED)


def capture_time(path):
    """Extract a timestamp from common dataset filenames."""
    match = re.search(r"(\d{8})_(\d{6})", path.name)

    if not match:
        return None

    try:
        return datetime.strptime(
            match.group(1) + match.group(2),
            "%Y%m%d%H%M%S"
        )
    except ValueError:
        return None


def file_hash(path):
    h = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


if DEST.exists():
    raise SystemExit(
        f"{DEST} already exists. "
        "Move it aside before creating a new dataset."
    )

classes = sorted({
    folder.name
    for split in ("train", "val", "test")
    for folder in (SOURCE / split).iterdir()
    if folder.is_dir()
})

records = []

for plant in classes:
    paths = sorted(
        p
        for split in ("train", "val", "test")
        for p in (SOURCE / split / plant).glob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    # Connect exact duplicates and images with nearby capture times.
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = {}
    timestamps = []

    for i, path in enumerate(paths):
        digest = file_hash(path)

        if digest in hashes:
            union(i, hashes[digest])
        else:
            hashes[digest] = i

        stamp = capture_time(path)
        if stamp is not None:
            timestamps.append((stamp, i))

    timestamps.sort()

    for (previous_time, previous_i), (current_time, current_i) in zip(
        timestamps, timestamps[1:]
    ):
        seconds = (current_time - previous_time).total_seconds()

        if seconds <= 10:
            union(previous_i, current_i)

    groups = defaultdict(list)

    for i, path in enumerate(paths):
        groups[find(i)].append(path)

    grouped_images = list(groups.values())
    rng.shuffle(grouped_images)

    target_train = round(len(paths) * RATIOS[0])
    target_val = round(len(paths) * RATIOS[1])

    counts = {"train": 0, "val": 0, "test": 0}

    for group_index, group in enumerate(grouped_images):
        if counts["train"] < target_train:
            split = "train"
        elif counts["val"] < target_val:
            split = "val"
        else:
            split = "test"

        group_id = f"{plant}_{group_index:04d}"

        for path in group:
            output_folder = DEST / split / plant
            output_folder.mkdir(parents=True, exist_ok=True)

            # Preserve the filename and avoid collisions.
            output_name = f"{group_id}_{path.name}"
            destination = output_folder / output_name

            shutil.copy2(path, destination)

            records.append({
                "class": plant,
                "group_id": group_id,
                "split": split,
                "source": str(path),
                "destination": str(destination)
            })

            counts[split] += 1

    print(f"{plant}: {counts}")

with MANIFEST.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "group_id",
            "split",
            "source",
            "destination"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print("\nGrouped dataset created successfully.")
print("Images:", len(records))
print("Classes:", len(classes))
print("Saved to:", DEST)
print("Manifest:", MANIFEST)