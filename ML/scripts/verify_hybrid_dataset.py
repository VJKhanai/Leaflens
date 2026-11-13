import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HYBRID_DIR = ROOT / "ML" / "data" / "hybrid"
MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

with MANIFEST.open(encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f))

print("\n===== HYBRID DATASET VERIFICATION =====")

print("Manifest images:", len(records))

counts = Counter()
modalities = Counter()
class_splits = defaultdict(set)
group_splits = defaultdict(set)

missing = []
duplicate_paths = set()
seen_paths = set()

for record in records:
    plant = record["class"]
    split = record["split"]
    modality = record["modality"]

    path = Path(record["destination"])

    if not path.is_file():
        missing.append(str(path))

    if str(path) in seen_paths:
        duplicate_paths.add(str(path))

    seen_paths.add(str(path))

    counts[split] += 1
    modalities[(split, modality)] += 1
    class_splits[plant].add(split)

    group_key = (
        plant,
        modality,
        record["group_id"],
    )

    group_splits[group_key].add(split)

print("\nSplit counts:")
for split in SPLITS:
    print(f"{split}: {counts[split]}")

print("\nImage types:")
for split in SPLITS:
    print(
        f"{split}: "
        f"leaf={modalities[(split, 'leaf')]}, "
        f"plant={modalities[(split, 'plant')]}"
    )

print("\nClasses:", len(class_splits))
print("Missing files:", len(missing))
print("Duplicate destinations:", len(duplicate_paths))

incomplete_classes = [
    plant for plant, splits in class_splits.items()
    if set(SPLITS) != splits
]

cross_split_groups = [
    key for key, splits in group_splits.items()
    if len(splits) > 1
]

print("Classes missing a split:", len(incomplete_classes))
print("Groups crossing splits:", len(cross_split_groups))

if incomplete_classes:
    print("Incomplete classes:", incomplete_classes)

if cross_split_groups:
    print(
        "First cross-split groups:",
        cross_split_groups[:10]
    )

print("\nVerification complete.")