
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "ML" / "data" / "processed"
DEST = ROOT / "ML" / "data" / "cleaned"

SPLITS = ["test", "val", "train"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

if DEST.exists():
    raise FileExistsError(
        f"Destination already exists: {DEST}. "
        "Inspect it before creating another cleaned dataset."
    )

hashes = defaultdict(list)

for split in SPLITS:
    for path in sorted((SOURCE / split).rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest].append((split, path))

# Preserve test first, validation second, training last.
# This prevents identical files appearing in multiple splits.
priority = {split: i for i, split in enumerate(SPLITS)}

kept = defaultdict(int)
removed = defaultdict(int)
conflicts = []

for digest, files in hashes.items():
    files.sort(key=lambda item: (
        priority[item[0]],
        str(item[1])
    ))

    keep_split, keep_path = files[0]
    target = DEST / keep_split / keep_path.relative_to(
        SOURCE / keep_split
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(keep_path, target)
    kept[keep_split] += 1

    for split, path in files[1:]:
        removed[split] += 1

        if path.parent.name != keep_path.parent.name:
            conflicts.append((keep_path, path))

# Ensure all original class folders exist, including empty ones.
for split in SPLITS:
    for folder in (SOURCE / split).iterdir():
        if folder.is_dir():
            (DEST / split / folder.name).mkdir(
                parents=True, exist_ok=True
            )

print("\n--- CLEANED DATASET SUMMARY ---")

for split in ["train", "val", "test"]:
    print(
        f"{split}: kept={kept[split]}, "
        f"duplicates skipped={removed[split]}"
    )

print("Conflicting class labels:", len(conflicts))

for first, second in conflicts[:20]:
    print("CONFLICT:", first, "<->", second)

print("\nCleaned dataset:", DEST)
print("Original dataset was not modified.")
