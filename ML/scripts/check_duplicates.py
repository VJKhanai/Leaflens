
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "processed"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")

hashes = defaultdict(list)

for split in SPLITS:
    folder = DATA_DIR / split

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append((split, path))

duplicate_groups = [
    files for files in hashes.values()
    if len(files) > 1
]

cross_split_groups = [
    files for files in duplicate_groups
    if len({split for split, _ in files}) > 1
]

print("\n--- DUPLICATE IMAGE CHECK ---")
print("Total images:", sum(len(v) for v in hashes.values()))
print("Exact duplicate groups:", len(duplicate_groups))
print("Duplicate groups across splits:", len(cross_split_groups))

for group in cross_split_groups[:20]:
    print("\nDuplicate group:")
    for split, path in group:
        print(f"  [{split}] {path}")

print("\nCheck complete.")
