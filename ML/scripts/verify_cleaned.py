
from pathlib import Path
import hashlib
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "cleaned"

SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

hashes = defaultdict(list)
classes = {}
empty_folders = []

for split in SPLITS:
    split_dir = DATA_DIR / split

    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)

    class_folders = sorted(
        p for p in split_dir.iterdir() if p.is_dir()
    )

    classes[split] = {p.name for p in class_folders}
    count = 0

    for folder in class_folders:
        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        if not images:
            empty_folders.append((split, folder.name))

        for path in images:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest].append((split, path))
            count += 1

    print(f"{split}: {count} images, {len(class_folders)} classes")

cross_split = [
    paths for paths in hashes.values()
    if len({split for split, _ in paths}) > 1
]

same_classes = all(
    classes[split] == classes["train"]
    for split in SPLITS
)

print("\n--- VERIFICATION ---")
print("Same class folders across splits:", same_classes)
print("Empty class folders:", len(empty_folders))
print("Cross-split exact duplicate groups:", len(cross_split))

for split, name in empty_folders:
    print("EMPTY:", split, name)

if same_classes and not empty_folders and not cross_split:
    print("\nPASS: Basic cleaned dataset checks passed.")
else:
    print("\nCHECK REQUIRED: Review the issues above.")
