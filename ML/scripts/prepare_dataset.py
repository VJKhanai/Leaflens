
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "ML" / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_IMAGES = 30
SEED = 42

random.seed(SEED)

# Find the extracted dataset folder
matches = [
    p for p in ROOT.rglob("Medicinal Leaf dataset")
    if p.is_dir()
]

if len(matches) != 1:
    raise RuntimeError(
        "Expected exactly one 'Medicinal Leaf dataset' folder. "
        f"Found {len(matches)}. Check the dataset location."
    )

DATASET = matches[0]
print("Dataset:", DATASET)

# Inspect and collect valid image files
classes = {}

for folder in sorted(DATASET.iterdir()):
    if not folder.is_dir():
        continue

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) >= MIN_IMAGES:
        classes[folder.name] = images
    elif images:
        print(f"Skipping {folder.name}: only {len(images)} images")

if len(classes) < 2:
    raise RuntimeError("Not enough classes to train a classifier.")

# Recreate only the generated processed dataset
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

counts = {"train": 0, "val": 0, "test": 0}

for class_name, images in sorted(classes.items()):
    random.shuffle(images)

    n = len(images)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, files in splits.items():
        destination = OUTPUT / split / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, source in enumerate(files):
            target = destination / f"{index:04d}_{source.name}"
            shutil.copy2(source, target)
            counts[split] += 1

print("\nDataset preparation complete.")
print("Classes:", len(classes))
print("Split totals:", counts)
print("Output:", OUTPUT)
print("\nClasses included:")
for name, images in sorted(classes.items()):
    print(f"  {name}: {len(images)} images")
