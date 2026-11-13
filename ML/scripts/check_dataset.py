
from pathlib import Path
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "processed"

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

total = 0
corrupted = []

for split in ["train", "val", "test"]:
    folder = DATA_DIR / split

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        total += 1

        try:
            with Image.open(path) as image:
                image.verify()

            # Verify that image pixels can actually be decoded.
            with Image.open(path) as image:
                image.load()

        except (OSError, ValueError, UnidentifiedImageError) as exc:
            corrupted.append((str(path), str(exc)))

print("\n--- DATASET CHECK ---")
print("Total images checked:", total)
print("Problematic images:", len(corrupted))

for path, error in corrupted:
    print(f"\nFile: {path}")
    print(f"Error: {error}")

print("\nCheck complete.")
