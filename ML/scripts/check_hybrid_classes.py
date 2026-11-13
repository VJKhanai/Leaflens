from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LEAF_DIR = ROOT / "ML" / "data" / "grouped" / "train"

PLANT_DIR = Path(
    "/media/vijay/New Volume1/"
    "Indian Medicinal Leaves Image Datasets/"
    "Indian Medicinal Leaves Image Datasets/"
    "Medicinal plant dataset"
)

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

leaf_classes = {
    p.name for p in LEAF_DIR.iterdir()
    if p.is_dir()
}

matched = {}
unmatched = {}

for folder in sorted(PLANT_DIR.iterdir()):

    if not folder.is_dir():
        continue

    plant_name = folder.name
    target_name = MAPPING.get(plant_name, plant_name)

    image_count = sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if target_name in leaf_classes:
        matched[plant_name] = (target_name, image_count)
    else:
        unmatched[plant_name] = image_count

print("\n===== HYBRID DATASET ANALYSIS =====")

print("\nMATCHED PLANT CLASSES:")

matched_images = 0

for source, (target, count) in matched.items():
    print(f"{source} -> {target}: {count} images")
    matched_images += count

print("\nUNMATCHED PLANT CLASSES:")

for name, count in unmatched.items():
    print(f"{name}: {count} images")

print("\n===== SUMMARY =====")
print("Existing leaf classes:", len(leaf_classes))
print("Matched plant classes:", len(matched))
print("Unmatched plant classes:", len(unmatched))
print("Additional matched plant images:", matched_images)