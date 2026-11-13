import csv
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ML/hybrid_manifest.csv"

with MANIFEST.open(newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

data = defaultdict(lambda: defaultdict(set))
counts = defaultdict(int)

for row in records:
    if row["modality"] != "leaf":
        continue

    plant = row["class"]
    split = row["split"]
    filename = Path(row["destination"]).name

    # Match dates embedded in filenames, e.g. IMG_20190909
    match = re.search(r"(20\d{6})", filename)

    if match:
        capture_date = match.group(1)
        data[plant][split].add(capture_date)

    counts[plant] += 1

print("\n===== CAPTURE DATE AUDIT =====")

for plant in sorted(counts):
    train_dates = data[plant]["train"]
    val_dates = data[plant]["val"]
    test_dates = data[plant]["test"]

    all_dates = train_dates | val_dates | test_dates

    overlaps = (
        (train_dates & val_dates)
        | (train_dates & test_dates)
        | (val_dates & test_dates)
    )

    if overlaps:
        print(
            f"{plant:25s} "
            f"images={counts[plant]:4d} "
            f"dates={len(all_dates):2d} "
            f"shared_dates={sorted(overlaps)}"
        )

print("\nAudit completed.")
print("Note: Missing filename dates cannot be assessed.")