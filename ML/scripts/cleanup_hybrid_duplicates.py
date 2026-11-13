import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"
NEAR_DUPLICATES = ROOT / "ML" / "hybrid_near_duplicates.csv"

BACKUP = ROOT / "ML" / "hybrid_manifest_before_cleanup.csv"
QUARANTINE = ROOT / "ML" / "hybrid_quarantine"

# Only the cross-dataset Tulsi pair is selected.
SELECTED_PAIRS = {8}

with MANIFEST.open(newline="", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

with NEAR_DUPLICATES.open(newline="", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

if BACKUP.exists():
    raise SystemExit("Backup already exists. No changes made.")

# Identify plant-side images belonging to selected pairs.
remove_paths = set()

for index, pair in enumerate(pairs, 1):
    if index not in SELECTED_PAIRS:
        continue

    for side in ("1", "2"):
        if pair[f"modality{side}"] == "plant":
            remove_paths.add(pair[f"image{side}"])

if not remove_paths:
    raise SystemExit("No plant images selected. No changes made.")

# Verify all selected paths exist in the manifest and on disk.
manifest_paths = {r["destination"] for r in records}

for path in remove_paths:
    if path not in manifest_paths:
        raise SystemExit(f"Not in manifest: {path}")

    if not Path(path).is_file():
        raise SystemExit(f"Missing image: {path}")

BACKUP.write_bytes(MANIFEST.read_bytes())

remaining = []

for record in records:
    path = record["destination"]

    if path in remove_paths:
        source = Path(path)

        destination = (
            QUARANTINE
            / record["split"]
            / record["class"]
            / source.name
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, destination)

        print("Quarantined:", source.name)
    else:
        remaining.append(record)

with MANIFEST.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(remaining)

print("\nCleanup completed.")
print("Removed from hybrid:", len(remove_paths))
print("Remaining images:", len(remaining))
print("Quarantine:", QUARANTINE)