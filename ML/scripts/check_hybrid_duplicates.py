import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"

def file_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()

with MANIFEST.open(encoding="utf-8", newline="") as f:
    records = list(csv.DictReader(f))

hash_groups = defaultdict(list)

print("Checking hybrid dataset for duplicate images...")

for record in records:
    path = Path(record["destination"])

    if not path.is_file():
        raise FileNotFoundError(path)

    hash_groups[file_hash(path)].append(record)

duplicates = []
cross_split = []
cross_class = []

for group in hash_groups.values():
    if len(group) < 2:
        continue

    duplicates.append(group)

    if len({r["split"] for r in group}) > 1:
        cross_split.append(group)

    if len({r["class"] for r in group}) > 1:
        cross_class.append(group)

print("\n===== DUPLICATE CHECK RESULTS =====")
print("Total images:", len(records))
print("Duplicate image groups:", len(duplicates))
print("Duplicates crossing splits:", len(cross_split))
print("Duplicates with conflicting classes:", len(cross_class))

if cross_split:
    print("\nFirst cross-split duplicate groups:")

    for group in cross_split[:10]:
        for r in group:
            print(r["split"], r["class"], r["destination"])
        print()

print("\nCheck completed.")