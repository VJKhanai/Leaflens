import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

HYBRID_MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"
GROUPED_MANIFEST = ROOT / "ML" / "grouped_manifest.csv"


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def filename_from_path(path):
    # Works with both Windows and Ubuntu paths
    return path.replace("\\", "/").split("/")[-1]


hybrid_records = read_csv(HYBRID_MANIFEST)
grouped_records = read_csv(GROUPED_MANIFEST)

# Create lookup using split, class and filename.
group_lookup = {}

for record in grouped_records:
    key = (
        record["split"],
        record["class"],
        filename_from_path(record["destination"])
    )

    if key in group_lookup:
        raise ValueError(f"Duplicate mapping: {key}")

    group_lookup[key] = record


updated = 0
missing = []

for record in hybrid_records:

    if record["modality"] != "leaf":
        continue

    source_key = (
        record["split"],
        record["class"],
        filename_from_path(record["source"])
    )

    original = group_lookup.get(source_key)

    if original is None:
        missing.append(record["source"])
        continue

    record["group_id"] = original["group_id"]
    updated += 1


if missing:
    print("Could not match these images:")

    for path in missing[:10]:
        print(path)

    raise SystemExit(
        f"Missing mappings: {len(missing)}. "
        "Manifest was not changed."
    )


# Create backup before updating.
backup_path = HYBRID_MANIFEST.with_name(
    "hybrid_manifest_before_group_fix.csv"
)

if backup_path.exists():
    raise SystemExit(
        f"Backup already exists: {backup_path}"
    )

backup_path.write_bytes(
    HYBRID_MANIFEST.read_bytes()
)

with HYBRID_MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=hybrid_records[0].keys()
    )

    writer.writeheader()
    writer.writerows(hybrid_records)


print("\nHYBRID GROUP IDS UPDATED")
print("Leaf records corrected:", updated)
print("Backup:", backup_path)
print("Hybrid manifest:", HYBRID_MANIFEST)