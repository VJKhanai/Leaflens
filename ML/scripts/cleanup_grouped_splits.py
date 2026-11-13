
import argparse
import csv
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "ML" / "grouped_manifest.csv"
PAIRS = ROOT / "ML" / "groups_to_review.csv"
DATA = ROOT / "ML" / "data" / "grouped"

SPLITS = ("train", "val", "test")

# Pair numbers refer to the original review sheets.
# These visually questionable pairs are excluded.
EXCLUDE_PAIR_NUMBERS = {1, 3, 5, 29, 64, 69, 86}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(apply_changes=False):

    if not MANIFEST.is_file():
        raise SystemExit("Missing grouped_manifest.csv")

    if not PAIRS.is_file():
        raise SystemExit("Missing groups_to_review.csv")

    rows = read_csv(MANIFEST)
    pairs = read_csv(PAIRS)

    if len(rows) != 6843:
        raise SystemExit(
            f"Expected 6843 images, found {len(rows)}. "
            "No changes made."
        )

    required = {"class", "group_id", "split", "destination"}

    if not rows or not required.issubset(rows[0]):
        raise SystemExit("Manifest columns are invalid.")

    if not pairs or not {
        "Class", "GroupA", "GroupB"
    }.issubset(pairs[0]):
        raise SystemExit("Review CSV columns are invalid.")

    group_rows = defaultdict(list)
    classes = defaultdict(set)

    for record in rows:

        if record["split"] not in SPLITS:
            raise SystemExit("Invalid split found.")

        key = (record["class"], record["group_id"])

        group_rows[key].append(record)
        classes[record["class"]].add(key)

        if not Path(record["destination"]).is_file():
            raise SystemExit(
                f"Missing image: {record['destination']}"
            )

    if len(classes) != 78:
        raise SystemExit(
            f"Expected 78 classes, found {len(classes)}"
        )

    # Union-find: connect related groups.
    parent = {key: key for key in group_rows}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        x = find(a)
        y = find(b)

        if x != y:
            parent[max(x, y)] = min(x, y)

    accepted = 0

    for index, pair in enumerate(pairs, start=1):

        if index in EXCLUDE_PAIR_NUMBERS:
            continue

        a = (pair["Class"], pair["GroupA"])
        b = (pair["Class"], pair["GroupB"])

        if a not in parent or b not in parent:
            raise SystemExit(
                f"Unknown group in pair {index}: {a}, {b}"
            )

        union(a, b)
        accepted += 1

    components = defaultdict(list)

    for key in group_rows:
        components[find(key)].append(key)

    by_class = defaultdict(list)

    for component in components.values():

        plant = component[0][0]

        records = [
            record
            for key in component
            for record in group_rows[key]
        ]

        original = Counter(
            record["split"] for record in records
        )

        canonical = min(
            key[1] for key in component
        )

        by_class[plant].append({
            "records": records,
            "original": original,
            "canonical": canonical,
            "plant": plant,
            "split": None
        })

    # Assign each connected component to one split.
    # Prefer the split already containing most images.

    for plant, component_list in by_class.items():

        if len(component_list) < 3:
            raise SystemExit(
                f"{plant} has fewer than 3 independent groups. "
                "Cannot preserve all three splits."
            )

        for component in component_list:

            component["split"] = sorted(
                SPLITS,
                key=lambda s: (
                    -component["original"][s],
                    SPLITS.index(s)
                )
            )[0]

        # Ensure train, validation and test contain
        # at least one component for each plant.

        for desired in SPLITS:

            if any(
                c["split"] == desired
                for c in component_list
            ):
                continue

            split_counts = Counter(
                c["split"] for c in component_list
            )

            candidates = [
                c for c in component_list
                if split_counts[c["split"]] > 1
            ]

            if not candidates:
                raise SystemExit(
                    f"Cannot populate {plant}/{desired}"
                )

            chosen = min(
                candidates,
                key=lambda c: (
                    c["original"][c["split"]]
                    - c["original"][desired],
                    len(c["records"]),
                    c["canonical"]
                )
            )

            chosen["split"] = desired

    planned_moves = []
    new_rows = []

    for component_list in by_class.values():

        for component in component_list:

            for record in component["records"]:

                old = Path(record["destination"])

                target = (
                    DATA
                    / component["split"]
                    / record["class"]
                    / old.name
                )

                if old != target:
                    planned_moves.append((old, target))

                updated = dict(record)

                updated["split"] = component["split"]
                updated["group_id"] = component["canonical"]
                updated["destination"] = str(target)

                new_rows.append(updated)

    # Validate before moving anything.

    old_paths = [
        str(Path(r["destination"]).resolve()).lower()
        for r in rows
    ]

    new_paths = [
        str(Path(r["destination"]).resolve()).lower()
        for r in new_rows
    ]

    if (
        len(set(old_paths)) != len(rows)
        or len(set(new_paths)) != len(rows)
    ):
        raise SystemExit(
            "Duplicate source or destination path found."
        )

    old_set = set(old_paths)

    for old, target in planned_moves:

        if (
            target.exists()
            and str(target.resolve()).lower() not in old_set
        ):
            raise SystemExit(
                f"Destination already exists: {target}"
            )

    for plant in by_class:

        for split in SPLITS:

            if not any(
                r["class"] == plant
                and r["split"] == split
                for r in new_rows
            ):
                raise SystemExit(
                    f"Empty class after cleanup: {plant}/{split}"
                )

    final_counts = Counter(
        r["split"] for r in new_rows
    )

    print("\n--- GROUPED DATASET CLEANUP ---")
    print("Total images:", len(rows))
    print("Plant classes:", len(by_class))
    print("Candidate pairs:", len(pairs))
    print("Accepted candidate pairs:", accepted)
    print("Connected components:", len(components))
    print("Proposed image moves:", len(planned_moves))
    print("Final split sizes:", dict(final_counts))

    if not apply_changes:

        print("\nDRY RUN COMPLETE.")
        print("No files have been modified.")
        print(
            "Run with --apply to make the changes."
        )
        return

    # Back up manifest before applying changes.

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup = MANIFEST.with_name(
        f"grouped_manifest_backup_{stamp}.csv"
    )

    shutil.copy2(MANIFEST, backup)

    print("\nManifest backup:", backup)

    staging = DATA.parent / f"_grouped_move_stage_{stamp}"

    staging.mkdir(parents=True, exist_ok=False)

    staged = []
    completed = []

    try:

        # Move changed images to temporary staging.
        for index, (old, target) in enumerate(planned_moves):

            temp = staging / f"{index:06d}{old.suffix}"

            shutil.move(str(old), str(temp))

            staged.append((old, temp, target))

        # Move staged images to their final split.
        for old, temp, target in staged:

            target.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.move(str(temp), str(target))

            completed.append((old, target))

        write_csv(
            MANIFEST,
            new_rows,
            list(rows[0].keys())
        )

    except Exception:

        print(
            "Error applying changes. "
            "Attempting to restore original files."
        )

        for old, target in reversed(completed):

            if target.exists():

                old.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.move(str(target), str(old))

        for old, temp, target in reversed(staged):

            if temp.exists():

                old.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.move(str(temp), str(old))

        shutil.copy2(backup, MANIFEST)

        raise

    finally:

        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()

    # Verify results.

    checked_rows = read_csv(MANIFEST)

    assert len(checked_rows) == 6843

    assert all(
        Path(r["destination"]).is_file()
        for r in checked_rows
    )

    group_splits = defaultdict(set)

    for record in checked_rows:

        key = (
            record["class"],
            record["group_id"]
        )

        group_splits[key].add(record["split"])

    assert all(
        len(splits) == 1
        for splits in group_splits.values()
    )

    for plant in by_class:

        for split in SPLITS:

            assert any(
                r["class"] == plant
                and r["split"] == split
                for r in checked_rows
            )

    print("\nCLEANUP SUCCESSFUL!")
    print("Images verified:", len(checked_rows))
    print("Classes verified:", len(by_class))
    print("Groups are contained within one split.")
    print("Manifest updated successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes instead of previewing"
    )

    args = parser.parse_args()

    main(apply_changes=args.apply)
