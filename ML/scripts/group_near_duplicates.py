
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REPORT = ROOT / "ML" / "near_duplicates_report.csv"
OUTPUT = ROOT / "ML" / "near_duplicate_groups.csv"

parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a

image_info = {}

with REPORT.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        a = row["image_a"]
        b = row["image_b"]

        union(a, b)

        image_info[a] = (
            row["plant"], row["split_a"]
        )
        image_info[b] = (
            row["plant"], row["split_b"]
        )

groups = defaultdict(list)

for image in image_info:
    groups[find(image)].append(image)

print("\n--- NEAR-DUPLICATE GROUP REPORT ---")
print("Candidate image groups:", len(groups))
print("Images involved:", len(image_info))

cross_split_groups = 0

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "group_id", "plant", "split", "image_path"
    ])

    for group_id, images in enumerate(groups.values(), 1):
        splits = {
            image_info[image][1]
            for image in images
        }

        if len(splits) > 1:
            cross_split_groups += 1

        for image in images:
            plant, split = image_info[image]
            writer.writerow([
                group_id, plant, split, image
            ])

print("Groups spanning multiple splits:", cross_split_groups)
print("Group report saved:", OUTPUT)
print("No dataset files were modified.")
