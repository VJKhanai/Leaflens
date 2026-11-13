import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "ML/models/leaflens_mobilenetv2_hybrid.keras"
LABELS_PATH = ROOT / "ML/models/class_names_hybrid.json"
MANIFEST_PATH = ROOT / "ML/hybrid_manifest.csv"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

print("Loading hybrid model...")
model = keras.models.load_model(MODEL_PATH)

with LABELS_PATH.open(encoding="utf-8") as f:
    class_names = json.load(f)

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

with MANIFEST_PATH.open(
    encoding="utf-8",
    newline=""
) as f:
    records = [
        row for row in csv.DictReader(f)
        if row["split"] == "val"
    ]

results = defaultdict(lambda: [0, 0])
class_results = defaultdict(lambda: [0, 0])

print(f"\nEvaluating {len(records)} validation images...")

for start in range(0, len(records), BATCH_SIZE):
    batch = records[start:start + BATCH_SIZE]

    images = []

    for row in batch:
        path = Path(row["destination"])

        image = keras.utils.load_img(
            path,
            target_size=IMAGE_SIZE,
            color_mode="rgb",
            interpolation="bilinear",
        )

        images.append(
            keras.utils.img_to_array(image)
        )

    images = np.stack(images).astype("float32")

    # The model already performs MobileNetV2 preprocessing.
    predictions = model.predict(images, verbose=0)
    predicted_indices = np.argmax(predictions, axis=1)

    for row, predicted_index in zip(batch, predicted_indices):
        actual_index = class_to_index[row["class"]]
        correct = int(predicted_index == actual_index)

        modality = row["modality"]
        plant_name = row["class"]

        results[modality][0] += correct
        results[modality][1] += 1

        class_results[(plant_name, modality)][0] += correct
        class_results[(plant_name, modality)][1] += 1


def display_accuracy(correct, total):
    if total == 0:
        return "N/A"

    return f"{correct / total * 100:.2f}%"


print("\n====================================")
print("HYBRID MODEL VALIDATION RESULTS")
print("====================================")

total_correct = 0
total_images = 0

for modality in ("leaf", "plant"):
    correct, total = results[modality]

    total_correct += correct
    total_images += total

    print(
        f"{modality.upper()} accuracy: "
        f"{display_accuracy(correct, total)} "
        f"({correct}/{total})"
    )

print(
    f"\nOVERALL accuracy: "
    f"{display_accuracy(total_correct, total_images)}"
)

print("\n====================================")
print("LOWEST ACCURACY CLASSES BY MODALITY")
print("====================================")

for modality in ("leaf", "plant"):

    items = []

    for (plant_name, image_type), (correct, total) in class_results.items():
        if image_type != modality:
            continue

        items.append(
            (correct / total, plant_name, correct, total)
        )

    items.sort()

    print(f"\n{modality.upper()} — lowest 10 classes:")

    for accuracy, name, correct, total in items[:10]:
        print(
            f"{name:20s} "
            f"{accuracy * 100:6.2f}% "
            f"({correct}/{total})"
        )

print("\nEvaluation completed.")
print("Reserved test images were not used.")