import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tensorflow import keras


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ML" / "data" / "hybrid" / "val"

MANIFEST = ROOT / "ML" / "hybrid_manifest.csv"

MODEL_PATH = (
    ROOT / "ML" / "models"
    / "leaflens_mobilenetv2_hybrid_finetuned.keras"
)

LABELS_PATH = (
    ROOT / "ML" / "models" / "class_names_hybrid.json"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8


# ==================================================
# 2. LOAD MODEL AND CLASS NAMES
# ==================================================

print("\n====================================")
print("LEAFLENS-AI HYBRID MODEL EVALUATION")
print("====================================")

print("\nLoading fine-tuned hybrid model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = keras.models.load_model(MODEL_PATH)

with LABELS_PATH.open(
    "r",
    encoding="utf-8"
) as f:
    class_names = json.load(f)

print("Model:", MODEL_PATH.name)
print("Total classes:", len(class_names))


# ==================================================
# 3. LOAD VALIDATION DATASET
# ==================================================

print("\nLoading validation dataset...")

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if val_ds.class_names != class_names:
    raise ValueError(
        "Class names do not match the saved model labels!"
    )

print("Validation images:", len(val_ds.file_paths))


# ==================================================
# 4. LOAD MANIFEST AND IMAGE MODALITIES
# ==================================================

print("\nReading hybrid dataset manifest...")

with MANIFEST.open(
    newline="",
    encoding="utf-8"
) as f:
    records = list(csv.DictReader(f))

modality_lookup = {}

for row in records:

    if row["split"] != "val":
        continue

    image_path = str(
        Path(row["destination"]).resolve()
    )

    if image_path in modality_lookup:
        raise ValueError(
            f"Duplicate manifest path: {image_path}"
        )

    modality_lookup[image_path] = row["modality"]


image_paths = [
    str(Path(path).resolve())
    for path in val_ds.file_paths
]

if len(image_paths) != len(set(image_paths)):
    raise ValueError(
        "Duplicate validation image paths detected!"
    )

missing_paths = [
    path for path in image_paths
    if path not in modality_lookup
]

if missing_paths:
    raise ValueError(
        f"{len(missing_paths)} images are missing "
        "from the hybrid manifest!"
    )

print("Manifest verification successful.")


# ==================================================
# 5. EVALUATE MODEL USING KERAS
# ==================================================

print("\nEvaluating saved model...")

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\n====================================")
print("KERAS VALIDATION RESULTS")
print("====================================")

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: {val_loss:.4f}"
)


# ==================================================
# 6. COLLECT INDIVIDUAL PREDICTIONS
# ==================================================

print("\nCollecting individual predictions...")

predicted = []
actual = []

for images, labels in val_ds:

    probabilities = model(
        images,
        training=False
    ).numpy()

    predicted_indices = np.argmax(
        probabilities,
        axis=1
    )

    predicted.extend(
        predicted_indices.tolist()
    )

    actual.extend(
        labels.numpy().tolist()
    )

if len(predicted) != len(image_paths):
    raise ValueError(
        "Prediction count does not match image count!"
    )

if len(actual) != len(image_paths):
    raise ValueError(
        "Actual label count does not match image count!"
    )


# ==================================================
# 7. CALCULATE ACCURACY BY IMAGE TYPE
# ==================================================

results = defaultdict(lambda: [0, 0])

class_results = defaultdict(lambda: [0, 0])

for path, prediction, label in zip(
    image_paths,
    predicted,
    actual
):

    modality = modality_lookup[path]

    plant_name = class_names[label]

    correct = int(prediction == label)

    # Overall accuracy for leaf or plant images
    results[modality][0] += correct
    results[modality][1] += 1

    # Per-class accuracy for each image type
    class_results[
        (plant_name, modality)
    ][0] += correct

    class_results[
        (plant_name, modality)
    ][1] += 1


# ==================================================
# 8. DISPLAY LEAF AND PLANT ACCURACY
# ==================================================

print("\n====================================")
print("HYBRID MODEL VALIDATION RESULTS")
print("====================================")

for modality in ("leaf", "plant"):

    correct, total = results[modality]

    if total == 0:
        print(f"{modality.upper()}: No images")
        continue

    accuracy = (correct / total) * 100

    print(
        f"{modality.upper()} accuracy: "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )


total_correct = sum(
    values[0]
    for values in results.values()
)

total_images = sum(
    values[1]
    for values in results.values()
)

overall_accuracy = (
    total_correct / total_images
)

print(
    f"\nOVERALL accuracy: "
    f"{overall_accuracy * 100:.2f}% "
    f"({total_correct}/{total_images})"
)


# ==================================================
# 9. VERIFY ACCURACY CONSISTENCY
# ==================================================

if abs(overall_accuracy - val_accuracy) > 0.0001:
    raise ValueError(
        "Prediction-based accuracy does not match "
        "Keras model.evaluate() accuracy!"
    )

print("\nAccuracy consistency check: PASSED")


# ==================================================
# 10. LOWEST ACCURACY CLASSES
# ==================================================

print("\n====================================")
print("LOWEST ACCURACY CLASSES")
print("====================================")

for modality in ("leaf", "plant"):

    print(
        f"\n{modality.upper()} — "
        "10 lowest-performing classes:"
    )

    items = []

    for (
        plant_name,
        image_type
    ), (correct, total) in class_results.items():

        if image_type != modality:
            continue

        if total == 0:
            continue

        accuracy = correct / total

        items.append(
            (
                accuracy,
                plant_name,
                correct,
                total
            )
        )

    items.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    for (
        accuracy,
        plant_name,
        correct,
        total
    ) in items[:10]:

        print(
            f"{plant_name:25s} "
            f"{accuracy * 100:6.2f}% "
            f"({correct}/{total})"
        )


# ==================================================
# 11. OVERALL CONFUSION REPORT
# ==================================================

print("\n====================================")
print("TOP 20 MOST COMMON CLASS CONFUSIONS")
print("====================================")

confusions = Counter()

for actual_index, predicted_index in zip(
    actual,
    predicted
):

    if actual_index == predicted_index:
        continue

    actual_name = class_names[actual_index]
    predicted_name = class_names[predicted_index]

    confusions[
        (actual_name, predicted_name)
    ] += 1


if not confusions:
    print("No incorrect predictions found.")

else:
    for (
        actual_name,
        predicted_name
    ), count in confusions.most_common(20):

        print(
            f"{actual_name:25s} "
            f"-> {predicted_name:25s} "
            f": {count} images"
        )


# ==================================================
# 12. CONFUSIONS BY IMAGE TYPE
# ==================================================

print("\n====================================")
print("CONFUSIONS BY IMAGE TYPE")
print("====================================")

modality_confusions = {
    "leaf": Counter(),
    "plant": Counter(),
}

for path, actual_index, predicted_index in zip(
    image_paths,
    actual,
    predicted
):

    if actual_index == predicted_index:
        continue

    modality = modality_lookup[path]

    actual_name = class_names[actual_index]
    predicted_name = class_names[predicted_index]

    modality_confusions.setdefault(
        modality,
        Counter()
    )[
        (actual_name, predicted_name)
    ] += 1


for modality in ("leaf", "plant"):

    print(
        f"\n{modality.upper()} — "
        "Top 10 confusions:"
    )

    confusion_items = (
        modality_confusions[modality].most_common(10)
    )

    if not confusion_items:
        print("No incorrect predictions found.")
        continue

    for (
        actual_name,
        predicted_name
    ), count in confusion_items:

        print(
            f"{actual_name:25s} "
            f"-> {predicted_name:25s} "
            f": {count} images"
        )


# ==================================================
# 13. FINAL SUMMARY
# ==================================================

print("\n====================================")
print("EVALUATION COMPLETED")
print("====================================")

print("Model:", MODEL_PATH.name)
print("Classes:", len(class_names))

print(
    f"Validation images: {total_images}"
)

print(
    f"Overall accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: {val_loss:.4f}"
)

print("\nReserved test dataset was not used.")

from PIL import Image, ImageDraw
import math

TARGET_CLASSES = {"Pea", "Seethaashoka"}

samples = []

for path, actual_index, predicted_index in zip(
    image_paths,
    actual,
    predicted
):
    actual_name = class_names[actual_index]
    predicted_name = class_names[predicted_index]

    if actual_name in TARGET_CLASSES:
        samples.append(
            (path, actual_name, predicted_name)
        )

# Create a visual comparison sheet
THUMB_W = 220
THUMB_H = 200
CELL_W = 260
CELL_H = 270
COLS = 4

rows = math.ceil(len(samples) / COLS)

sheet = Image.new(
    "RGB",
    (COLS * CELL_W, rows * CELL_H),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (path, actual_name, predicted_name) in enumerate(samples):

    col = i % COLS
    row = i // COLS

    x = col * CELL_W
    y = row * CELL_H

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB_W, THUMB_H))
        sheet.paste(img, (x + 10, y + 10))

    correct = actual_name == predicted_name

    draw.text(
        (x + 10, y + 215),
        f"Actual: {actual_name}",
        fill="black"
    )

    draw.text(
        (x + 10, y + 235),
        f"Pred: {predicted_name}",
        fill="green" if correct else "red"
    )

OUTPUT = ROOT / "ML" / "hybrid_error_review.jpg"

sheet.save(OUTPUT, quality=95)

print("\nError review image saved:", OUTPUT)
print("Images reviewed:", len(samples))