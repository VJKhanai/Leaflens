
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "ML" / "models"
    / "leaflens_mobilenetv2_grouped_windows.keras"
)

LABELS_PATH = (
    ROOT / "ML" / "models"
    / "class_names_grouped_windows.json"
)

VAL_DIR = ROOT / "ML" / "data" / "grouped" / "val"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

with LABELS_PATH.open(encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
    interpolation="bilinear",
)

y_true = []
y_pred = []

for images, labels in dataset:

    probabilities = model.predict(
        images,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

accuracy = accuracy_score(y_true, y_pred)

print("\n===================================")
print("LEAFLENS-AI VALIDATION REPORT")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

print("\n--- TARGET PLANT PERFORMANCE ---")

for plant in ("Tulsi", "Ganike", "Kohlrabi", "Curry", "Neem"):

    index = class_names.index(plant)

    total = cm[index].sum()
    correct = cm[index, index]

    recall = (
        correct / total * 100
        if total else 0
    )

    print(
        f"{plant}: "
        f"{correct}/{total} correct "
        f"({recall:.2f}% recall)"
    )

print("\nValidation evaluation complete.")
