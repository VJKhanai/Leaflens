
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "ML" / "models"
VAL_DIR = ROOT / "ML" / "data" / "grouped" / "val"

with open(MODEL_DIR / "class_names.json", encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda x: x[1])
    ]

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    class_names=class_names,
    image_size=(160, 160),
    batch_size=8,
    shuffle=False,
)

y_true = np.concatenate([
    labels.numpy() for _, labels in val_ds
])

models = {
    "Grouped with augmentation": (
        MODEL_DIR / "leaflens_mobilenetv2_grouped.keras"
    ),
    "Grouped without augmentation": (
        MODEL_DIR / "leaflens_mobilenetv2_grouped_baseline.keras"
    ),
}

print("\n--- VALIDATION MODEL COMPARISON ---")

for name, path in models.items():
    model = tf.keras.models.load_model(
        path, compile=False
    )

    probabilities = model.predict(val_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, predictions) * 100

    tulsi_index = class_names.index("Tulsi")
    tulsi_mask = y_true == tulsi_index

    tulsi_accuracy = (
        np.mean(predictions[tulsi_mask] == tulsi_index) * 100
        if np.any(tulsi_mask) else float("nan")
    )

    print(f"\n{name} model:")
    print(f"Overall validation accuracy: {accuracy:.2f}%")
    print(f"Tulsi validation recall: {tulsi_accuracy:.2f}%")

print("\nComparison complete.")
