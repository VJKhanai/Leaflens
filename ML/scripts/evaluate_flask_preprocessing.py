
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ML/models/leaflens_mobilenetv2.keras"
LABELS_PATH = ROOT / "ML/models/class_names.json"
TEST_DIR = ROOT / "ML/data/processed/test"

with open(LABELS_PATH, encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda item: item[1])
    ]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

y_true = []
y_pred = []

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for class_index, class_name in enumerate(class_names):
    folder = TEST_DIR / class_name

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in extensions:
            continue

        with Image.open(path) as original:
            image = original.convert("RGB")
            image_array = np.asarray(image, dtype=np.float32)

        image_tensor = tf.image.resize(
            image_array,
            (160, 160),
            method="bilinear"
        )

        batch = tf.expand_dims(image_tensor, axis=0)

        probabilities = model.predict(batch, verbose=0)[0]
        prediction = int(np.argmax(probabilities))

        y_true.append(class_index)
        y_pred.append(prediction)

accuracy = accuracy_score(y_true, y_pred) * 100

print("\n--- FLASK PREPROCESSING EVALUATION ---")
print("Total images:", len(y_true))
print(f"Overall accuracy: {accuracy:.2f}%")

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))
