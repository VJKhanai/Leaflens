
import json
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ML/models/leaflens_mobilenetv2.keras"
LABELS_PATH = ROOT / "ML/models/class_names.json"
TEST_DIR = ROOT / "ML/data/processed/test"

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda item: item[1])
    ]

tulsi_dir = next(
    (p for p in TEST_DIR.iterdir()
     if p.is_dir() and p.name.lower() == "tulsi"),
    None
)

if tulsi_dir is None:
    raise FileNotFoundError(f"Tulsi folder not found in {TEST_DIR}")

results = Counter()
correct = 0
total = 0

for path in sorted(tulsi_dir.iterdir()):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue

    try:
        image = tf.keras.utils.load_img(
    path,
    target_size=(160, 160),
    color_mode="rgb",
    interpolation="bilinear"
)
        array = tf.keras.utils.img_to_array(image)
        array = np.expand_dims(array, axis=0)

        # The model already contains MobileNetV2 preprocessing.
        probabilities = model.predict(array, verbose=0)[0]
        top_index = int(np.argmax(probabilities))
        predicted = class_names[top_index]
        confidence = float(probabilities[top_index]) * 100

        results[predicted] += 1
        correct += predicted.lower() == "tulsi"
        total += 1

        if predicted.lower() != "tulsi":
            print(f"{path.name} -> {predicted} ({confidence:.2f}%)")

    except Exception as exc:
        print(f"ERROR: {path.name}: {exc}")

print("\n--- TULSI DIAGNOSTIC ---")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct / total * 100:.2f}%" if total else "No images found")
print("Predictions:", dict(results))
