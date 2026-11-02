
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ML" / "models" / "leaflens_mobilenetv2.keras"
LABELS_PATH = ROOT / "ML" / "models" / "class_names.json"
TEST_DIR = ROOT / "ML" / "data" / "processed" / "test"

with open(LABELS_PATH, encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda item: item[1])
    ]

model = tf.keras.models.load_model(
    MODEL_PATH, compile=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=class_names,
    image_size=(160, 160),
    batch_size=8,
    shuffle=False,
    interpolation="bilinear",
)

correct = 0
total = 0

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    for actual, predicted in zip(labels.numpy(), predictions):
        if class_names[int(actual)].lower() == "tulsi":
            total += 1
            correct += class_names[int(predicted)].lower() == "tulsi"

print("\n--- TULSI DATASET PIPELINE ---")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {100 * correct / total:.2f}%" if total else "No Tulsi images found")
