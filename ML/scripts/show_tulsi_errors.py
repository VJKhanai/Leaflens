
from pathlib import Path
import json
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "ML/models/leaflens_mobilenetv2.keras"
LABELS = ROOT / "ML/models/class_names.json"
TULSI_TEST = ROOT / "ML/data/processed/test/Tulsi"

with open(LABELS, encoding="utf-8") as f:
    labels = json.load(f)

if isinstance(labels, dict):
    labels = [
        name for name, index in
        sorted(labels.items(), key=lambda item: item[1])
    ]

model = tf.keras.models.load_model(MODEL)
errors = []

for path in sorted(TULSI_TEST.iterdir()):
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue

    image = tf.keras.utils.load_img(path, target_size=(160, 160))
    array = tf.keras.utils.img_to_array(image)[None, ...]

    # Preprocessing is already inside the saved model.
    probs = model.predict(array, verbose=0)[0]
    idx = int(np.argmax(probs))

    if labels[idx].lower() != "tulsi":
        errors.append((path, labels[idx], float(probs[idx]) * 100))

if not errors:
    print("No misclassified Tulsi test images found.")
    raise SystemExit

cols = 4
rows = math.ceil(len(errors) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
axes = np.array(axes).reshape(-1)

for ax, (path, predicted, confidence) in zip(axes, errors):
    ax.imshow(plt.imread(path))
    ax.set_title(f"{path.name}\nPredicted: {predicted} ({confidence:.1f}%)")
    ax.axis("off")

for ax in axes[len(errors):]:
    ax.axis("off")

plt.tight_layout()
output = ROOT / "ML" / "tulsi_misclassified.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
print(f"Misclassified images: {len(errors)}")
print(f"Contact sheet saved to: {output}")
plt.show()
