
import json
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ML/data/processed/test"
MODELS = ROOT / "ML/models"

with open(MODELS / "class_names.json", encoding="utf-8") as f:
    names = json.load(f)

if isinstance(names, dict):
    names = [
        name for name, idx in
        sorted(names.items(), key=lambda item: item[1])
    ]

ds = tf.keras.utils.image_dataset_from_directory(
    DATA,
    class_names=names,
    image_size=(160, 160),
    batch_size=8,
    shuffle=False,
    label_mode="int",
)

models = [
    ("Original", MODELS / "leaflens_mobilenetv2.keras"),
    ("Fine-tuned", MODELS / "leaflens_mobilenetv2_finetuned.keras"),
]

for model_name, path in models:
    print(f"\n{'=' * 18} {model_name} {'=' * 18}")
    model = tf.keras.models.load_model(path)

    correct = total = 0
    tulsi_total = tulsi_correct = 0
    tulsi_errors = Counter()

    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        predicted = np.argmax(probs, axis=1)
        actual = labels.numpy()

        correct += int(np.sum(predicted == actual))
        total += len(actual)

        tulsi_idx = names.index("Tulsi")
        for true_idx, pred_idx in zip(actual, predicted):
            if true_idx == tulsi_idx:
                tulsi_total += 1
                if pred_idx == tulsi_idx:
                    tulsi_correct += 1
                else:
                    tulsi_errors[names[pred_idx]] += 1

    print(f"Overall accuracy: {correct / total * 100:.2f}%")
    print(f"Tulsi accuracy: {tulsi_correct}/{tulsi_total} "
          f"({tulsi_correct / tulsi_total * 100:.2f}%)")
    print("Tulsi misclassified as:", dict(tulsi_errors))
