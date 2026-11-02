import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "ML" / "models"
    / "leaflens_mobilenetv2_grouped_baseline.keras"
)

LABELS_PATH = ROOT / "ML" / "models" / "class_names.json"
VAL_DIR = ROOT / "ML" / "data" / "grouped" / "val"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

with LABELS_PATH.open(encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda x: x[1])
    ]

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = []
y_true = []

for index, plant in enumerate(class_names):
    folder = VAL_DIR / plant

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in EXTENSIONS:
            image_paths.append(path)
            y_true.append(index)

y_true = np.asarray(y_true)

def crop_image(image, fraction):
    if fraction == 1.0:
        return image

    width, height = image.size

    new_width = max(1, int(width * fraction))
    new_height = max(1, int(height * fraction))

    left = (width - new_width) // 2
    top = (height - new_height) // 2

    return image.crop((
        left,
        top,
        left + new_width,
        top + new_height
    ))

def evaluate(fraction):
    predictions = []

    for start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        batch_images = []

        for path in batch_paths:
            with Image.open(path) as original:
                image = original.convert("RGB")
                image = crop_image(image, fraction)

                array = np.asarray(
                    image,
                    dtype=np.float32
                )

            resized = tf.image.resize(
                array,
                IMAGE_SIZE,
                method="bilinear"
            )

            batch_images.append(resized.numpy())

        batch = np.stack(batch_images)

        probabilities = model.predict(
            batch,
            verbose=0
        )

        predictions.extend(
            np.argmax(probabilities, axis=1)
        )

    predictions = np.asarray(predictions)

    accuracy = np.mean(
        predictions == y_true
    ) * 100

    tulsi_index = class_names.index("Tulsi")
    tulsi_mask = y_true == tulsi_index

    tulsi_recall = np.mean(
        predictions[tulsi_mask] == tulsi_index
    ) * 100

    return accuracy, tulsi_recall


print("\n--- CROPPING VALIDATION EXPERIMENT ---")
print("Images:", len(image_paths))

for fraction in (1.0, 0.8, 0.6):
    accuracy, tulsi_recall = evaluate(fraction)

    print(
        f"Crop {fraction:.1f}: "
        f"Accuracy={accuracy:.2f}% | "
        f"Tulsi recall={tulsi_recall:.2f}%"
    )

print("\nEvaluation complete.")