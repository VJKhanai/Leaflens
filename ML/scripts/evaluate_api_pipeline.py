
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ML" / "models" / "leaflens_mobilenetv2.keras"
LABELS_PATH = ROOT / "ML" / "models" / "class_names.json"
TEST_DIR = ROOT / "ML" / "data" / "processed" / "test"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8

with open(LABELS_PATH, encoding="utf-8") as f:
    class_names = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

# Dataset images are RGB float32 in the range 0–255.
# The model itself performs MobileNetV2 preprocessing.
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\nAccuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
print("\nClassification report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    digits=3,
))

print("\nFirst 10 test predictions:")
for actual, predicted, probs in zip(y_true[:10], y_pred[:10], probabilities[:10]):
    print(
        f"Actual: {class_names[actual]:25s} "
        f"Predicted: {class_names[predicted]:25s} "
        f"Confidence: {np.max(probs) * 100:.2f}%"
    )
