
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "ML" / "models" / "leaflens_mobilenetv2_finetuned.keras"
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

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)


from collections import Counter

# Show the most common classification mistakes
mistakes = Counter()

for actual, predicted in zip(y_true, y_pred):
    if actual != predicted:
        mistakes[(class_names[actual], class_names[predicted])] += 1

print("\nMost common misclassification pairs:")
for (actual, predicted), count in mistakes.most_common(15):
    print(f"{actual} -> {predicted}: {count} images")

# Specifically investigate Tulsi
if "Tulsi" in class_names:
    tulsi_index = class_names.index("Tulsi")
    tulsi_mask = y_true == tulsi_index
    tulsi_errors = Counter(
        class_names[p]
        for p in y_pred[tulsi_mask]
        if p != tulsi_index
    )

    print("\nTulsi misclassification breakdown:")
    if tulsi_errors:
        for predicted, count in tulsi_errors.most_common():
            print(f"Tulsi predicted as {predicted}: {count}")
    else:
        print("No Tulsi errors in this test set.")


print("\nClassification report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
    digits=3,
))

print("\nConfusion matrix shape:")
print(confusion_matrix(y_true, y_pred).shape)

print("\nOverall accuracy:")
print(f"{np.mean(y_true == y_pred) * 100:.2f}%")
