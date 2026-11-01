
import json
import shutil
from datetime import datetime
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "processed"
MODEL_DIR = ROOT / "ML" / "models"

MODEL_PATH = MODEL_DIR / "leaflens_mobilenetv2.keras"
LABELS_PATH = MODEL_DIR / "class_names.json"
CANDIDATE_PATH = MODEL_DIR / "leaflens_mobilenetv2_finetuned.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42
LEARNING_RATE = 0.00001
LAST_LAYERS = 30

tf.keras.utils.set_random_seed(SEED)

for path in (
    MODEL_PATH,
    LABELS_PATH,
    DATA_DIR / "train",
    DATA_DIR / "val",
    DATA_DIR / "test",
):
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

with open(LABELS_PATH, encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda item: item[1])
    ]

def load_dataset(split, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int",
    ).prefetch(tf.data.AUTOTUNE)

train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val")
test_ds = load_dataset("test")

# Preserve the original model before doing anything else.
backup_path = MODEL_DIR / (
    f"leaflens_mobilenetv2_backup_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
)
shutil.copy2(MODEL_PATH, backup_path)
print(f"Original model backed up to: {backup_path}")

# Load the existing model: no second architecture or new classifier.
original = tf.keras.models.load_model(MODEL_PATH)

# Find the pretrained MobileNetV2 inside the saved model.
base_model = next(
    (
        layer for layer in original.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    ),
    None,
)

if base_model is None:
    raise RuntimeError(
        "Could not locate MobileNetV2 inside the saved model. "
        "No training was started."
    )

# Freeze the model, then unfreeze only its final convolutional layers.
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-LAST_LAYERS:]:
    # Keep Batch Normalization statistics stable on this small dataset.
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Add gentle augmentation before the original model.
# It runs only during training; inference uses the normal image pipeline.
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="training_augmentation",
)

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
augmented = augmentation(inputs)
outputs = original(augmented)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CANDIDATE_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    ),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save the best candidate after restoring the best weights.
model.save(CANDIDATE_PATH)

best_model = tf.keras.models.load_model(CANDIDATE_PATH)

test_loss, test_accuracy = best_model.evaluate(
    test_ds, verbose=1
)

print("\nFine-tuning finished.")
print(f"Original model backup: {backup_path}")
print(f"Candidate model: {CANDIDATE_PATH}")
print(f"Test loss: {test_loss:.4f}")
print(f"Overall test accuracy: {test_accuracy * 100:.2f}%")
print("The live website model has NOT been replaced.")
