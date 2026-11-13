
import json
import csv
from pathlib import Path

import tensorflow as tf

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ML" / "data" / "grouped"
MODEL_DIR = ROOT / "ML" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR / "leaflens_mobilenetv2_grouped_windows.keras"
)

LABELS_PATH = (
    MODEL_DIR / "class_names_grouped_windows.json"
)

HISTORY_PATH = (
    MODEL_DIR / "training_history_grouped_windows.csv"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# 2. Dataset folders
# --------------------------------------------------

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"

for folder in (train_dir, val_dir, test_dir):
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folder: {folder}"
        )

# --------------------------------------------------
# 3. Class names
# --------------------------------------------------

class_names = sorted(
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
)

NUM_CLASSES = len(class_names)

if NUM_CLASSES != 78:
    raise ValueError(
        f"Expected 78 classes, found {NUM_CLASSES}"
    )

# Check that all splits have the same classes.
for split_dir in (val_dir, test_dir):

    split_classes = {
        folder.name
        for folder in split_dir.iterdir()
        if folder.is_dir()
    }

    if split_classes != set(class_names):
        raise ValueError(
            f"Class folders do not match: {split_dir}"
        )

# Check every class has at least one image in every split.
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

for split_dir in (train_dir, val_dir, test_dir):

    for plant in class_names:

        folder = split_dir / plant

        count = sum(
            1 for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in extensions
        )

        if count == 0:
            raise ValueError(
                f"No images found in: {folder}"
            )

# --------------------------------------------------
# 4. Load training and validation datasets
# --------------------------------------------------

def load_dataset(folder, shuffle):

    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int",
        interpolation="bilinear",
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


train_ds = load_dataset(train_dir, shuffle=True)
val_ds = load_dataset(val_dir, shuffle=False)

print("\n===================================")
print("LEAFLENS-AI GROUPED MODEL TRAINING")
print("===================================")

print(f"Classes: {NUM_CLASSES}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Training folder: {train_dir}")
print(f"Validation folder: {val_dir}")
print(f"Model output: {MODEL_PATH}")

# --------------------------------------------------
# 5. MobileNetV2 feature extractor
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained layers for baseline training.
base_model.trainable = False

# --------------------------------------------------
# 6. Classification model
# --------------------------------------------------

inputs = tf.keras.Input(
    shape=(*IMAGE_SIZE, 3),
    name="leaf_image"
)

# Preprocessing is included INSIDE the model.
# The Flask backend should supply raw RGB images
# with pixel values in the range 0-255.

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    inputs
)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="plant_predictions"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="LeafLens_Grouped_Windows"
)

# --------------------------------------------------
# 7. Compile
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 8. Training callbacks
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]

# --------------------------------------------------
# 9. Train
# --------------------------------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# --------------------------------------------------
# 10. Evaluate the best checkpoint on validation
# --------------------------------------------------

best_model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1,
)

# --------------------------------------------------
# 11. Save class names
# --------------------------------------------------

LABELS_PATH.write_text(
    json.dumps(class_names, indent=2),
    encoding="utf-8",
)

# --------------------------------------------------
# 12. Save training history
# --------------------------------------------------

history_columns = list(history.history.keys())

with HISTORY_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)
    writer.writerow(["epoch", *history_columns])

    for epoch in range(len(history.epoch)):

        writer.writerow([
            epoch + 1,
            *[
                history.history[column][epoch]
                for column in history_columns
            ]
        ])

# --------------------------------------------------
# 13. Final summary
# --------------------------------------------------

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print(f"Model saved: {MODEL_PATH}")
print(f"Labels saved: {LABELS_PATH}")
print(f"History saved: {HISTORY_PATH}")

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

print("\nTest dataset has not been evaluated.")
print("It remains reserved for final evaluation.")
