
import json
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "processed"
MODEL_DIR = ROOT / "ML" / "models"

LABELS_PATH = MODEL_DIR / "class_names.json"
CANDIDATE_PATH = (
    MODEL_DIR / "leaflens_mobilenetv2_augmented.keras"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)

with LABELS_PATH.open("r", encoding="utf-8") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [
        name for name, index in
        sorted(class_names.items(), key=lambda x: x[1])
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

# Augmentation is active only during training.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
    tf.keras.layers.RandomContrast(0.10),
], name="leaf_augmentation")

# One pretrained MobileNetV2 feature extractor.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CANDIDATE_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nTraining complete!")
print("Candidate model:", CANDIDATE_PATH)
print("Best validation accuracy:",
      round(max(history.history["val_accuracy"]) * 100, 2),
      "%")
print("Original model has not been modified.")
