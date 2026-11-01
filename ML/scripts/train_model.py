
import json
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "ML" / "data" / "processed"
MODEL_DIR = ROOT / "ML" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42

tf.keras.utils.set_random_seed(SEED)

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"

for folder in (train_dir, val_dir, test_dir):
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

# Derive class labels from the training folders.
class_names = sorted(
    p.name for p in train_dir.iterdir() if p.is_dir()
)

if len(class_names) < 2:
    raise ValueError("At least two classes are required.")

# Load all splits with exactly the same class ordering.
def load_dataset(folder, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int",
    ).prefetch(tf.data.AUTOTUNE)

train_ds = load_dataset(train_dir, True)
val_ds = load_dataset(val_dir, False)
test_ds = load_dataset(test_dir, False)

print(f"\nNumber of classes: {len(class_names)}")
print(f"Image size: {IMAGE_SIZE}")
print("Training on CPU is supported; GPU is not required.")

# ONE pretrained MobileNetV2 feature extractor.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model_path = MODEL_DIR / "leaflens_mobilenetv2.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
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
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Evaluate the best saved model on unseen test images.
best_model = tf.keras.models.load_model(model_path)
test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1,
)

# Save the exact class ordering used by the model.
labels_path = MODEL_DIR / "class_names.json"
labels_path.write_text(
    json.dumps(class_names, indent=2),
    encoding="utf-8",
)

print("\nTraining complete!")
print(f"Model: {model_path}")
print(f"Labels: {labels_path}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
