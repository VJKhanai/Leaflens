from pathlib import Path

import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ML/data/hybrid"

ORIGINAL_MODEL = (
    ROOT / "ML/models/leaflens_mobilenetv2_hybrid.keras"
)

FINETUNED_MODEL = (
    ROOT / "ML/models/leaflens_mobilenetv2_hybrid_finetuned.keras"
)

TRAINING_LOG = (
    ROOT / "ML/models/hybrid_finetuning_log.csv"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 6
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# ----------------------------------
# 1. Load datasets
# ----------------------------------

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if train_ds.class_names != val_ds.class_names:
    raise ValueError("Class names do not match")

if len(train_ds.class_names) != 78:
    raise ValueError("Expected 78 classes")

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ----------------------------------
# 2. Load existing hybrid model
# ----------------------------------

if not ORIGINAL_MODEL.is_file():
    raise FileNotFoundError(ORIGINAL_MODEL)

if FINETUNED_MODEL.exists():
    raise SystemExit(
        "Fine-tuned model already exists. "
        "Rename it before starting another experiment."
    )

model = keras.models.load_model(ORIGINAL_MODEL)

# ----------------------------------
# 3. Find MobileNetV2 backbone
# ----------------------------------

base_model = next(
    (
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model)
        and "mobilenetv2" in layer.name.lower()
    ),
    None,
)

if base_model is None:
    raise ValueError("MobileNetV2 backbone not found")

# ----------------------------------
# 4. Unfreeze only final layers
# ----------------------------------

base_model.trainable = True

# Freeze earlier pretrained layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization frozen.
for layer in base_model.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

print("\nTrainable backbone layers:")

for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

# ----------------------------------
# 5. Compile with low learning rate
# ----------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ----------------------------------
# 6. Callbacks
# ----------------------------------

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(FINETUNED_MODEL),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        str(TRAINING_LOG)
    ),
]

# ----------------------------------
# 7. Fine-tuning
# ----------------------------------

print("\n================================")
print("STARTING HYBRID FINE-TUNING")
print("================================")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ----------------------------------
# 8. Evaluate best saved checkpoint
# ----------------------------------

best_model = keras.models.load_model(
    FINETUNED_MODEL
)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1,
)

print("\n================================")
print("HYBRID FINE-TUNING COMPLETED")
print("================================")

print(
    f"Validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation loss: {val_loss:.4f}"
)

print("Saved model:", FINETUNED_MODEL)
print("Training log:", TRAINING_LOG)
print("\nReserved test dataset was not used.")