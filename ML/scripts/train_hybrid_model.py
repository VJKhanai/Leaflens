import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ML" / "data" / "hybrid"
MODEL_DIR = ROOT / "ML" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = (
    MODEL_DIR / "leaflens_mobilenetv2_hybrid.keras"
)

LABELS_PATH = (
    MODEL_DIR / "class_names_hybrid.json"
)

LOG_PATH = (
    MODEL_DIR / "hybrid_training_log.csv"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# ==========================================
# 2. LOAD DATASETS
# ==========================================

print("\nLoading hybrid training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

print("\nLoading hybrid validation dataset...")

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names

if class_names != val_ds.class_names:
    raise ValueError(
        "Training and validation class names do not match!"
    )

NUM_CLASSES = len(class_names)

print("\nTotal classes:", NUM_CLASSES)

if NUM_CLASSES != 78:
    raise ValueError(
        f"Expected 78 classes, found {NUM_CLASSES}"
    )

if MODEL_PATH.exists():
    raise SystemExit(
        f"Model already exists: {MODEL_PATH}\n"
        "Rename or back it up before retraining."
    )

with LABELS_PATH.open("w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print("Class labels saved:", LABELS_PATH)


# ==========================================
# 3. OPTIMIZE DATA PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
    ],
    name="data_augmentation",
)


# ==========================================
# 5. LOAD PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained layers initially
base_model.trainable = False


# ==========================================
# 6. BUILD HYBRID MODEL
# ==========================================

inputs = keras.Input(
    shape=(160, 160, 3),
    name="input_image",
)

x = augmentation(inputs)

# MobileNetV2 expects values in [-1, 1].
# Preprocessing is inside the saved model.
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1.0,
    name="mobilenet_preprocessing",
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="plant_prediction",
)(x)

model = keras.Model(
    inputs,
    outputs,
    name="LeafLens_Hybrid_MobileNetV2",
)


# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ==========================================
# 8. TRAINING CALLBACKS
# ==========================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
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
        min_lr=0.00001,
        verbose=1,
    ),

    keras.callbacks.CSVLogger(
        str(LOG_PATH)
    ),
]


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\n================================")
print("STARTING HYBRID MODEL TRAINING")
print("================================")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ==========================================
# 10. VALIDATION RESULTS
# ==========================================

print("\nLoading best saved model...")

best_model = keras.models.load_model(
    MODEL_PATH
)

val_loss, val_accuracy = best_model.evaluate(
    val_ds,
    verbose=1,
)

print("\n================================")
print("HYBRID TRAINING COMPLETED")
print("================================")

print(
    f"Best validation accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Best validation loss: "
    f"{val_loss:.4f}"
)

print("Model saved:", MODEL_PATH)
print("Labels saved:", LABELS_PATH)
print("Training log:", LOG_PATH)

print("\nTest dataset remains untouched.")