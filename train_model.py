from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------
# Configuration / disclosed hyperparameters
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
TEST_DIR = PROJECT_DIR / "data" / "test"
OUTPUT_DIR = PROJECT_DIR / "outputs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42

LEARNING_RATE = 1e-4
EPOCHS = 25
DROPOUT_RATE = 0.30
EARLY_STOPPING_PATIENCE = 5

MODEL_PATH = OUTPUT_DIR / "casting_model.keras"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"


def validate_directories() -> None:
    """Check that the expected dataset structure exists."""
    expected = [
        TRAIN_DIR / "def_front",
        TRAIN_DIR / "ok_front",
        TEST_DIR / "def_front",
        TEST_DIR / "ok_front",
    ]

    missing = [str(path) for path in expected if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "The following required folders were not found:\n"
            + "\n".join(missing)
        )


def load_datasets():
    """Load training, validation and test datasets."""

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["def_front", "ok_front"],
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=True,
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["def_front", "ok_front"],
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["def_front", "ok_front"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=autotune)
    validation_ds = validation_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, validation_ds, test_ds


def build_model() -> tf.keras.Model:
    """Create a MobileNetV2 transfer-learning model."""

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
            ),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    # First training stage: retain pretrained visual features.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,), name="image")

    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # training=False keeps batch-normalisation layers stable.
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    # Because class 0 = defective and class 1 = OK,
    # this output represents P(OK).
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="ok_probability",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision_ok"),
            tf.keras.metrics.Recall(name="recall_ok"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )

    return model


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save accuracy and loss plots."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"], label="Training accuracy")
    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], label="Training loss")
    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=200)
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    threshold_ok: float = 0.50,
) -> None:
    """
    Evaluate the model.

    The network outputs P(OK), because:
        defective = 0
        acceptable = 1

    A casting is predicted as defective whenever:
        P(OK) < threshold_ok
    """

    y_true = np.concatenate(
        [labels.numpy().ravel() for _, labels in test_ds]
    ).astype(int)

    ok_probabilities = model.predict(test_ds, verbose=1).ravel()
    y_pred = (ok_probabilities >= threshold_ok).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Defective", "OK"],
        digits=4,
        output_dict=True,
    )

    print("\nClassification report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Defective", "OK"],
            digits=4,
        )
    )

    matrix = confusion_matrix(y_true, y_pred)

    # For defective = 0:
    # matrix[0, 1] means defective predicted as OK.
    shipped_defective_parts = int(matrix[0, 1])
    detected_defective_parts = int(matrix[0, 0])

    defective_recall = report["Defective"]["recall"]
    defective_precision = report["Defective"]["precision"]
    defective_f1 = report["Defective"]["f1-score"]

    print(f"Decision threshold for P(OK): {threshold_ok:.2f}")
    print(f"Defective recall: {defective_recall:.4f}")
    print(f"Defective precision: {defective_precision:.4f}")
    print(f"Defective F1-score: {defective_f1:.4f}")
    print(
        "False negatives — defective parts incorrectly passed as OK: "
        f"{shipped_defective_parts}"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Defective", "OK"],
    )
    display.plot(values_format="d")
    plt.title(f"Confusion Matrix — threshold {threshold_ok:.2f}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    results = {
        "threshold_ok": threshold_ok,
        "detected_defective_parts": detected_defective_parts,
        "false_negative_defective_parts": shipped_defective_parts,
        "defective_recall": defective_recall,
        "defective_precision": defective_precision,
        "defective_f1": defective_f1,
        "classification_report": report,
    }

    with open(
        OUTPUT_DIR / "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    validate_directories()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))

    train_ds, validation_ds, test_ds = load_datasets()

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    plot_training_history(history)
    evaluate_model(model, test_ds, threshold_ok=0.50)

    metadata = {
        "model": "MobileNetV2 transfer learning",
        "image_size": list(IMAGE_SIZE),
        "class_names": ["def_front", "ok_front"],
        "class_mapping": {
            "def_front": 0,
            "ok_front": 1,
        },
        "output_meaning": "Probability that the image is OK",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
        "dropout_rate": DROPOUT_RATE,
        "optimizer": "Adam",
        "loss": "Binary cross-entropy",
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()