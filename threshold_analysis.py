from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "outputs" / "casting_model.keras"
TEST_DIR = PROJECT_DIR / "data" / "test"
OUTPUT_DIR = PROJECT_DIR / "outputs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def load_test_dataset() -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["def_front", "ok_front"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        shuffle=False,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    test_ds = load_test_dataset()

    y_true = np.concatenate(
        [labels.numpy().ravel() for _, labels in test_ds]
    ).astype(int)

    ok_probabilities = model.predict(
        test_ds,
        verbose=1,
    ).ravel()

    rows = []

    for threshold in THRESHOLDS:
        y_pred = (
            ok_probabilities >= threshold
        ).astype(int)

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        true_defective = matrix[0, 0]
        missed_defective = matrix[0, 1]
        false_alarm_ok = matrix[1, 0]
        true_ok = matrix[1, 1]

        defective_recall = (
            true_defective
            / (true_defective + missed_defective)
        )

        defective_precision = (
            true_defective
            / (true_defective + false_alarm_ok)
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        rows.append(
            {
                "threshold_ok": threshold,
                "accuracy": accuracy,
                "defective_recall": defective_recall,
                "defective_precision": defective_precision,
                "missed_defective_parts": missed_defective,
                "acceptable_parts_flagged": false_alarm_ok,
                "correctly_detected_defects": true_defective,
                "correctly_accepted_parts": true_ok,
            }
        )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        index=False,
    )

    print("\nThreshold comparison")
    print(results.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.plot(
        results["threshold_ok"],
        results["defective_recall"],
        marker="o",
        label="Defective recall",
    )
    plt.plot(
        results["threshold_ok"],
        results["accuracy"],
        marker="o",
        label="Accuracy",
    )
    plt.xlabel("Minimum probability required to pass as OK")
    plt.ylabel("Score")
    plt.title("Threshold Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "threshold_tradeoff.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        results["threshold_ok"],
        results["missed_defective_parts"],
        marker="o",
        label="Missed defective parts",
    )
    plt.plot(
        results["threshold_ok"],
        results["acceptable_parts_flagged"],
        marker="o",
        label="Acceptable parts flagged",
    )
    plt.xlabel("Minimum probability required to pass as OK")
    plt.ylabel("Number of parts")
    plt.title("Operational Cost Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "threshold_operational_tradeoff.png",
        dpi=200,
    )
    plt.close()

    print(
        "\nSaved threshold_analysis.csv and two charts "
        f"to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()