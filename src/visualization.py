"""Plot a saved CIC-IDS2017 MLP's row-normalized confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mlp import (
    DATA_DIR,
    latest_model_path,
    load_data,
    load_model_artifact,
    predict_with_thresholds,
    prepare_train_test,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-rows-per-file", type=int, default=170_366)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Saved Joblib model; defaults to the latest models/mlp_vN.joblib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / "confusion_matrix.png",
        help="File path for the heatmap image.",
    )
    args = parser.parse_args()

    model_path = args.model or latest_model_path()
    model, thresholds = load_model_artifact(model_path)
    print(f"Loaded model from {model_path}")

    data = load_data(args.data_dir, args.max_rows_per_file, args.random_state)
    X_train, X_test, y_train, y_test = prepare_train_test(
        data, args.test_size, args.random_state
    )
    # X_train/y_train are prepared for the same deterministic split but are
    # intentionally not oversampled or refitted here: this evaluates the
    # already-saved model against untouched test rows.
    del X_train, y_train
    baseline_predictions = model.predict(X_test)
    adjusted_predictions = predict_with_thresholds(model, X_test, thresholds)

    print("\nOriginal test results:")
    print(f"Accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
    baseline_report = classification_report(
        y_test, baseline_predictions, zero_division=0
    )
    print(baseline_report.encode("ascii", errors="backslashreplace").decode("ascii"))
    print("\nTest results with validation-calibrated class thresholds:")
    print(f"Accuracy: {accuracy_score(y_test, adjusted_predictions):.4f}")
    adjusted_report = classification_report(
        y_test, adjusted_predictions, zero_division=0
    )
    print(adjusted_report.encode("ascii", errors="backslashreplace").decode("ascii"))

    # Include predicted-only labels too, so false positives for a class that
    # has no actual test examples remain visible in the heatmap.
    labels = sorted(set(y_test.unique()) | set(adjusted_predictions))
    matrix = confusion_matrix(
        y_test, adjusted_predictions, labels=labels, normalize="true"
    )
    figure_size = max(10, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Share of actual class"},
        ax=ax,
    )
    ax.set_title("CIC-IDS2017 MLP Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {args.output.resolve()}")


if __name__ == "__main__":
    main()
