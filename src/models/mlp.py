"""Train and evaluate a scikit-learn MLP on CIC-IDS2017 flow data."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import load_dataset, split_and_preprocess


def make_holdout_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    attack_name: str,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, int, int]:
    """Train on other attacks and test on the held-out attack plus benign flows."""
    labels = metadata["attack_label"].astype("string").str.strip()
    held_out_mask = labels.str.casefold().eq(attack_name.casefold()).fillna(False)
    if not held_out_mask.any():
        available = sorted(labels.dropna().unique().tolist())
        raise ValueError(
            f"No rows found for held-out attack {attack_name!r}. "
            f"Available labels: {available}"
        )

    held_out_X = X.loc[held_out_mask].reset_index(drop=True)
    held_out_count = len(held_out_X)
    seen_mask = ~held_out_mask
    seen_X = X.loc[seen_mask].reset_index(drop=True)
    seen_y = y.loc[seen_mask].reset_index(drop=True)
    seen_metadata = metadata.loc[seen_mask].reset_index(drop=True)

    # Exclude identical feature rows from training when they also occur in the
    # held-out attack, preventing duplicate-flow leakage across the experiment.
    held_out_hashes = pd.util.hash_pandas_object(held_out_X, index=False).unique()
    seen_hashes = pd.util.hash_pandas_object(seen_X, index=False).to_numpy()
    duplicate_mask = np.isin(seen_hashes, held_out_hashes)
    duplicates_removed = int(duplicate_mask.sum())
    if duplicates_removed:
        keep = ~duplicate_mask
        seen_X = seen_X.loc[keep].reset_index(drop=True)
        seen_y = seen_y.loc[keep].reset_index(drop=True)
        seen_metadata = seen_metadata.loc[keep].reset_index(drop=True)

    data = split_and_preprocess(seen_X, seen_y, seen_metadata, random_seed=seed)

    # For a realistic unseen-attack test, pair every held-out Infiltration
    # flow with benign flows reserved in the seen-data test partition.
    benign_test_mask = data["y_test"] == 0
    benign_test_X = data["X_test"][benign_test_mask]
    benign_test_count = len(benign_test_X)
    test_values = held_out_X.to_numpy(dtype=np.float32, copy=True)
    test_values[~np.isfinite(test_values)] = np.nan
    medians = data["imputer_medians"]
    missing_rows, missing_columns = np.where(np.isnan(test_values))
    test_values[missing_rows, missing_columns] = medians[missing_columns]
    test_values = (
        (test_values - data["scaler_means"]) / data["scaler_scales"]
    ).astype(np.float32, copy=False)

    X_test = np.concatenate((benign_test_X, test_values), axis=0)
    y_test = np.concatenate(
        (np.zeros(benign_test_count, dtype=np.int64),
         np.ones(held_out_count, dtype=np.int64))
    )
    data["holdout_attack"] = attack_name
    data["heldout_attack_count"] = held_out_count
    data["benign_test_count"] = benign_test_count
    data["duplicate_rows_removed"] = duplicates_removed
    data["heldout_source_files"] = (
        metadata.loc[held_out_mask, "source_file"].astype(str).tolist()
    )
    return data, X_test, y_test, held_out_count, benign_test_count


def print_metrics(title: str, labels: np.ndarray, probabilities: np.ndarray) -> None:
    predictions = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    print(f"\n{title} metrics at threshold 0.50")
    print(f"  Accuracy             {accuracy_score(labels, predictions):.4f}")
    print(f"  Balanced accuracy    {balanced_accuracy_score(labels, predictions):.4f}")
    print(
        "  Attack precision     "
        f"{precision_score(labels, predictions, pos_label=1, zero_division=0):.4f}"
    )
    print(
        "  Attack recall        "
        f"{recall_score(labels, predictions, pos_label=1, zero_division=0):.4f}"
    )
    print(
        "  Attack F1            "
        f"{f1_score(labels, predictions, pos_label=1, zero_division=0):.4f}"
    )
    print(f"  Macro F1             {f1_score(labels, predictions, average='macro'):.4f}")
    print("  Confusion matrix (actual rows: benign, attack; predicted columns: benign, attack)")
    print(f"  [[{matrix[0, 0]:,}, {matrix[0, 1]:,}],")
    print(f"   [{matrix[1, 0]:,}, {matrix[1, 1]:,}]]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_source",
        nargs="?",
        type=Path,
        default=Path("data/raw/MachineLearningCSV.zip"),
        help="CIC-IDS2017 ZIP archive or extracted CSV directory.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument(
        "--holdout-attack",
        help="Exclude this exact attack label from training and test it as unseen.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch size, and patience must all be positive")

    np.random.seed(args.seed)
    print(f"scikit-learn: {sklearn.__version__}")
    print("Loading data and rebuilding the prepared splits...")
    started = time.perf_counter()
    X, y, metadata = load_dataset(args.data_source)
    if args.holdout_attack:
        data, X_test, y_test, attack_test_count, benign_test_count = (
            make_holdout_experiment(
                X, y, metadata, args.holdout_attack, args.seed
            )
        )
        output_name = f"sklearn_mlp_holdout_{args.holdout_attack.lower()}.joblib"
        print(
            f"Holdout attack: {args.holdout_attack} ({attack_test_count:,} flows); "
            f"benign test flows: {benign_test_count:,}; "
            f"duplicate rows excluded from seen data: {data['duplicate_rows_removed']:,}."
        )
    else:
        data = split_and_preprocess(X, y, metadata, random_seed=args.seed)
        X_test = data["X_test"]
        y_test = data["y_test"]
        output_name = "sklearn_mlp_binary.joblib"
    del X, y, metadata
    print(
        f"Rows: train={len(data['y_train']):,}, "
        f"validation={len(data['y_validation']):,}, "
        f"test={len(y_test):,}; features={len(data['feature_names'])}."
    )

    train_labels = data["y_train"]
    benign_count = int(np.count_nonzero(train_labels == 0))
    attack_count = int(np.count_nonzero(train_labels == 1))
    sample_weights = np.where(
        train_labels == 1,
        benign_count / max(attack_count, 1),
        1.0,
    ).astype(np.float32)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=args.batch_size,
        learning_rate_init=1e-3,
        max_iter=1,
        shuffle=True,
        random_state=args.seed,
        early_stopping=False,
        verbose=False,
    )

    best_validation_loss = float("inf")
    best_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.partial_fit(
            data["X_train"],
            train_labels,
            classes=np.array([0, 1]) if epoch == 1 else None,
            sample_weight=sample_weights,
        )
        validation_probabilities = model.predict_proba(data["X_validation"])[:, 1]
        validation_loss = log_loss(
            data["y_validation"],
            validation_probabilities,
            labels=[0, 1],
        )
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"training loss={model.loss_:.4f}, validation log loss={validation_loss:.4f}, "
            f"time={time.perf_counter() - epoch_started:.1f}s"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = (
                [weights.copy() for weights in model.coefs_],
                [biases.copy() for biases in model.intercepts_],
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation log loss did not improve.")
                break

    if best_weights is None:
        raise RuntimeError("Training ended before a model checkpoint was created.")
    model.coefs_, model.intercepts_ = best_weights

    output_path = PROJECT_ROOT / "models" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": data["feature_names"],
            "imputer_medians": data["imputer_medians"],
            "scaler_means": data["scaler_means"],
            "scaler_scales": data["scaler_scales"],
            "threshold": 0.5,
            "seed": args.seed,
            "best_validation_log_loss": best_validation_loss,
        },
        output_path,
        compress=3,
    )

    validation_probabilities = model.predict_proba(data["X_validation"])[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    print_metrics("Validation", data["y_validation"], validation_probabilities)
    test_title = (
        f"Unseen {args.holdout_attack} + benign test"
        if args.holdout_attack
        else "Test"
    )
    print_metrics(test_title, y_test, test_probabilities)
    if args.holdout_attack:
        attack_probabilities = test_probabilities[-attack_test_count:]
        attack_predictions = (attack_probabilities >= 0.5).astype(np.int64)
        probability_summary = pd.Series(attack_probabilities).describe(
            percentiles=[0.25, 0.5, 0.75]
        )
        print(f"\n{args.holdout_attack} predicted attack probabilities")
        print(
            "  min={min:.6f}, 25%={p25:.6f}, median={median:.6f}, "
            "mean={mean:.6f}, 75%={p75:.6f}, max={max:.6f}".format(
                min=probability_summary["min"],
                p25=probability_summary["25%"],
                median=probability_summary["50%"],
                mean=probability_summary["mean"],
                p75=probability_summary["75%"],
                max=probability_summary["max"],
            )
        )
        print(
            f"  Predicted as attack at threshold 0.50: "
            f"{int(attack_predictions.sum())}/{attack_test_count}"
        )
        attack_slug = "".join(
            character.lower() if character.isalnum() else "_"
            for character in args.holdout_attack
        ).strip("_")
        report_path = PROJECT_ROOT / "results" / f"holdout_{attack_slug}_probabilities.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "source_file": data["heldout_source_files"],
                "heldout_flow_number": np.arange(1, attack_test_count + 1),
                "true_label": np.ones(attack_test_count, dtype=np.int64),
                "predicted_attack_probability": attack_probabilities,
                "predicted_label_at_0_50": attack_predictions,
            }
        ).to_csv(report_path, index=False)
        print(f"Per-flow probabilities saved to {report_path}")
    print(f"\nBest model saved to {output_path}")
    print(f"Total time: {(time.perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
