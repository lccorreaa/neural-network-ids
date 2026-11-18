"""Train a small scikit-learn MLP on CIC-IDS2017 flow CSVs.

Run from the repository root with ``python src/mlp.py``. By default, the
script takes a reproducible sample from each CSV so the first experiment fits
in memory on a typical workstation. Set ``--max-rows-per-file 0`` to load all
rows (which can require several GB of RAM).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "MachineLearningCVE"
LABEL_COLUMN = "Label"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
OVERSAMPLED_ATTACKS = {
    "infiltration": "Infiltration",
    "webattackbruteforce": "Web Attack Brute Force",
    "webattacksqlinjection": "Web Attack SQL Injection",
    "webattackxss": "Web Attack XSS",
}
ATTACK_ALIASES = {"webattackxss": ("webattackxss", "webatackxss")}
FULL_DATA_FILES = {"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"}
LOW_RECALL_LABELS = {
    "bot",
    "infiltration",
    "webattackbruteforce",
    "webattacksqlinjection",
    "webattackxss",
}


def safe_print(message: str) -> None:
    """Print text safely on Windows consoles that cannot encode dataset labels."""
    print(message.encode("ascii", errors="backslashreplace").decode("ascii"))


def save_model(
    model: Pipeline,
    thresholds: dict[str, float],
    model_dir: Path = MODEL_DIR,
) -> Path:
    """Save a fitted model without overwriting an earlier version."""
    model_dir.mkdir(parents=True, exist_ok=True)
    version = 1
    while (model_dir / f"mlp_v{version}.joblib").exists():
        version += 1
    model_path = model_dir / f"mlp_v{version}.joblib"
    joblib.dump({"estimator": model, "thresholds": thresholds}, model_path)
    return model_path


def load_model_artifact(model_path: Path) -> tuple[Pipeline, dict[str, float]]:
    """Load a saved estimator and its calibrated thresholds, if present."""
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "estimator" in artifact:
        return artifact["estimator"], artifact.get("thresholds", {})
    # Support pipeline-only files saved by earlier script versions.
    return artifact, {}


def latest_model_path(model_dir: Path = MODEL_DIR) -> Path:
    """Return the highest-numbered saved MLP model."""
    paths = list(model_dir.glob("mlp_v*.joblib"))
    if not paths:
        raise FileNotFoundError(
            f"No saved MLP models in {model_dir}; run python src/mlp.py first"
        )
    return max(paths, key=lambda path: int(path.stem.removeprefix("mlp_v")))


def prepare_train_test(
    data: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Clean features and create a stratified split, keeping singletons in train."""
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[LABEL_COLUMN])
    y = data[LABEL_COLUMN].astype(str).str.strip()
    X = data.drop(columns=[LABEL_COLUMN]).select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("No numeric feature columns found")
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")

    singleton_mask = y.map(y.value_counts()) == 1
    if singleton_mask.any():
        singleton_labels = sorted(y[singleton_mask].unique())
        print(
            "Keeping singleton classes in training (not represented in test): "
            + ", ".join(singleton_labels)
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X.loc[~singleton_mask],
            y.loc[~singleton_mask],
            test_size=test_size,
            random_state=random_state,
            stratify=y.loc[~singleton_mask],
        )
        X_train = pd.concat([X_train, X.loc[singleton_mask]])
        y_train = pd.concat([y_train, y.loc[singleton_mask]])
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    return X_train, X_test, y_train, y_test


def split_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out validation rows while keeping training singletons in fit data."""
    singleton_mask = y_train.map(y_train.value_counts()) == 1
    if singleton_mask.any():
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train.loc[~singleton_mask],
            y_train.loc[~singleton_mask],
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train.loc[~singleton_mask],
        )
        X_fit = pd.concat([X_fit, X_train.loc[singleton_mask]])
        y_fit = pd.concat([y_fit, y_train.loc[singleton_mask]])
        return X_fit, X_val, y_fit, y_val
    return train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )


def oversample_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample the selected attack classes to their median training count."""
    counts = y_train.value_counts()
    normalized_counts = {
        re.sub(r"[^a-z0-9]", "", label.casefold()): label for label in counts.index
    }
    selected: dict[str, str] = {}
    matched_keys: set[str] = set()
    for key, display_name in OVERSAMPLED_ATTACKS.items():
        aliases = ATTACK_ALIASES.get(key, (key,))
        matched_label = next(
            (normalized_counts[alias] for alias in aliases if alias in normalized_counts),
            None,
        )
        if matched_label is not None:
            selected[matched_label] = display_name
            matched_keys.add(key)

    missing = [
        display_name
        for key, display_name in OVERSAMPLED_ATTACKS.items()
        if key not in matched_keys
    ]
    if missing:
        print("Selected classes not found in training split: " + ", ".join(missing))
    if not selected:
        return X_train, y_train

    selected_labels = list(selected)
    target_count = int(counts.loc[selected_labels].max())
    print(f"Oversampling target for selected classes: {target_count:,} rows each")
    extra_X = [X_train]
    extra_y = [y_train]
    for label, display_name in selected.items():
        count = counts[label]
        if count < target_count:
            label_X = X_train.loc[y_train == label]
            label_y = y_train.loc[y_train == label]
            additional = target_count - int(count)
            extra_X.append(
                label_X.sample(n=additional, replace=True, random_state=random_state)
            )
            extra_y.append(
                label_y.sample(n=additional, replace=True, random_state=random_state)
            )
            safe_print(
                f"Oversampled {display_name}: "
                f"{count:,} -> {target_count:,} training rows"
            )

    X_balanced = pd.concat(extra_X, ignore_index=True)
    y_balanced = pd.concat(extra_y, ignore_index=True)
    order = np.random.default_rng(random_state).permutation(len(X_balanced))
    return X_balanced.iloc[order], y_balanced.iloc[order]


def build_model(random_state: int) -> Pipeline:
    """Create the MLP pipeline used by training and evaluation scripts."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                max_iter=60,
                early_stopping=False,
                alpha=0.001,
                    random_state=random_state,
                    verbose=True,
                ),
            ),
        ]
    )


def choose_class_thresholds(
    model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> dict[str, float]:
    """Choose one-vs-rest probability cutoffs maximizing validation F2."""
    probabilities = model.predict_proba(X_val)
    classes = model.named_steps["mlp"].classes_
    thresholds: dict[str, float] = {}
    for index, label in enumerate(classes):
        normalized = re.sub(r"[^a-z0-9]", "", str(label).casefold())
        if normalized not in LOW_RECALL_LABELS:
            continue
        actual = y_val.to_numpy() == label
        support = int(actual.sum())
        if support == 0:
            safe_print(f"No validation examples for {label!r}; keeping argmax behavior")
            continue

        precision, recall, candidates = precision_recall_curve(
            actual, probabilities[:, index]
        )
        beta_squared = 4.0
        f2 = (1 + beta_squared) * precision[:-1] * recall[:-1] / (
            beta_squared * precision[:-1] + recall[:-1] + 1e-12
        )
        best_score = f2.max()
        # On a tie, use the highest threshold to avoid unnecessary positives.
        best_index = np.flatnonzero(f2 == best_score)[-1]
        thresholds[normalized] = float(candidates[best_index])
        safe_print(
            f"Validation threshold for {label!r}: "
            f"{thresholds[normalized]:.6f} (support={support}, F2={best_score:.3f}, "
            f"precision={precision[best_index]:.3f}, recall={recall[best_index]:.3f})"
        )
    return thresholds


def predict_with_thresholds(
    model: Pipeline, X: pd.DataFrame, thresholds: dict[str, float]
) -> np.ndarray:
    """Apply calibrated one-vs-rest thresholds, falling back to argmax."""
    probabilities = model.predict_proba(X)
    classes = model.named_steps["mlp"].classes_
    baseline = np.argmax(probabilities, axis=1)
    eligible_scores = np.full_like(probabilities, -np.inf)

    for index, label in enumerate(classes):
        normalized = re.sub(r"[^a-z0-9]", "", str(label).casefold())
        if normalized in thresholds:
            threshold = max(thresholds[normalized], np.finfo(float).eps)
            eligible = probabilities[:, index] >= threshold
            eligible_scores[eligible, index] = probabilities[eligible, index] / threshold

    has_candidate = np.isfinite(eligible_scores).any(axis=1)
    chosen = baseline.copy()
    chosen[has_candidate] = np.argmax(eligible_scores[has_candidate], axis=1)
    return classes[chosen]


def load_data(data_dir: Path, max_rows_per_file: int, random_state: int) -> pd.DataFrame:
    """Load all CSVs, optionally taking a reproducible sample per file."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        if max_rows_per_file <= 0 or csv_path.name in FULL_DATA_FILES:
            frame = pd.read_csv(csv_path, low_memory=False)
            if csv_path.name in FULL_DATA_FILES and max_rows_per_file > 0:
                print(f"Loaded all {len(frame):,} rows from {csv_path.name}")
        else:
            # Uniform reservoir sampling keeps a representative sample across
            # the whole CSV, including files whose labels occur in blocks.
            rng = np.random.default_rng(random_state)
            reservoir = pd.DataFrame()
            seen = 0
            for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                previous_seen = seen
                seen += len(chunk)
                if seen <= max_rows_per_file:
                    reservoir = pd.concat([reservoir, chunk], ignore_index=True)
                else:
                    keep_from_old = int(
                        rng.hypergeometric(previous_seen, len(chunk), max_rows_per_file)
                    )
                    keep_from_chunk = max_rows_per_file - keep_from_old
                    old_sample = (
                        reservoir.sample(
                            n=keep_from_old,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        )
                        if keep_from_old
                        else reservoir.iloc[:0]
                    )
                    chunk_sample = chunk.sample(
                        n=keep_from_chunk,
                        random_state=int(rng.integers(0, 2**32 - 1)),
                    )
                    reservoir = pd.concat(
                        [old_sample, chunk_sample], ignore_index=True
                    )
            frame = reservoir
            print(f"Loaded {len(frame):,} of {seen:,} rows from {csv_path.name}")

        frame.columns = frame.columns.str.strip()
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"Expected a {LABEL_COLUMN!r} column in the CIC-IDS2017 CSVs")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=170_366,
        help="Rows sampled from each CSV; use 0 to load every row (default: 170366).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_data(args.data_dir, args.max_rows_per_file, args.random_state)
    X_train, X_test, y_train, y_test = prepare_train_test(
        data, args.test_size, args.random_state
    )
    X_fit, X_val, y_fit, y_val = split_validation(
        X_train, y_train, args.validation_size, args.random_state
    )
    print(
        f"Rows: {len(data):,}; fit: {len(X_fit):,}; validation: {len(X_val):,}; "
        f"test: {len(X_test):,}; features: {X_fit.shape[1]}; "
        f"classes: {data[LABEL_COLUMN].nunique()}"
    )
    X_fit, y_fit = oversample_training_data(X_fit, y_fit, args.random_state)
    model = build_model(args.random_state)
    model.fit(X_fit, y_fit)
    thresholds = choose_class_thresholds(model, X_val, y_val)
    model_path = save_model(model, thresholds)
    baseline_predictions = model.predict(X_test)
    adjusted_predictions = predict_with_thresholds(model, X_test, thresholds)
    print(f"Saved model to {model_path}")
    print(f"\nTest accuracy (argmax): {accuracy_score(y_test, baseline_predictions):.4f}")
    print("\nClassification report (argmax):")
    report = classification_report(y_test, baseline_predictions, zero_division=0)
    print(report.encode("ascii", errors="backslashreplace").decode("ascii"))
    print(
        "\nTest accuracy (validation-calibrated thresholds): "
        f"{accuracy_score(y_test, adjusted_predictions):.4f}"
    )
    print("\nClassification report (validation-calibrated thresholds):")
    report = classification_report(y_test, adjusted_predictions, zero_division=0)
    print(report.encode("ascii", errors="backslashreplace").decode("ascii"))


if __name__ == "__main__":
    main()
