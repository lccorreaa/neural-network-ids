"""Prepare CIC-IDS2017 flow CSVs as numeric features, binary labels, and metadata."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import warnings
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data/raw/MachineLearningCSV.zip")
LABEL_COLUMN = " Label"


def _unique_column_names(raw_header: list[str] | None) -> tuple[list[str], list[str]]:
    """Return unique pandas-safe names and names that duplicate an earlier header."""
    if not raw_header:
        raise ValueError("CSV is empty or has no header.")

    trimmed = [name.strip() for name in raw_header]
    counts: Counter[str] = Counter()
    unique_names: list[str] = []
    duplicate_names: list[str] = []
    for name in trimmed:
        counts[name] += 1
        if counts[name] == 1:
            unique_names.append(name)
        else:
            unique_names.append(f"{name}__duplicate_{counts[name]}")
            duplicate_names.append(unique_names[-1])
    return unique_names, duplicate_names


def _read_header(opener) -> tuple[list[str], list[str]]:
    """Read and normalize a CSV header from a path or ZIP member opener."""
    with opener() as raw_file:
        text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
        raw_header = next(csv.reader(text_file), None)
    return _unique_column_names(raw_header)


def prepare_frame(
    frame: pd.DataFrame,
    *,
    source_file: str,
    duplicate_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate model features, binary labels, and non-feature metadata.

    Labels are encoded as 0 for BENIGN and 1 for every attack. The exact source
    label and source filename are retained in metadata, not in the feature matrix.
    Non-numeric feature values are converted to NaN for handling after splitting.
    """
    frame = frame.copy()
    frame.columns = [str(name).strip() for name in frame.columns]

    label_matches = [name for name in frame.columns if name.casefold() == "label"]
    if len(label_matches) != 1:
        raise ValueError(
            f"Expected exactly one {LABEL_COLUMN!r} column; found {label_matches}."
        )
    label_column = label_matches[0]

    # This CIC-IDS2017 export repeats Fwd Header Length. Keep the first copy.
    duplicate_columns = duplicate_columns or []
    drop_columns = [name for name in duplicate_columns if name in frame.columns]
    frame = frame.drop(columns=drop_columns)

    attack_label = frame[label_column].astype("string").str.strip()
    valid_label = attack_label.notna() & attack_label.ne("")
    frame = frame.loc[valid_label].copy()
    attack_label = attack_label.loc[valid_label]

    feature_frame = frame.drop(columns=[label_column])
    numeric_features: dict[str, pd.Series] = {}
    invalid_counts: dict[str, int] = {}
    for column in feature_frame.columns:
        numeric = pd.to_numeric(feature_frame[column], errors="coerce")
        invalid = feature_frame[column].notna() & numeric.isna()
        if invalid.any():
            invalid_counts[column] = int(invalid.sum())
        # Float32 is sufficient for these flow measurements and halves the
        # memory needed for the multi-million-row feature matrix.
        numeric_features[column] = numeric.astype(np.float32)

    X = pd.DataFrame(numeric_features, index=frame.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    if invalid_counts:
        warnings.warn(
            "Some non-missing feature values were not numeric and became NaN: "
            f"{invalid_counts}",
            stacklevel=2,
        )

    y = attack_label.ne("BENIGN").astype("int64").rename("is_attack")
    metadata = pd.DataFrame(
        {
            "source_file": pd.Categorical([source_file] * len(attack_label)),
            "attack_label": pd.Categorical(attack_label),
        },
        index=frame.index,
    )
    return X, y, metadata


def load_dataset(
    data_source: Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load and prepare every CSV in an extracted folder or ZIP archive.

    This straightforward version holds the combined dataset in memory. The raw
    CSVs are only read; they are never rewritten.
    """
    expected_columns: list[str] | None = None
    feature_parts: list[pd.DataFrame] = []
    label_parts: list[pd.Series] = []
    metadata_parts: list[pd.DataFrame] = []

    if data_source.is_file() and is_zipfile(data_source):
        with ZipFile(data_source) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
            )
            sources = [
                (Path(name).name, lambda name=name: archive.open(name)) for name in members
            ]
            _load_sources(
                sources,
                feature_parts,
                label_parts,
                metadata_parts,
            )
    elif data_source.is_dir():
        paths = sorted(data_source.glob("*.csv"))
        sources = [
            (path.name, lambda path=path: path.open("rb")) for path in paths
        ]
        _load_sources(sources, feature_parts, label_parts, metadata_parts)
    else:
        raise FileNotFoundError(
            f"Expected a directory of CSVs or a ZIP archive, found: {data_source}"
        )

    X = pd.concat(feature_parts, ignore_index=True)
    y = pd.concat(label_parts, ignore_index=True)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    return X, y, metadata


def _load_sources(
    sources,
    feature_parts: list[pd.DataFrame],
    label_parts: list[pd.Series],
    metadata_parts: list[pd.DataFrame],
) -> None:
    """Read, validate, and prepare a sequence of named CSV openers."""
    if not sources:
        raise FileNotFoundError("No CSV files found in the selected data source.")

    expected_columns: list[str] | None = None
    first_name = sources[0][0]
    for name, opener in sources:
        columns, duplicate_columns = _read_header(opener)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(
                f"Column schema differs in {name}; compare it with {first_name}."
            )

        with opener() as csv_stream:
            frame = pd.read_csv(
                csv_stream,
                encoding="utf-8-sig",
                skiprows=1,
                names=columns,
                header=None,
                low_memory=False,
            )
        X_part, y_part, metadata_part = prepare_frame(
            frame,
            source_file=name,
            duplicate_columns=duplicate_columns,
        )
        feature_parts.append(X_part)
        label_parts.append(y_part)
        metadata_parts.append(metadata_part)


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    *,
    random_seed: int = 42,
) -> dict[str, object]:
    """Make leakage-conscious 70/15/15 splits and preprocess their features.

    Exact duplicate feature rows are assigned to one split using their row hash.
    The split proportions are approximate because duplicate groups stay intact.
    Median imputation and standardization are fitted on the training partition
    only, then applied unchanged to validation and test.
    """
    if not (len(X) == len(y) == len(metadata)):
        raise ValueError("X, y, and metadata must contain the same number of rows.")
    if X.empty:
        raise ValueError("X has no rows to split.")

    X = X.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(y).reset_index(drop=True).astype("int64")
    metadata = metadata.reset_index(drop=True)

    # Group identical feature vectors so copies cannot cross train/validation/test.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    y_values = y.to_numpy(dtype=np.int64, copy=False)
    # Hash identical feature vectors to a shared group. Assign groups by their
    # majority label, shuffling groups within each label to approximate a
    # stratified 70/15/15 split without allowing duplicate rows across splits.
    unique_groups, group_for_row = np.unique(groups, return_inverse=True)
    group_sizes = np.bincount(group_for_row, minlength=len(unique_groups))
    group_attacks = np.bincount(
        group_for_row,
        weights=y_values,
        minlength=len(unique_groups),
    )
    group_strata = (group_attacks >= (group_sizes / 2)).astype(np.int8)

    rng = np.random.default_rng(random_seed)
    group_partition = np.full(len(unique_groups), -1, dtype=np.int8)
    for class_value in (0, 1):
        class_groups = np.flatnonzero(group_strata == class_value)
        rng.shuffle(class_groups)
        class_rows = group_sizes[class_groups]
        midpoint_rows = np.cumsum(class_rows) - class_rows / 2
        class_total = class_rows.sum()
        group_partition[class_groups[midpoint_rows < class_total * 0.15]] = 2
        group_partition[
            class_groups[
                (midpoint_rows >= class_total * 0.15)
                & (midpoint_rows < class_total * 0.30)
            ]
        ] = 1
        group_partition[class_groups[midpoint_rows >= class_total * 0.30]] = 0

    row_partition = group_partition[group_for_row]
    train_indices = np.flatnonzero(row_partition == 0)
    validation_indices = np.flatnonzero(row_partition == 1)
    test_indices = np.flatnonzero(row_partition == 2)

    # Fit median imputation on training values only. Entirely missing columns
    # receive a median of zero so all feature columns remain available.
    X_values = X.to_numpy(dtype=np.float32, copy=False)
    train_values = X_values[train_indices].copy()
    validation_values = X_values[validation_indices].copy()
    test_values = X_values[test_indices].copy()
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians = np.nan_to_num(medians, nan=0.0)
    for values in (train_values, validation_values, test_values):
        missing_rows, missing_columns = np.where(np.isnan(values))
        values[missing_rows, missing_columns] = medians[missing_columns]

    # Fit means and standard deviations on imputed training values only.
    means = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scales = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    train_values = ((train_values - means) / scales).astype(np.float32, copy=False)
    validation_values = ((validation_values - means) / scales).astype(
        np.float32, copy=False
    )
    test_values = ((test_values - means) / scales).astype(np.float32, copy=False)

    return {
        "X_train": train_values,
        "y_train": y_values[train_indices],
        "metadata_train": metadata.iloc[train_indices].reset_index(drop=True),
        "X_validation": validation_values,
        "y_validation": y_values[validation_indices],
        "metadata_validation": metadata.iloc[validation_indices].reset_index(drop=True),
        "X_test": test_values,
        "y_test": y_values[test_indices],
        "metadata_test": metadata.iloc[test_indices].reset_index(drop=True),
        "imputer_medians": medians,
        "scaler_means": means,
        "scaler_scales": scales,
        "feature_names": X.columns.tolist(),
    }


def main() -> None:
    # CIC-IDS2017 may contain replacement characters that Windows cp1252 cannot
    # print. Escaping them keeps the report readable without altering the labels.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(
        description="Prepare CIC-IDS2017 CSVs into model inputs and separate labels."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing CIC-IDS2017 CSVs (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    X, y, metadata = load_dataset(args.data_dir)
    print(f"Features: {X.shape[1]} columns")
    print(f"Rows: {len(X):,}")
    print("Binary label counts (0=BENIGN, 1=attack):")
    print(y.value_counts().sort_index().to_string())
    print("\nOriginal attack label counts:")
    print(metadata["attack_label"].value_counts().to_string())
    print("\nRows by source file:")
    print(metadata["source_file"].value_counts().to_string())
    print(f"\nMissing/non-finite feature values: {int(X.isna().sum().sum()):,}")

    print("\nSplitting and preprocessing (this can take a few minutes)...")
    splits = split_and_preprocess(X, y, metadata)
    for split_name in ("train", "validation", "test"):
        split_y = splits[f"y_{split_name}"]
        benign = int(np.count_nonzero(split_y == 0))
        attacks = int(np.count_nonzero(split_y == 1))
        print(
            f"{split_name.title():10} {len(split_y):>9,} rows "
            f"({benign:,} benign, {attacks:,} attack)"
        )
    print("Preprocessing: median imputation and standard scaling fit on training data only.")
    print("Prepared feature arrays have dtype float32; labels have dtype int64.")


if __name__ == "__main__":
    main()
