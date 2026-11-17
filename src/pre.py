from pathlib import Path
from typing import Iterator

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "MachineLearningCVE"
CHUNK_SIZE = 50_000


def iter_flow_chunks(
    data_dir: Path = DATA_DIR, chunksize: int = CHUNK_SIZE
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield (source file, chunk) pairs, keeping only one chunk in memory at a time."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for csv_path in csv_files:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
            # CIC-IDS2017 headers often contain leading/trailing spaces.
            chunk.columns = chunk.columns.str.strip()
            yield csv_path, chunk


def main() -> None:
    #Read just the first chunk for a quick structure and label audit
    first_file = next(iter(sorted(DATA_DIR.glob("*.csv"))), None)
    if first_file is None:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    chunk = next(
        pd.read_csv(first_file, chunksize=CHUNK_SIZE, low_memory=False)
    )
    chunk.columns = chunk.columns.str.strip()

    for col in chunk.select_dtypes(include=["integer"]).columns:
        chunk[col] = chunk[col].astype("int32")

    for col in chunk.select_dtypes(include=["floating"]).columns:
        chunk[col] = chunk[col].astype("float32")

    print(f"File: {first_file.name}")
    print(f"Chunk shape: {chunk.shape}")
    print("Columns and dtypes:")
    print(chunk.dtypes.to_string())

    print("\nNrows:")
    print(chunk.shape[0])
    # for chunk

    label_column = "Label"
    if label_column in chunk.columns:
        print("\nLabel counts in this chunk:")
        print(chunk[label_column].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
