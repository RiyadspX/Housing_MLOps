"""
prepare_data.py
================

This script reads the raw housing price dataset, performs basic cleaning and
feature engineering, and writes out processed training and test sets as CSV files.

The dataset comes from Kaggle's Housing Prices dataset and contains 545
observations with 13 columns. The columns include numeric features like
`price`, `area`, `bedrooms`, `bathrooms` and `stories` along with several
categorical indicators such as whether the house has access to a main road,
has a guest room, basement, hot‑water heating, air conditioning, parking,
and a preferred area.  A `furnishingstatus` variable describes whether the
property is furnished, semi‑furnished or unfurnished.  A quick summary of
the data shows there are no missing values and 545 rows in total【948383289304597†L35-L53】.

For a simple MLOps demonstration we perform the following steps:

* Load the raw CSV file into a pandas DataFrame.
* Convert binary categorical variables (e.g. `mainroad`, `guestroom`, etc.) to
  integer codes using pandas' ``astype('category').cat.codes``.
* One‑hot encode the multi‑category ``furnishingstatus`` variable using
  ``pd.get_dummies``.
* Split the data into training and test sets with a configurable test
  proportion.
* Write the processed training and test sets to disk for downstream stages.

Example usage from the command line:

.. code-block:: bash

   python src/prepare_data.py \
       --input-path data/raw/Housing.csv \
       --train-path data/processed/train.csv \
       --test-path data/processed/test.csv \
       --test-size 0.2 \
       --random-state 42

This script is intended to be used as a stage in a DVC pipeline.  DVC will
track the inputs and outputs automatically when the corresponding stage is
defined in ``dvc.yaml``.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load the raw housing data from a CSV file.

    Args:
        input_path: Path to the raw CSV file.

    Returns:
        A pandas DataFrame with the raw data.
    """
    df = pd.read_csv(input_path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing and feature engineering.

    Converts categorical variables to numeric codes and one‑hot encodes the
    ``furnishingstatus`` feature.  Leaves numeric features unchanged.

    Args:
        df: Raw DataFrame.

    Returns:
        Processed DataFrame suitable for modeling.
    """
    # Copy to avoid modifying the original data
    df = df.copy()

    # Convert binary categorical variables to numerical codes (0/1)
    binary_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    for col in binary_cols:
        # Some columns may already be numeric; category codes handle both
        df[col] = df[col].astype("category").cat.codes

    # Convert parking to integer (already numeric but ensure type)
    df["parking"] = df["parking"].astype(int)

    # One‑hot encode furnishingstatus (three categories)
    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    return df


def split_data(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and test sets.

    Args:
        df: Processed DataFrame.
        test_size: Fraction of the data to use for the test set.
        random_state: Seed for reproducibility.

    Returns:
        A tuple of (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV.

    Args:
        df: DataFrame to save.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare housing data for modeling")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/raw/Housing.csv"),
        help="Path to the raw input CSV file.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Output path for the processed training data.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Output path for the processed test data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df_raw = load_raw_data(args.input_path)
    df_processed = preprocess(df_raw)
    train_df, test_df = split_data(
        df_processed, test_size=args.test_size, random_state=args.random_state
    )
    save_data(train_df, args.train_path)
    save_data(test_df, args.test_path)


if __name__ == "__main__":
    main()