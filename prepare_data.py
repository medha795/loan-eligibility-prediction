"""
Utility for engineering the combined loan dataset for modeling (chunked version).

This script cleans and standardizes the combined LendingClub dataset that
merges accepted and rejected applications. It processes the data in chunks to
avoid MemoryError on very large files (e.g., 30M+ rows).

Output:
    A single model-ready CSV with:
    - numeric features
    - one-hot encoded categorical indicators
    - optional binary target column (1=accepted, 0=rejected)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration / column lists
# -----------------------------

RAW_CATEGORICAL_COLUMNS = [
    "term",
    "emp_length",
    "home_ownership",
    "purpose",
    "addr_state",
    "application_type",
]

RAW_NUMERIC_COLUMNS = [
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_high",
    "fico_range_low",
    "inq_last_6mths",
]

TARGET_COLUMN = "loan_status"


# -----------------------------
# Helper parsing functions
# -----------------------------

def _parse_term(term_value: str) -> float:
    """Convert terms like '36 months' to the numeric month count."""
    if pd.isna(term_value):
        return np.nan
    digits = "".join(ch for ch in str(term_value) if ch.isdigit())
    return float(digits) if digits else np.nan


def _parse_emp_length(emp_length_value: str) -> float:
    """Normalize employment length strings to a numeric year count."""
    if pd.isna(emp_length_value):
        return np.nan

    value = str(emp_length_value).strip().lower()
    if value in {"10+ years", "10 years", "10+"}:
        return 10.0
    if value in {"< 1 year", "<1 year", "<1"}:
        return 0.0
    digits = "".join(ch for ch in value if ch.isdigit())
    return float(digits) if digits else np.nan


def _parse_percentage(value: str) -> float:
    """Strip percent signs and cast to float."""
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace("%", "").strip()
    return float(cleaned) if cleaned else np.nan


def _build_fico_score(df: pd.DataFrame) -> pd.Series:
    """Combine high/low ranges into a single representative score."""
    low = df.get("fico_range_low")
    high = df.get("fico_range_high")

    if low is not None and high is not None:
        return (pd.to_numeric(low, errors="coerce") +
                pd.to_numeric(high, errors="coerce")) / 2
    if low is not None:
        return pd.to_numeric(low, errors="coerce")
    if high is not None:
        return pd.to_numeric(high, errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip column names for consistency."""
    normalized = df.copy()
    normalized.columns = normalized.columns.str.lower().str.strip()
    return normalized


# -----------------------------
# Core feature engineering
# -----------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned, model-ready dataframe for a given chunk.

    Steps
    -----
    * Standardize column casing
    * Parse text percentages and term/tenure fields to numerics
    * Build a single fico_score
    * Impute missing values with medians (numeric) and most-common labels (categorical)
    * One-hot encode categorical features
    * Convert the target label to a binary indicator (1=accepted, 0=rejected)
    """
    # Normalize columns
    df = _normalize_columns(df)
    working = df.copy()

    # ----- Derived numeric fields -----
    if "term" in working.columns:
        working["term_months"] = working["term"].apply(_parse_term)

    if "emp_length" in working.columns:
        working["emp_length_years"] = working["emp_length"].apply(_parse_emp_length)

    if "int_rate" in working.columns:
        working["int_rate"] = working["int_rate"].apply(_parse_percentage)

    if "dti" in working.columns:
        working["dti"] = working["dti"].apply(_parse_percentage)

    working["fico_score"] = _build_fico_score(working)

    # ----- Target encoding -----
    if TARGET_COLUMN in working.columns:
        working[TARGET_COLUMN] = (
            working[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"accepted": 1, "rejected": 0})
        )
        # Keep only rows where the label is valid (0 or 1)
        mask_valid = working[TARGET_COLUMN].isin([0, 1])
        working = working[mask_valid].copy()

    # ----- Identify numeric & categorical features -----
    numeric_features: List[str] = []
    for column in RAW_NUMERIC_COLUMNS + ["term_months", "emp_length_years", "fico_score"]:
        if column in working.columns:
            numeric_features.append(column)
            working[column] = pd.to_numeric(working[column], errors="coerce")

    categorical_features: List[str] = []
    for column in RAW_CATEGORICAL_COLUMNS:
        if column in working.columns:
            categorical_features.append(column)
            working[column] = working[column].astype(str).str.strip().str.lower()

    # ----- Impute missing values -----
    if numeric_features:
        for column in numeric_features:
            median_val = working[column].median()
            working[column] = working[column].fillna(median_val)

    if categorical_features:
        for column in categorical_features:
            working[column] = working[column].replace({"nan": np.nan})
            mode = working[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            working[column] = working[column].fillna(fill_value)

    # ----- One-hot encode categoricals -----
    if categorical_features:
        encoded = pd.get_dummies(
            working[categorical_features],
            prefix=categorical_features,
            drop_first=False,
        )
    else:
        encoded = pd.DataFrame(index=working.index)

    # ----- Assemble final dataframe -----
    features_part = pd.concat(
        [working[numeric_features], encoded],
        axis=1,
    )

    if TARGET_COLUMN in working.columns:
        features_part[TARGET_COLUMN] = working[TARGET_COLUMN].astype(int)

    return features_part


# -----------------------------
# Chunked processing utilities
# -----------------------------

def engineer_sample_and_get_schema(
    input_path: Path,
    sample_rows: int,
) -> Tuple[pd.DataFrame, list]:
    """
    Read a sample from the big CSV, engineer features, and extract
    the canonical column order (schema) to use for all subsequent chunks.
    """
    print(f"📥 Reading sample (up to {sample_rows:,} rows) from: {input_path}")
    df_sample = pd.read_csv(input_path, nrows=sample_rows, low_memory=False)
    print(f"   Sample raw shape: {df_sample.shape}")

    engineered_sample = engineer_features(df_sample)
    print(f"✅ Engineered sample shape: {engineered_sample.shape}")

    schema_columns = engineered_sample.columns.tolist()
    return engineered_sample, schema_columns


def process_full_dataset_chunked(
    input_path: Path,
    output_path: Path,
    sample_rows: int = 500_000,
    chunk_size: int = 100_000,
) -> None:
    """
    Engineer the entire dataset in a memory-safe way.

    Strategy:
        1. Read a sample of `sample_rows`, engineer features, and infer schema.
        2. Write the engineered sample to the output file (with header).
        3. Re-read the CSV in chunks, skipping the rows used for the sample.
        4. For each chunk: engineer features, align columns to the schema, append.

    Note:
        Categories that never appear in the sample but appear later in the file
        will be dropped (their dummy columns won't be in the schema). Using a
        reasonably large sample (e.g., 500k rows) minimizes this risk.
    """
    # 1) Sample & schema
    engineered_sample, schema_columns = engineer_sample_and_get_schema(
        input_path=input_path,
        sample_rows=sample_rows,
    )

    # 2) Write the sample to output with header
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Writing engineered sample to: {output_path}")
    engineered_sample.to_csv(output_path, index=False)

    # 3) Iterate over remaining chunks
    print("\n🚚 Processing remaining chunks...")
    total_rows_processed = len(engineered_sample)

    # skiprows: skip data rows used in the sample (row indices 1..sample_rows),
    # but keep the header row (0).
    skip_rows = range(1, sample_rows + 1)

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        skiprows=skip_rows,
    )

    chunk_idx = 0
    for chunk in reader:
        chunk_idx += 1
        print(f"  🔹 Chunk {chunk_idx}: raw shape {chunk.shape}")

        engineered_chunk = engineer_features(chunk)
        if engineered_chunk.empty:
            print("     (chunk produced no valid rows after label mapping; skipping)")
            continue

        # Align columns to schema: keep only schema columns, fill missing with 0
        engineered_chunk = engineered_chunk.reindex(columns=schema_columns, fill_value=0)

        # Append to CSV (no header)
        engineered_chunk.to_csv(
            output_path,
            mode="a",
            index=False,
            header=False,
        )

        total_rows_processed += len(engineered_chunk)
        print(f"     -> engineered shape {engineered_chunk.shape}, "
              f"total rows written so far: {total_rows_processed:,}")

    print("\n✅ Completed processing.")
    print(f"   Final engineered dataset written to: {output_path}")
    print(f"   Approx. total rows (after cleaning): {total_rows_processed:,}")


# -----------------------------
# CLI interface
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Engineer the combined LendingClub dataset into model-ready features "
            "using chunked processing to avoid MemoryError."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/combined_loan_data_processed.csv"),
        help="Path to the combined_loan_data_processed CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/engineered_loan_dataset.csv"),
        help="Where to save the engineered dataset.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=500_000,
        help="Number of initial rows to sample for inferring the schema.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk when processing the full file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("🔧 Configuration:")
    print(f"   Input path   : {args.input}")
    print(f"   Output path  : {args.output}")
    print(f"   Sample rows  : {args.sample_rows:,}")
    print(f"   Chunk size   : {args.chunk_size:,}\n")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found at: {args.input}")

    process_full_dataset_chunked(
        input_path=args.input,
        output_path=args.output,
        sample_rows=args.sample_rows,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
