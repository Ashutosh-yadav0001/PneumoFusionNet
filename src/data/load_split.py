"""
src/data/load_split.py
======================
Helper to load the canonical patient-level split from ``patient_split.csv``
and merge it with any phase-specific master DataFrame.

Experiment 1 — Corrected patient-level split
--------------------------------------------
All phases must call ``load_patient_split()`` to obtain their train / val /
test DataFrames.  They must NOT create their own splits.

Public API
----------
load_patient_split(master_df, split_csv_path, subject_col)
    -> (train_df, val_df, test_df)

Usage in notebooks::

    from src.data.load_split import load_patient_split

    SPLIT_CSV = r'C:/2026/PneumoFusionNet/mimic/main/dataset/patient_split.csv'
    train_df, val_df, test_df = load_patient_split(df, SPLIT_CSV)
"""

from __future__ import annotations

import pandas as pd


def load_patient_split(
    master_df: pd.DataFrame,
    split_csv_path: str,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the canonical split and apply it to ``master_df``.

    Reads ``patient_split.csv`` (generated once by ``create_split.py``),
    merges on ``subject_col``, and returns the three split DataFrames.

    Args:
        master_df      (pd.DataFrame): The phase-specific DataFrame (may be a
                                       subset of the original split population).
        split_csv_path (str):  Absolute or relative path to
                               ``patient_split.csv``.
        subject_col    (str):  Patient ID column present in both ``master_df``
                               and ``patient_split.csv``. Default: ``"subject_id"``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            ``(train_df, val_df, test_df)`` with indices reset.

    Raises:
        FileNotFoundError: If ``split_csv_path`` does not exist.
        ValueError: If ``subject_col`` is missing from ``master_df`` or the
                    split CSV, or if any rows in ``master_df`` cannot be matched
                    to a split assignment.

    Example::

        train_df, val_df, test_df = load_patient_split(
            df,
            split_csv_path=r'C:/2026/PneumoFusionNet/mimic/main/dataset/patient_split.csv',
        )
    """
    import os

    if not os.path.isfile(split_csv_path):
        raise FileNotFoundError(
            f"patient_split.csv not found at: {split_csv_path}\n"
            "Run  python src/data/create_split.py  once to generate it."
        )

    split_map = pd.read_csv(split_csv_path)

    # ── Validate columns ──────────────────────────────────────────────────────
    if subject_col not in master_df.columns:
        raise ValueError(
            f"Column '{subject_col}' not found in master_df. "
            f"Available columns: {list(master_df.columns)}"
        )
    if subject_col not in split_map.columns:
        raise ValueError(
            f"Column '{subject_col}' not found in patient_split.csv. "
            f"CSV columns: {list(split_map.columns)}"
        )
    if "split" not in split_map.columns:
        raise ValueError(
            "Column 'split' not found in patient_split.csv. "
            "Re-generate using create_split.py."
        )

    # ── Merge split assignments onto master_df ────────────────────────────────
    merged = master_df.merge(
        split_map[[subject_col, "split"]],
        on=subject_col,
        how="left",
    )

    # ── Warn about unmatched rows ─────────────────────────────────────────────
    unmatched = merged["split"].isna().sum()
    if unmatched > 0:
        unmatched_ids = merged.loc[merged["split"].isna(), subject_col].unique()
        raise ValueError(
            f"{unmatched} rows in master_df have subject_ids not present in "
            f"patient_split.csv: {unmatched_ids[:10].tolist()} ..."
            "\nEnsure create_split.py was run on the same (or a superset) CSV."
        )

    # ── Partition into three DataFrames ───────────────────────────────────────
    train_df = (
        merged[merged["split"] == "train"]
        .drop(columns=["split"])
        .copy()
        .reset_index(drop=True)
    )
    val_df = (
        merged[merged["split"] == "val"]
        .drop(columns=["split"])
        .copy()
        .reset_index(drop=True)
    )
    test_df = (
        merged[merged["split"] == "test"]
        .drop(columns=["split"])
        .copy()
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df
