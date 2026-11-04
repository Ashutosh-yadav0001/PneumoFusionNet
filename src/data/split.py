"""
src/data/split.py
=================
Canonical patient-level train / val / test split for PneumoFusionNet.

Experiment 1 — Corrected patient-level split
--------------------------------------------
All three phases (image-only, image+text, image+text+clinical) must share
*exactly* the same split so that cross-phase metric comparisons are valid.

Public API
----------
make_patient_split(df, subject_col, label_col, seed)
    -> (train_df, val_df, test_df)

Design decisions
----------------
* Split is performed on **unique patients** (subject_id), not individual rows.
* Uses ``StratifiedShuffleSplit`` at the patient level so that each split
  contains a representative pneumonia / normal ratio.
* Two-stage split:
    Stage 1: isolate test  (~15% of patients)
    Stage 2: split remainder into train (~70%) and val (~15%)
* The patient-to-split mapping is returned via the DataFrames; call
  ``create_split.py`` once to persist it as a CSV.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def _majority_label(group: pd.Series) -> int:
    """Return the majority label for a patient who has multiple rows."""
    return int(group.mode().iloc[0])


def make_patient_split(
    df: pd.DataFrame,
    subject_col: str = "subject_id",
    label_col: str = "label",
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a stratified patient-level train / val / test split.

    The split is performed on unique patients (``subject_col``), NOT on
    individual radiograph rows, to prevent the same patient appearing in
    multiple sets.

    Args:
        df          (pd.DataFrame): Master DataFrame with at least
                                    ``subject_col`` and ``label_col`` columns.
        subject_col (str):  Column name for the patient identifier.
                            Default: ``"subject_id"``.
        label_col   (str):  Column name for the binary label (0/1).
                            Default: ``"label"``.
        seed        (int):  Random seed for reproducibility. Default: ``42``.
        test_size   (float): Fraction of patients to allocate to the test set.
                             Default: ``0.15``.
        val_size    (float): Fraction of patients to allocate to the val set.
                             Default: ``0.15``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            ``(train_df, val_df, test_df)`` — row-level DataFrames with
            the original columns preserved. Indices are reset.

    Raises:
        ValueError: If ``subject_col`` or ``label_col`` is not in ``df``.
        ValueError: If a patient in ``df`` is not covered by the split
                    (should not happen under normal usage).

    Example::

        train_df, val_df, test_df = make_patient_split(
            df, subject_col='subject_id', label_col='label', seed=42
        )
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    for col in (subject_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # ── Build a one-row-per-patient table with their majority label ───────────
    patient_labels = (
        df.groupby(subject_col)[label_col]
        .apply(_majority_label)
        .reset_index()
        .rename(columns={label_col: "_patient_label"})
    )

    patients = patient_labels[subject_col].values
    strat_labels = patient_labels["_patient_label"].values

    # ── Stage 1: isolate test set (~15% of patients) ──────────────────────────
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    trainval_idx, test_idx = next(
        sss_test.split(patients, strat_labels)
    )

    trainval_patients = patients[trainval_idx]
    trainval_labels   = strat_labels[trainval_idx]
    test_patients     = patients[test_idx]

    # ── Stage 2: split remainder into train and val ───────────────────────────
    # val_size as a fraction of the remaining (trainval) patients:
    #   val_frac = val_size / (1 - test_size)
    val_frac = val_size / (1.0 - test_size)

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=seed
    )
    train_idx, val_idx = next(
        sss_val.split(trainval_patients, trainval_labels)
    )

    train_patients = trainval_patients[train_idx]
    val_patients   = trainval_patients[val_idx]

    # ── Expand patient lists back to row level ────────────────────────────────
    train_df = df[df[subject_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df   = df[df[subject_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df  = df[df[subject_col].isin(test_patients)].copy().reset_index(drop=True)

    # ── Sanity: every row in df must appear in exactly one split ──────────────
    covered = len(train_df) + len(val_df) + len(test_df)
    if covered != len(df):
        raise ValueError(
            f"Split coverage mismatch: {covered} rows assigned but df has {len(df)} rows. "
            "Some patients may have been dropped — check for NaN subject_ids."
        )

    return train_df, val_df, test_df


def print_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subject_col: str = "subject_id",
    label_col: str = "label",
) -> None:
    """Print a formatted split summary to stdout.

    Args:
        train_df    (pd.DataFrame): Training split.
        val_df      (pd.DataFrame): Validation split.
        test_df     (pd.DataFrame): Test split.
        subject_col (str): Patient ID column.
        label_col   (str): Binary label column.
    """
    header = f"{'Split':<8} {'Patients':>9} {'Rows':>7} {'Pneumonia':>10} {'Normal':>8} {'Pneu%':>7}"
    print("=" * 55)
    print("  Patient-Level Split Report")
    print("=" * 55)
    print(header)
    print("-" * 55)

    all_patients = set()
    for name, sdf in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_patients = sdf[subject_col].nunique()
        n_rows     = len(sdf)
        n_pneu     = int((sdf[label_col] == 1).sum())
        n_norm     = int((sdf[label_col] == 0).sum())
        pct        = 100.0 * n_pneu / n_rows if n_rows else 0.0
        print(f"  {name:<6} {n_patients:>9,} {n_rows:>7,} {n_pneu:>10,} {n_norm:>8,} {pct:>6.1f}%")
        all_patients.update(sdf[subject_col].tolist())

    total_patients = (
        train_df[subject_col].nunique()
        + val_df[subject_col].nunique()
        + test_df[subject_col].nunique()
    )
    total_rows = len(train_df) + len(val_df) + len(test_df)
    total_pneu = int(
        (train_df[label_col] == 1).sum()
        + (val_df[label_col] == 1).sum()
        + (test_df[label_col] == 1).sum()
    )
    total_norm = total_rows - total_pneu
    total_pct  = 100.0 * total_pneu / total_rows if total_rows else 0.0
    print("-" * 55)
    print(
        f"  {'Total':<6} {total_patients:>9,} {total_rows:>7,} "
        f"{total_pneu:>10,} {total_norm:>8,} {total_pct:>6.1f}%"
    )
    print("=" * 55)

    # Overlap check
    train_sids = set(train_df[subject_col])
    val_sids   = set(val_df[subject_col])
    test_sids  = set(test_df[subject_col])
    overlaps = (
        len(train_sids & val_sids)
        + len(train_sids & test_sids)
        + len(val_sids & test_sids)
    )
    status = "PASS" if overlaps == 0 else f"FAIL ({overlaps} overlapping patients)"
    print(f"  No patient overlap across splits : {status}")
    print("=" * 55)
