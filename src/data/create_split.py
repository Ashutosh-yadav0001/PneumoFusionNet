"""
src/data/create_split.py
========================
One-time runner that generates the canonical patient-level split CSV.

Experiment 1 — Corrected patient-level split
--------------------------------------------
Run this script **once** from the project root:

    python src/data/create_split.py

It reads the Phase-3 master CSV (the richest superset of all patients),
calls ``make_patient_split()``, and writes:

    mimic/main/dataset/patient_split.csv

with columns:
    subject_id  (int)  — MIMIC patient identifier
    split       (str)  — one of "train", "val", "test"

All subsequent notebooks load this file via ``load_patient_split()``.

CAUTION
-------
Re-running this script with a different seed **invalidates** all prior
experiment results.  The file is intentionally versioned in git so that
the split is locked for the lifetime of the project.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — adjust BASE_DIR if running from a different working directory
# ---------------------------------------------------------------------------
BASE_DIR   = r"C:\2026\PneumoFusionNet\mimic"
MASTER_CSV = os.path.join(BASE_DIR, "main", "dataset", "phase3_paired_scaleup_final.csv")
OUT_CSV    = os.path.join(BASE_DIR, "main", "dataset", "patient_split.csv")

SEED       = 42
TEST_SIZE  = 0.15
VAL_SIZE   = 0.15

# ---------------------------------------------------------------------------
# Allow running from project root (src/ not on PYTHONPATH by default)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.split import make_patient_split, print_split_report  # noqa: E402


def main() -> None:
    # ── Guard: do not overwrite an existing split without explicit deletion ───
    if os.path.isfile(OUT_CSV):
        print(f"[WARNING] {OUT_CSV} already exists.")
        print("  If you intentionally want to regenerate the split, delete the")
        print("  file first, then re-run this script.")
        print("  Exiting without changes.")
        sys.exit(0)

    # ── Load master CSV ───────────────────────────────────────────────────────
    print(f"Loading master CSV: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    print(f"  Rows: {len(df):,}  |  Patients: {df['subject_id'].nunique():,}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # Drop rows with missing label or missing subject_id
    before = len(df)
    df = df[df["label"].notna() & df["subject_id"].notna()].copy()
    df["label"] = df["label"].astype(int)
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing label/subject_id.")

    # ── Create the patient-level split ────────────────────────────────────────
    print(f"\nCreating stratified patient-level split (seed={SEED}) ...")
    print(f"  Target: ~{int((1-TEST_SIZE-VAL_SIZE)*100)}% train / "
          f"~{int(VAL_SIZE*100)}% val / ~{int(TEST_SIZE*100)}% test  (at patient level)")

    train_df, val_df, test_df = make_patient_split(
        df,
        subject_col="subject_id",
        label_col="label",
        seed=SEED,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
    )

    # ── Print report ──────────────────────────────────────────────────────────
    print()
    print_split_report(train_df, val_df, test_df,
                       subject_col="subject_id", label_col="label")

    # ── Build the mapping table (one row per patient) ─────────────────────────
    train_map = pd.DataFrame({"subject_id": train_df["subject_id"].unique(), "split": "train"})
    val_map   = pd.DataFrame({"subject_id": val_df["subject_id"].unique(),   "split": "val"})
    test_map  = pd.DataFrame({"subject_id": test_df["subject_id"].unique(),  "split": "test"})
    split_map = pd.concat([train_map, val_map, test_map], ignore_index=True)
    split_map["subject_id"] = split_map["subject_id"].astype(int)

    # ── Verify no duplicate patient assignments ───────────────────────────────
    assert split_map["subject_id"].is_unique, \
        "BUG: a subject_id appears more than once in the split map!"

    # ── Save ──────────────────────────────────────────────────────────────────
    split_map.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(f"  Rows in CSV : {len(split_map):,} (one per unique patient)")
    print(f"  Splits      : {split_map['split'].value_counts().to_dict()}")
    print("\nDone. All notebooks should now load this file via load_patient_split().")


if __name__ == "__main__":
    main()
