"""
src/data/preprocessing.py
==========================
Image and text preprocessing utilities for PneumoFusionNet.

Compatible with Python 3.8+.

Contains:
    apply_clahe           — CLAHE contrast enhancement for CXR images
    compute_lung_bbox     — Lung bounding-box detection via Otsu thresholding
    build_report_text     — Anti-leakage text construction from report fields
    get_train_transforms  — Training image transform pipeline
    get_val_transforms    — Validation / test image transform pipeline
    get_tta_transforms    — 10-view Test-Time Augmentation transform list
    compute_dataset_stats — Estimate pixel mean and std from a DataFrame sample

Source notebooks:
    mimic/main/Phase-1/Phase-1.1v4-crossval_tta_PA.ipynb
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms

# ---------------------------------------------------------------------------
# Default image statistics (computed over MIMIC-CXR CLAHE-processed images)
# ---------------------------------------------------------------------------
DEFAULT_MEAN = [0.5020]
DEFAULT_STD  = [0.2703]
DEFAULT_IMG_SIZE = 320   # Phase 2 / 3 resolution (Phase 1 uses 224)

# ---------------------------------------------------------------------------
# Diagnostic keyword redaction regex (anti-leakage protocol)
# ---------------------------------------------------------------------------
_LEAKAGE_RE = re.compile(
    r"\bpneumonia\b|\bpneumonic\b|\bno[ -]acute[ -]\w+"
    r"|\bno finding\w*|\bcompatible with\b|\bconsistent with\b"
    r"|\bnormal study\b|\bno significant\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast in chest X-ray images, making lung tissue
    boundaries and infiltrates more distinguishable.

    Args:
        image (np.ndarray): Grayscale uint8 image array (H, W).
        clip_limit (float): CLAHE clip limit. Default: 2.0.
        tile_grid (tuple): Tile grid size (rows, cols). Default: (8, 8).

    Returns:
        np.ndarray: CLAHE-enhanced grayscale image (H, W).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(image)


def compute_lung_bbox(
    image_path: str,
    margin: int = 10,
) -> Optional[dict]:
    """Detect lung bounding box using Otsu thresholding and contour detection.

    Applies CLAHE, Gaussian blur, Otsu's threshold, and contour analysis
    to estimate the minimal bounding rectangle around lung regions.

    Args:
        image_path (str): Absolute path to the chest X-ray JPEG/PNG.
        margin (int): Extra pixel padding around detected bounding box.
                      Default: 10.

    Returns:
        dict | None: Bounding box ``{'x_min', 'y_min', 'x_max', 'y_max'}``
                     or ``None`` if the image cannot be loaded.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        h, w = img.shape

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img)
        blurred = cv2.GaussianBlur(img_eq, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        min_area = h * w * 0.05
        lung_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if not lung_contours:
            # Fallback: centre crop 80% of image
            cx, cy = w // 2, h // 2
            crop_w, crop_h = int(w * 0.8), int(h * 0.8)
            return {
                "x_min": cx - crop_w // 2,
                "y_min": cy - crop_h // 2,
                "x_max": cx + crop_w // 2,
                "y_max": cy + crop_h // 2,
            }

        all_points = np.vstack(lung_contours)
        x, y, bw, bh = cv2.boundingRect(all_points)
        return {
            "x_min": max(0, x - margin),
            "y_min": max(0, y - margin),
            "x_max": min(w, x + bw + margin),
            "y_max": min(h, y + bh + margin),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text preprocessing (anti-leakage protocol)
# ---------------------------------------------------------------------------

def build_report_text(row: pd.Series) -> str:
    """Build leakage-safe clinical text from a radiology report row.

    Combines ``indication`` (clinical history) and ``findings`` sections,
    deliberately **excluding** the ``IMPRESSION`` / ``CONCLUSION`` section
    which contains the final diagnostic label.  Diagnostic trigger keywords
    are redacted with ``[REDACTED]``.

    Anti-leakage actions:
        1. IMPRESSION section is entirely excluded.
        2. Keyword redaction via regex (e.g. 'pneumonia', 'normal study').
        3. Fallback to first 512 chars of ``raw_report`` if both fields empty.

    Args:
        row (pd.Series): DataFrame row with keys ``indication``, ``findings``,
                         and optionally ``raw_report``.

    Returns:
        str: Sanitised, leakage-free clinical text string.
    """
    parts = []
    if len(str(row.get("indication", ""))) > 5:
        parts.append(str(row["indication"]).strip())
    if len(str(row.get("findings", ""))) > 5:
        parts.append(str(row["findings"]).strip())

    result = " ".join(parts).strip()

    if not result:
        raw = str(row.get("raw_report", ""))
        result = raw[:512] if raw else "no report available"

    # Redact diagnostic trigger phrases
    result = _LEAKAGE_RE.sub("[REDACTED]", result)
    return result


# ---------------------------------------------------------------------------
# Image Transform Pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    mean: list = DEFAULT_MEAN,
    std: list = DEFAULT_STD,
) -> transforms.Compose:
    """Training image augmentation pipeline for chest X-rays.

    Augmentations are carefully chosen to be medically safe:
    - No horizontal flip (anatomical laterality matters in CXR).
    - Only small rotations (±10°) to mimic patient positioning variance.
    - Mild brightness / contrast jitter and Gaussian blur for robustness.

    Args:
        img_size (int): Target square image size in pixels. Default: 320.
        mean (list): Per-channel normalisation mean. Default: [0.5020].
        std  (list): Per-channel normalisation std.  Default: [0.2703].

    Returns:
        transforms.Compose: Training transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    img_size: int = DEFAULT_IMG_SIZE,
    mean: list = DEFAULT_MEAN,
    std: list = DEFAULT_STD,
) -> transforms.Compose:
    """Validation / test image transform pipeline (no augmentation).

    Args:
        img_size (int): Target square image size in pixels. Default: 320.
        mean (list): Per-channel normalisation mean.
        std  (list): Per-channel normalisation std.

    Returns:
        transforms.Compose: Deterministic validation transform.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_tta_transforms(
    img_size: int = 224,
    mean: list = DEFAULT_MEAN,
    std: list = DEFAULT_STD,
) -> list:
    """Build 10 medically-safe Test-Time Augmentation (TTA) transforms.

    Each transform produces a slightly different view of the same image.
    Probabilities from all views are averaged at inference.

    Views:
        0: Original (identity)
        1: Rotation +5°
        2: Rotation -5°
        3: Centre-crop zoom
        4: Brightness +15%
        5: Contrast +15%
        6: Small translation
        7: Rotation +3°
        8: Rotation -3°
        9: Gaussian blur

    Args:
        img_size (int): Image size (Phase 1 uses 224). Default: 224.
        mean (list): Normalisation mean.
        std  (list): Normalisation std.

    Returns:
        list[transforms.Compose]: List of 10 transform pipelines.
    """
    base = [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]

    def _make(extra):
        return transforms.Compose(
            [transforms.Resize((img_size, img_size))] + extra +
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    return [
        transforms.Compose(base),                                        # 0: original
        _make([transforms.RandomRotation((5, 5))]),                      # 1: +5°
        _make([transforms.RandomRotation((-5, -5))]),                    # 2: -5°
        _make([transforms.Resize((256, 256)), transforms.CenterCrop(210),
               transforms.Resize((img_size, img_size))]),                # 3: zoom
        _make([transforms.ColorJitter(brightness=0.15)]),                # 4: brightness
        _make([transforms.ColorJitter(contrast=0.15)]),                  # 5: contrast
        _make([transforms.RandomAffine(degrees=0, translate=(0.03, 0.03))]),  # 6: translate
        _make([transforms.RandomRotation((3, 3))]),                      # 7: +3°
        _make([transforms.RandomRotation((-3, -3))]),                    # 8: -3°
        _make([transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.5))]),    # 9: blur
    ]


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(
    df: "pd.DataFrame",
    n_samples: int = 200,
    bbox_lookup: dict = None,
) -> Tuple[list, list]:
    """Estimate mean and std pixel values from a sample of CLAHE images.

    Iterates over up to ``n_samples`` rows in ``df``, applies CLAHE, optional
    bounding-box crop, and accumulates per-image statistics.

    Args:
        df (pd.DataFrame): DataFrame with an ``image_path`` column.
        n_samples (int): Number of images to sample. Default: 200.
        bbox_lookup (dict): Optional mapping of image_path → bbox dict.

    Returns:
        tuple[list, list]: ``(mean, std)`` each as a single-element list
                           suitable for ``transforms.Normalize``.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    means, stds = [], []
    bbox_lookup = bbox_lookup or {}

    for idx in range(min(len(df), n_samples)):
        try:
            path = df.iloc[idx]["image_path"]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = clahe.apply(img)
            bb = bbox_lookup.get(path, None)
            if bb and bb.get("x_max", 0) > 0:
                img = img[bb["y_min"]:bb["y_max"], bb["x_min"]:bb["x_max"]]
            arr = np.array(img, dtype=np.float32) / 255.0
            means.append(arr.mean())
            stds.append(arr.std())
        except Exception:
            continue

    mean = [float(np.mean(means))] if means else DEFAULT_MEAN
    std  = [float(np.mean(stds))]  if stds  else DEFAULT_STD
    print(f"[compute_dataset_stats] mean={mean[0]:.4f}  std={std[0]:.4f}  (n={len(means)})")
    return mean, std
