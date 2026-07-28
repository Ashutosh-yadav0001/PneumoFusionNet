"""
src/data/dataset.py
===================
PyTorch Dataset classes for PneumoFusionNet.

Contains:
    CXRDataset           — Phase 1: image-only (CLAHE + CBAM + bbox crop)
    MultimodalCXRDataset — Phase 2: image + radiology text pairs
    TripleModalCXRDataset— Phase 3: image + text + clinical metadata

All dataset classes:
    - Apply CLAHE contrast enhancement on grayscale CXR images.
    - Support optional lung bounding-box cropping via ``bbox_lookup`` dict.
    - Expose ``set_transform()`` for Test-Time Augmentation (TTA) view swapping.

Source notebooks:
    mimic/main/Phase-1/Phase-1.1v4-crossval_tta_PA.ipynb
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
    mimic/main/Phase-3/Phase-3-triple_fusion_PA_Scaleup.ipynb
"""

from __future__ import annotations

import numpy as np
import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Phase 1 — Image-only Dataset
# ---------------------------------------------------------------------------

class CXRDataset(Dataset):
    """Phase 1 chest X-ray dataset (image-only).

    Loads single-channel grayscale JPEG images, applies CLAHE contrast
    enhancement, optionally crops to a pre-computed lung bounding box,
    converts to PIL, and applies the provided transform pipeline.

    Args:
        df          (pd.DataFrame): DataFrame with columns ``image_path`` and ``label``.
        bbox_lookup (dict): Mapping ``image_path → {x_min, y_min, x_max, y_max}``.
                            Pass empty dict ``{}`` to disable cropping.
        transform   (callable | None): Torchvision transform pipeline.

    Returns per item:
        tuple[torch.Tensor, int]: ``(image_tensor, label)``

    Example::

        ds = CXRDataset(df, bbox_lookup, transform=get_val_transforms())
        img, label = ds[0]
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        bbox_lookup: dict,
        transform=None,
    ):
        self.df          = df.reset_index(drop=True)
        self.bbox_lookup = bbox_lookup
        self.transform   = transform
        self.clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
        img = self.clahe.apply(img)

        bb = self.bbox_lookup.get(row.image_path, None)
        if bb and bb.get("x_max", 0) > 0:
            img = img[bb["y_min"]:bb["y_max"], bb["x_min"]:bb["x_max"]]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, int(row.label)

    def set_transform(self, transform) -> None:
        """Swap the transform pipeline (used for TTA view switching).

        Args:
            transform: New torchvision transform to apply.
        """
        self.transform = transform


# ---------------------------------------------------------------------------
# Phase 2 — Image + Text Multimodal Dataset
# ---------------------------------------------------------------------------

class MultimodalCXRDataset(Dataset):
    """Phase 2 multimodal dataset: chest X-ray + radiology report text.

    Loads CXR images with CLAHE and optional bbox cropping, and tokenises
    the ``report_rich`` column (pre-processed with anti-leakage text builder)
    using a Bio_ClinicalBERT tokenizer.

    Args:
        df          (pd.DataFrame): DataFrame with ``image_path``, ``report_rich``,
                                    and ``label`` columns.
        bbox_lookup (dict): Lung bounding-box lookup dictionary.
        tokenizer:  Bio_ClinicalBERT HuggingFace tokenizer.
        img_transform: Torchvision transform pipeline.
        max_len     (int): Maximum token sequence length. Default: 256.

    Returns per item:
        tuple: ``(image_tensor, input_ids, attention_mask, label)``

    Example::

        ds = MultimodalCXRDataset(df, bbox_lookup, tokenizer,
                                   img_transform=get_val_transforms())
        img, ids, mask, label = ds[0]
    """

    def __init__(
        self,
        df:            pd.DataFrame,
        bbox_lookup:   dict,
        tokenizer,
        img_transform,
        max_len:       int = 256,
    ):
        self.df          = df.reset_index(drop=True)
        self.bbox_lookup = bbox_lookup
        self.tokenizer   = tokenizer
        self.transform   = img_transform
        self.max_len     = max_len
        self.clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image
        img = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
        img = self.clahe.apply(img)
        bb  = self.bbox_lookup.get(row.image_path, None)
        if bb and bb.get("x_max", 0) > 0:
            img = img[bb["y_min"]:bb["y_max"], bb["x_min"]:bb["x_max"]]
        img = self.transform(Image.fromarray(img))

        # Text
        enc = self.tokenizer(
            row.report_rich,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            img,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            int(row.label),
        )

    def set_transform(self, transform) -> None:
        """Swap the image transform (used for TTA)."""
        self.transform = transform


# ---------------------------------------------------------------------------
# Phase 3 — Image + Text + Clinical Metadata Dataset
# ---------------------------------------------------------------------------

class TripleModalCXRDataset(Dataset):
    """Phase 3 triple-modality dataset: image + text + clinical metadata.

    Extends MultimodalCXRDataset by additionally loading standardised
    clinical feature columns as a float32 tensor.

    Clinical features (17 total, pre-scaled with StandardScaler)::

        age, gender_M, is_deceased,
        heart_rate, respiratory_rate, spo2, systolic_bp, diastolic_bp, temperature_f,
        wbc, hemoglobin, hematocrit, creatinine, crp, alk_phos, albumin,
        has_vitals

    Args:
        df               (pd.DataFrame): DataFrame with image, text, metadata, and label.
        bbox_lookup      (dict): Lung bounding-box lookup dictionary.
        tokenizer:       Bio_ClinicalBERT tokenizer.
        img_transform:   Torchvision transform pipeline.
        clinical_features (list[str]): Ordered list of clinical column names.
        max_len          (int): Text token sequence length. Default: 256.

    Returns per item:
        tuple: ``(image_tensor, input_ids, attention_mask, metadata_tensor, label)``

    Example::

        FEATURES = ['age', 'gender_M', 'wbc', ...]   # 17 features
        ds = TripleModalCXRDataset(df, bbox_lookup, tokenizer,
                                    img_transform, FEATURES)
        img, ids, mask, meta, label = ds[0]
    """

    def __init__(
        self,
        df:               pd.DataFrame,
        bbox_lookup:      dict,
        tokenizer,
        img_transform,
        clinical_features: list,
        max_len:          int = 256,
    ):
        self.df               = df.reset_index(drop=True)
        self.bbox_lookup      = bbox_lookup
        self.tokenizer        = tokenizer
        self.transform        = img_transform
        self.clinical_features = clinical_features
        self.max_len          = max_len
        self.clahe            = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image
        img = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
        img = self.clahe.apply(img)
        bb  = self.bbox_lookup.get(row.image_path, None)
        if bb and bb.get("x_max", 0) > 0:
            img = img[bb["y_min"]:bb["y_max"], bb["x_min"]:bb["x_max"]]
        img = self.transform(Image.fromarray(img))

        # Text
        enc = self.tokenizer(
            row.report_rich,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Clinical metadata
        meta = torch.tensor(
            [float(row[c]) for c in self.clinical_features],
            dtype=torch.float32,
        )

        return (
            img,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            meta,
            int(row.label),
        )

    def set_transform(self, transform) -> None:
        """Swap image transform (used for TTA)."""
        self.transform = transform
