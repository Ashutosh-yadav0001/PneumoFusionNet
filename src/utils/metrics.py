"""
src/utils/metrics.py
====================
Evaluation and threshold-selection utilities for PneumoFusionNet.

Contains:
    evaluate             — Single-pass evaluation for Phase 1 image-only models
    evaluate_tta         — Test-Time Augmentation evaluation (10-view average)
    find_optimal_threshold  — Youden-J optimal decision threshold
    find_clinical_threshold — Sensitivity-targeted clinical threshold
    compute_metrics      — Confusion matrix, sensitivity, specificity, F1
    permutation_importance  — Phase 3 clinical feature importance ranking

Source notebooks:
    mimic/main/Phase-1/Phase-1.1v4-crossval_tta_PA.ipynb
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
    mimic/main/Phase-3/Phase-3-triple_fusion_PA_Scaleup.ipynb
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Phase 1 — Image-only evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Evaluate a Phase 1 image-only model on a DataLoader.

    Args:
        model  (nn.Module): Trained image classifier.
        loader (DataLoader): Validation or test DataLoader.
        device (torch.device): CPU or CUDA device.

    Returns:
        tuple: ``(accuracy, auc, f1_macro, predictions, labels, probs)``
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            probs = F.softmax(out, dim=1)[:, 1]
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    try:
        auc_val = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_val = 0.0

    return acc, auc_val, f1, all_preds, all_labels, all_probs


def evaluate_tta(
    model: torch.nn.Module,
    dataset,
    tta_transforms: list,
    device: torch.device,
    batch_size: int = 8,
) -> tuple:
    """Test-Time Augmentation evaluation — averages probabilities over N views.

    Swaps the dataset transform for each TTA view, runs inference, and
    averages the softmax probabilities for class 1 across all views.

    Args:
        model          (nn.Module): Trained image classifier.
        dataset:       Dataset instance supporting ``set_transform()``.
        tta_transforms (list): List of torchvision Compose transforms.
        device         (torch.device): Computing device.
        batch_size     (int): DataLoader batch size. Default: 8.

    Returns:
        tuple: ``(accuracy, auc, f1_macro, predictions, labels, avg_probs)``
    """
    model.eval()
    n_samples = len(dataset)
    n_views   = len(tta_transforms)
    all_probs  = np.zeros((n_samples, n_views))
    all_labels = []

    for v, tfm in enumerate(tta_transforms):
        dataset.set_transform(tfm)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        probs_v, labels_v = [], []

        with torch.no_grad():
            for imgs, labels in loader:
                out = model(imgs.to(device))
                p   = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs_v.extend(p)
                labels_v.extend(labels.tolist())

        all_probs[:, v] = probs_v
        if v == 0:
            all_labels = labels_v

    avg_probs = all_probs.mean(axis=1)
    avg_preds = (avg_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, avg_preds)
    f1  = f1_score(all_labels, avg_preds, average="macro")
    try:
        auc_val = roc_auc_score(all_labels, avg_probs)
    except ValueError:
        auc_val = 0.0

    return acc, auc_val, f1, avg_preds.tolist(), all_labels, avg_probs.tolist()


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(labels: list, probs: list) -> float:
    """Find the optimal decision threshold using Youden's J statistic.

    Youden's J = TPR − FPR (maximised at the best balance between
    sensitivity and specificity).

    Args:
        labels (list): Ground-truth binary labels.
        probs  (list): Predicted probabilities for the positive class.

    Returns:
        float: Optimal classification threshold.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def find_clinical_threshold(
    labels: list,
    probs: list,
    target_sensitivity: float = 0.90,
) -> float:
    """Find the lowest threshold that achieves ≥ target sensitivity.

    In clinical pneumonia detection, minimising false negatives is critical.
    This function scans the ROC curve and returns the first threshold
    achieving the requested sensitivity.  Falls back to Youden-J if the
    target cannot be met.

    Args:
        labels             (list): Ground-truth binary labels.
        probs              (list): Predicted positive-class probabilities.
        target_sensitivity (float): Minimum required sensitivity. Default: 0.90.

    Returns:
        float: Clinical decision threshold.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    for t, s in zip(thresholds, tpr):
        if s >= target_sensitivity:
            return float(t)
    # Fallback to Youden-J
    return float(thresholds[np.argmax(tpr - fpr)])


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: list,
    probs: list,
    threshold: float = 0.5,
    name: str = "Evaluation",
    verbose: bool = True,
) -> dict:
    """Compute full classification metrics at a given threshold.

    Args:
        labels    (list): Ground-truth binary labels.
        probs     (list): Predicted positive-class probabilities.
        threshold (float): Decision threshold. Default: 0.5.
        name      (str): Display name for printed results.
        verbose   (bool): Whether to print results. Default: True.

    Returns:
        dict: Keys ``accuracy``, ``sensitivity``, ``specificity``,
              ``f1``, ``auc``.
    """
    preds = [1 if p >= threshold else 0 for p in probs]
    cm    = confusion_matrix(labels, preds)

    sensitivity  = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
    specificity  = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
    accuracy     = (cm[0, 0] + cm[1, 1]) / cm.sum()
    f1           = f1_score(labels, preds)
    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = 0.0

    if verbose:
        print(
            f"{name} (thresh={threshold:.3f}): "
            f"AUC={auc_val:.4f}  Acc={accuracy:.3f}  "
            f"Sens={sensitivity:.3f}  Spec={specificity:.3f}  F1={f1:.3f}"
        )

    return {
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          f1,
        "auc":         auc_val,
    }


# ---------------------------------------------------------------------------
# Phase 3 — Permutation Feature Importance
# ---------------------------------------------------------------------------

def permutation_importance(
    fusion_model: torch.nn.Module,
    image_encoder: torch.nn.Module,
    text_encoder: torch.nn.Module,
    meta_encoder: torch.nn.Module,
    test_loader: DataLoader,
    feature_names: list,
    device: torch.device,
    baseline_auc: Optional[float] = None,
) -> dict:
    """Measure permutation importance of each clinical metadata feature.

    For each feature, shuffles its values across the batch, evaluates the
    resulting AUC drop, and records ``baseline_auc - permuted_auc`` as the
    importance score.  Higher score = more important feature.

    Args:
        fusion_model   (nn.Module): Phase 3 TripleFusionNet.
        image_encoder  (nn.Module): Frozen ImageEncoder.
        text_encoder   (nn.Module): TextEncoder.
        meta_encoder   (nn.Module): MetadataEncoder.
        test_loader    (DataLoader): Test DataLoader.
        feature_names  (list): Ordered list of clinical feature column names.
        device         (torch.device): Computing device.
        baseline_auc   (float | None): Pre-computed baseline AUC. If None,
                                       it is computed first.

    Returns:
        dict: Mapping of feature name → AUC importance score (sorted desc).
    """
    for m in [fusion_model, image_encoder, text_encoder, meta_encoder]:
        m.eval()

    # Compute baseline if not provided
    if baseline_auc is None:
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, ids, masks, meta, labels in test_loader:
                img_f  = image_encoder(imgs.to(device))
                txt_t  = text_encoder(ids.to(device), masks.to(device))
                meta_f = meta_encoder(meta.to(device))
                logits = fusion_model(img_f, txt_t, meta_f)
                probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.tolist())
        baseline_auc = roc_auc_score(all_labels, all_probs)
        print(f"[permutation_importance] Baseline AUC: {baseline_auc:.4f}")

    importances = {}
    for feat_idx, feat_name in enumerate(feature_names):
        all_probs_perm, all_labels_perm = [], []

        with torch.no_grad():
            for imgs, ids, masks, meta, labels in test_loader:
                imgs   = imgs.to(device)
                ids    = ids.to(device)
                masks  = masks.to(device)
                meta   = meta.clone().to(device)
                labels = labels.to(device)

                # Permute this feature across the batch
                perm_idx = torch.randperm(meta.size(0))
                meta[:, feat_idx] = meta[perm_idx, feat_idx]

                img_f  = image_encoder(imgs)
                txt_t  = text_encoder(ids, masks)
                meta_f = meta_encoder(meta)
                logits = fusion_model(img_f, txt_t, meta_f)
                probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs_perm.extend(probs)
                all_labels_perm.extend(labels.cpu().tolist())

        perm_auc = roc_auc_score(all_labels_perm, all_probs_perm)
        importances[feat_name] = baseline_auc - perm_auc
        print(f"  {feat_name:20s}: {importances[feat_name]:+.4f}")

    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
