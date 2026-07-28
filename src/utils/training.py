"""
src/utils/training.py
=====================
Training utilities shared across all phases of PneumoFusionNet.

Contains:
    FocalLoss            — Focal loss with optional class weighting (all phases)
    mixup_data           — Image-space Mixup (Phase 1)
    mixup_embeddings     — Embedding-space Mixup (Phase 2)
    mixup_triple         — Triple embedding Mixup (Phase 3)
    mixup_criterion      — Mixed loss computation helper
    train_one_epoch_p1   — Phase 1 training epoch (image-only)
    set_seed             — Reproducibility seed setter
    build_optimizer_p1   — Differential-LR AdamW + cosine scheduler (Phase 1)

Source notebooks:
    mimic/main/Phase-1/Phase-1.1v4-crossval_tta_PA.ipynb
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
    mimic/main/Phase-3/Phase-3-triple_fusion_PA_Scaleup.ipynb
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across Python / NumPy / PyTorch.

    Args:
        seed (int): Random seed value. Default: 42 (used in all experiments).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    FL(p_t) = -(1 - p_t)^γ · log(p_t)

    Down-weights easy, well-classified examples and focuses training on
    hard, misclassified samples.  Used across Phase 1, 2, and 3 to improve
    clinical sensitivity for the minority pneumonia class.

    Args:
        alpha  (torch.Tensor | None): Per-class weights tensor [w0, w1].
                                      Computed from inverse class frequency.
        gamma  (float): Focusing parameter. Default: 2.0.
        label_smoothing (float): Optional label smoothing. Default: 0.0.
                                 Only used when ``alpha`` is provided (Phase 1).

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Example::

        criterion = FocalLoss(gamma=2.0, weight=class_weights)
        loss = criterion(logits, labels)
    """

    def __init__(
        self,
        alpha:          Optional[torch.Tensor] = None,
        gamma:          float = 2.0,
        label_smoothing: float = 0.0,
        weight:         Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Support both 'alpha' (Phase 1 style) and 'weight' (Phase 2/3 style)
        self.alpha          = alpha if alpha is not None else weight
        self.gamma          = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits  (torch.Tensor): Raw model outputs (B, num_classes).
            targets (torch.Tensor): Ground-truth class indices (B,).

        Returns:
            torch.Tensor: Scalar mean focal loss.
        """
        if self.alpha is not None:
            # Phase 1 path — optional label smoothing
            num_classes = logits.shape[1]
            if self.label_smoothing > 0:
                smooth = torch.zeros_like(logits).fill_(
                    self.label_smoothing / (num_classes - 1)
                )
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            else:
                smooth = F.one_hot(targets, num_classes).float()
            log_probs = F.log_softmax(logits, dim=1)
            probs     = torch.exp(log_probs)
            focal_w   = (1.0 - probs) ** self.gamma
            loss      = -focal_w * smooth * log_probs
            loss      = loss * self.alpha.to(logits.device).unsqueeze(0)
            return loss.sum(dim=1).mean()
        else:
            # Phase 2 / 3 path — no label smoothing
            ce   = F.cross_entropy(logits, targets, reduction="none")
            pt   = torch.exp(-ce)
            loss = ((1.0 - pt) ** self.gamma * ce).mean()
            return loss


# ---------------------------------------------------------------------------
# Mixup variants
# ---------------------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
) -> tuple:
    """Phase 1 image-space Mixup data augmentation.

    Randomly blends pairs of training images and their labels using a
    Beta(alpha, alpha) mixing coefficient.

    Args:
        x     (torch.Tensor): Input image batch (B, C, H, W).
        y     (torch.Tensor): Target label batch (B,).
        alpha (float): Beta distribution parameter. Default: 0.3.

    Returns:
        tuple: ``(mixed_x, y_a, y_b, lam)``
    """
    if alpha > 0:
        lam = float(
            max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
        )
    else:
        lam = 1.0
    index  = torch.randperm(x.size(0)).to(x.device)
    mixed  = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam


def mixup_embeddings(
    img_feat:   torch.Tensor,
    txt_tokens: torch.Tensor,
    labels:     torch.Tensor,
    alpha:      float = 0.2,
) -> tuple:
    """Phase 2 embedding-space Mixup on image features and text tokens.

    Args:
        img_feat   (torch.Tensor): Image feature batch (B, 1024).
        txt_tokens (torch.Tensor): Text token batch (B, seq_len, 768).
        labels     (torch.Tensor): Target labels (B,).
        alpha      (float): Beta distribution parameter. Default: 0.2.

    Returns:
        tuple: ``(img_mix, txt_mix, labels_a, labels_b, lam)``
    """
    if alpha <= 0:
        return img_feat, txt_tokens, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(img_feat.size(0), device=img_feat.device)
    img_mix = lam * img_feat   + (1 - lam) * img_feat[idx]
    txt_mix = lam * txt_tokens + (1 - lam) * txt_tokens[idx]
    return img_mix, txt_mix, labels, labels[idx], lam


def mixup_triple(
    img_feat:   torch.Tensor,
    txt_tokens: torch.Tensor,
    meta_feat:  torch.Tensor,
    labels:     torch.Tensor,
    alpha:      float = 0.2,
) -> tuple:
    """Phase 3 triple embedding-space Mixup (image + text + metadata).

    Args:
        img_feat   (torch.Tensor): Image features (B, 1024).
        txt_tokens (torch.Tensor): Text tokens (B, seq_len, 768).
        meta_feat  (torch.Tensor): Metadata embeddings (B, 64).
        labels     (torch.Tensor): Target labels (B,).
        alpha      (float): Beta distribution parameter. Default: 0.2.

    Returns:
        tuple: ``(img_mix, txt_mix, meta_mix, labels_a, labels_b, lam)``
    """
    if alpha <= 0:
        return img_feat, txt_tokens, meta_feat, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(img_feat.size(0), device=img_feat.device)
    img_mix  = lam * img_feat   + (1 - lam) * img_feat[idx]
    txt_mix  = lam * txt_tokens + (1 - lam) * txt_tokens[idx]
    meta_mix = lam * meta_feat  + (1 - lam) * meta_feat[idx]
    return img_mix, txt_mix, meta_mix, labels, labels[idx], lam


def mixup_criterion(
    criterion: nn.Module,
    logits:    torch.Tensor,
    labels_a:  torch.Tensor,
    labels_b:  torch.Tensor,
    lam:       float,
) -> torch.Tensor:
    """Compute mixed Mixup loss as weighted sum of two label losses.

    Args:
        criterion (nn.Module): Loss function (e.g. FocalLoss).
        logits    (torch.Tensor): Model output logits (B, num_classes).
        labels_a  (torch.Tensor): Original labels (B,).
        labels_b  (torch.Tensor): Shuffled partner labels (B,).
        lam       (float): Mixing coefficient.

    Returns:
        torch.Tensor: Scalar mixed loss.
    """
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ---------------------------------------------------------------------------
# Phase 1 training helpers
# ---------------------------------------------------------------------------

def train_one_epoch_p1(
    model:        nn.Module,
    loader,
    criterion:    nn.Module,
    optimizer:    torch.optim.Optimizer,
    device:       torch.device,
    mixup_alpha:  float = 0.3,
    grad_clip:    float = 1.0,
) -> tuple:
    """Run one Phase 1 training epoch with Mixup and gradient clipping.

    Args:
        model       (nn.Module): EnhancedPneumoNetV4.
        loader:     Training DataLoader.
        criterion   (nn.Module): FocalLoss instance.
        optimizer:  AdamW optimizer.
        device      (torch.device): Computing device.
        mixup_alpha (float): Mixup Beta parameter. Default: 0.3.
        grad_clip   (float): Gradient norm clip value. Default: 1.0.

    Returns:
        tuple: ``(avg_loss, avg_accuracy)``
    """
    model.train()
    total_loss = total_correct = total_n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        mixed, ya, yb, lam = mixup_data(imgs, labels, mixup_alpha)

        optimizer.zero_grad()
        out  = model(mixed)
        loss = mixup_criterion(criterion, out, ya, yb, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss    += loss.item() * len(labels)
        preds          = out.argmax(1)
        total_correct += (
            lam * (preds == ya).float().sum().item()
            + (1 - lam) * (preds == yb).float().sum().item()
        )
        total_n += len(labels)

    return total_loss / total_n, total_correct / total_n


def build_optimizer_p1(
    model:         nn.Module,
    lr:            float = 1e-4,
    lr_multiplier: float = 10.0,
    weight_decay:  float = 1e-4,
    warmup_epochs: int   = 3,
    total_epochs:  int   = 30,
) -> tuple:
    """Build AdamW optimizer and warm-up + cosine scheduler for Phase 1.

    Uses differential learning rates:
    - Pre-trained DenseNet backbone layers: ``lr``
    - New CBAM + classifier head: ``lr × lr_multiplier``

    Args:
        model         (nn.Module): EnhancedPneumoNetV4.
        lr            (float): Base learning rate. Default: 1e-4.
        lr_multiplier (float): LR multiplier for new layers. Default: 10.
        weight_decay  (float): AdamW weight decay. Default: 1e-4.
        warmup_epochs (int): Number of linear warm-up epochs. Default: 3.
        total_epochs  (int): Total training epochs. Default: 30.

    Returns:
        tuple: ``(optimizer, scheduler)``
    """
    pretrained_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "classifier" not in n and "cbam" not in n
    ]
    new_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ("classifier" in n or "cbam" in n)
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": pretrained_params, "lr": lr},
            {"params": new_params,        "lr": lr * lr_multiplier},
        ],
        weight_decay=weight_decay,
    )

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    return optimizer, scheduler
