# PneumoFusionNet API Reference

Complete reference for all importable modules in `src/`.

> **Install as package first:**
> ```bash
> pip install -e .
> ```

---

## Quick Import Guide

```python
# Models
from src.models import ImageEncoder, TextEncoder, TripleFusionNet, CrossAttnFusionNet

# Data
from src.data import CXRDataset, MultimodalCXRDataset, TripleModalCXRDataset
from src.data import build_report_text, get_val_transforms, get_tta_transforms

# Utils
from src.utils import FocalLoss, evaluate, find_optimal_threshold, set_seed
```

---

## `src.models.vision`

### `ChannelAttention(channels, reduction=16)`
CBAM channel attention. Applies avg-pool + max-pool → shared MLP → sigmoid gate.

### `SpatialAttention(kernel_size=7)`
CBAM spatial attention. Concatenates channel-wise avg/max → conv → sigmoid gate.

### `CBAM(channels, reduction=16)`
Full Convolutional Block Attention Module (channel then spatial, sequentially).

---

### `EnhancedPneumoNetV4(num_classes=2)`
**Phase 1 standalone image classifier.**

| Layer | Detail |
|:---|:---|
| Backbone | TorchXRayVision DenseNet-121 (`densenet121-res224-all`) |
| Frozen | `conv0` → `transition2` |
| Trainable | `denseblock3`, `transition3`, `denseblock4` |
| Attention | CBAM on 1024-ch feature map |
| Head | Dropout→Linear(1024→512)→BN→ReLU→Dropout→Linear(512→128)→BN→ReLU→Linear(128→2) |

```python
model = EnhancedPneumoNetV4(num_classes=2)
logits   = model(images)           # (B, 1, 224, 224) → (B, 2)
features = model.get_features(images)  # (B, 1024)
```

---

### `ImageEncoder(checkpoint_path)`
**Frozen Phase 1 feature extractor for Phase 2 & 3.**

Loads a Phase 1 `.pth` checkpoint, freezes all parameters.

```python
encoder = ImageEncoder("mimic/main/outputs/.../best_model_fold5.pth")
features = encoder(images)              # (B, 1024)
maps     = encoder.get_feature_maps(images)  # (B, 1024, H, W) for Grad-CAM
```

### `build_model(num_classes=2, device=None)`
Factory function for `EnhancedPneumoNetV4`. Prints parameter counts.

---

## `src.models.text_encoder`

### `TextEncoder(model_name='emilyalsentzer/Bio_ClinicalBERT')`
**Bio_ClinicalBERT encoder with partial fine-tuning.**

Freezes all layers except top 2 encoder layers (10, 11) and pooler.
Returns full `last_hidden_state` for cross-attention.

```python
encoder = TextEncoder()
tokens  = encoder(input_ids, attention_mask)  # → (B, seq_len, 768)
```

### `build_tokenizer(model_name='emilyalsentzer/Bio_ClinicalBERT')`
Returns a pre-configured `AutoTokenizer`.

```python
tokenizer = build_tokenizer()
enc = tokenizer(text, max_length=256, padding='max_length',
                truncation=True, return_tensors='pt')
```

---

## `src.models.fusion`

### `CrossAttnFusionNet(img_dim=1024, txt_dim=768, attn_dim=512, heads=8)`
**Phase 2 multimodal fusion (Image + Text).**

Image features query the BERT token sequence via 8-head cross-attention.
Fused dim: `1024 + 512 = 1536`.

```python
fusion = CrossAttnFusionNet()
logits = fusion(img_feat, txt_tokens)
# img_feat  : (B, 1024)
# txt_tokens: (B, 256, 768)
# logits    : (B, 2)
```

---

### `MetadataEncoder(in_dim=17, hidden=128, out_dim=64)`
**Phase 3 clinical metadata MLP.**

Maps 17 standardised clinical features → 64-d embedding.

```python
encoder  = MetadataEncoder(in_dim=17)
meta_emb = encoder(clinical_tensor)   # (B, 17) → (B, 64)
```

---

### `TripleFusionNet(img_dim=1024, txt_dim=768, attn_dim=512, heads=8, meta_out=64)`
**Phase 3 triple fusion (Image + Text + Metadata).**

Fused dim: `1024 + 512 + 64 = 1600`.

```python
model  = TripleFusionNet()
logits = model(img_feat, txt_tokens, meta_feat)
# (B, 2)

# Warm-start cross-attention from Phase 2 checkpoint:
model.load_phase2_weights("mimic/main/outputs/.../best_v2_model.pth")
```

---

## `src.data.dataset`

### `CXRDataset(df, bbox_lookup, transform=None)`
**Phase 1** — image-only. Returns `(image_tensor, label)`.

### `MultimodalCXRDataset(df, bbox_lookup, tokenizer, img_transform, max_len=256)`
**Phase 2** — image + text. Returns `(image, input_ids, attention_mask, label)`.

### `TripleModalCXRDataset(df, bbox_lookup, tokenizer, img_transform, clinical_features, max_len=256)`
**Phase 3** — image + text + metadata. Returns `(image, input_ids, attention_mask, meta_tensor, label)`.

All datasets support `set_transform(t)` for TTA view swapping.

---

## `src.data.preprocessing`

### `apply_clahe(image, clip_limit=2.0, tile_grid=(8,8))`
CLAHE contrast enhancement on a grayscale `np.ndarray`.

### `compute_lung_bbox(image_path, margin=10) → dict | None`
Detects lung bounding box using Otsu + contour detection.
Returns `{x_min, y_min, x_max, y_max}` or `None`.

### `build_report_text(row) → str`
**Anti-leakage** text builder. Combines `indication + findings`, excludes
`IMPRESSION`, redacts diagnostic keywords with `[REDACTED]`.

### `get_train_transforms(img_size=320, mean, std) → Compose`
Training transform with rotation, colour jitter, normalize.

### `get_val_transforms(img_size=320, mean, std) → Compose`
Deterministic resize + normalize.

### `get_tta_transforms(img_size=224, mean, std) → list[Compose]`
10-view TTA transform list (rotation, zoom, brightness, blur, translate).

### `compute_dataset_stats(df, n_samples=200, bbox_lookup=None) → (mean, std)`
Estimates per-channel mean/std from a sample of CLAHE-processed images.

---

## `src.utils.metrics`

### `evaluate(model, loader, device) → (acc, auc, f1, preds, labels, probs)`
Single-pass evaluation for Phase 1 image-only models.

### `evaluate_tta(model, dataset, tta_transforms, device, batch_size=8)`
TTA evaluation — averages softmax probabilities over 10 augmented views.

### `find_optimal_threshold(labels, probs) → float`
Youden-J statistic: maximises `TPR − FPR`.

### `find_clinical_threshold(labels, probs, target_sensitivity=0.90) → float`
Finds lowest threshold achieving ≥ target sensitivity. Critical for clinical deployment.

### `compute_metrics(labels, probs, threshold=0.5, name, verbose=True) → dict`
Returns `{accuracy, sensitivity, specificity, f1, auc}`.

### `permutation_importance(fusion_model, image_encoder, text_encoder, meta_encoder, test_loader, feature_names, device, baseline_auc=None) → dict`
Phase 3 clinical feature importance via permutation (AUC drop per feature).

---

## `src.utils.training`

### `set_seed(seed=42)`
Seeds Python, NumPy, and PyTorch for reproducibility.

### `FocalLoss(alpha=None, gamma=2.0, label_smoothing=0.0, weight=None)`
Focal Loss `FL(p_t) = -(1-p_t)^γ log(p_t)` with optional class weighting.

### `mixup_data(x, y, alpha=0.3) → (mixed_x, y_a, y_b, lam)`
Phase 1 image-space Mixup.

### `mixup_embeddings(img_feat, txt_tokens, labels, alpha=0.2)`
Phase 2 embedding-space Mixup on image + text.

### `mixup_triple(img_feat, txt_tokens, meta_feat, labels, alpha=0.2)`
Phase 3 triple embedding Mixup (image + text + metadata simultaneously).

### `mixup_criterion(criterion, logits, labels_a, labels_b, lam) → Tensor`
Mixed loss: `lam * loss(a) + (1-lam) * loss(b)`.

### `train_one_epoch_p1(model, loader, criterion, optimizer, device, mixup_alpha=0.3, grad_clip=1.0)`
Complete Phase 1 training epoch with Mixup + gradient clipping.

### `build_optimizer_p1(model, lr=1e-4, lr_multiplier=10.0, weight_decay=1e-4, warmup_epochs=3, total_epochs=30) → (optimizer, scheduler)`
AdamW with differential LRs + linear warm-up + cosine annealing.
