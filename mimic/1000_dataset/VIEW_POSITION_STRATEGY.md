# 📐 View Position Strategy Analysis for Pneumonia Detection

## Your Current Strategy (Pilot 139 Dataset)

Looking at your notebook code ([Phase-1.1mimic_image_classifier(balanced-Set).ipynb](Notebooks/Phase-1.1mimic_image_classifier(balanced-Set).ipynb)), your current approach is:

1. **One image per study** — Your `mimic_dataset.csv` has 1 row per `(subject_id, study_id)`, not per image. So you already pick a single image per study.
2. **No view filtering** — You pick images without filtering by `ViewPosition`. The model receives a random mix of AP, PA, LATERAL, and LL views.
3. **Grayscale + resize to 224×224** — All views are treated identically.

### What Views Are Actually In Your Pilot Dataset?

| View | Count | % |
|:---|:---:|:---:|
| AP | 55 | 39.6% |
| LATERAL | 40 | 28.8% |
| PA | 31 | 22.3% |
| LL | 8 | 5.8% |
| Unknown | 5 | 3.6% |

> ⚠️ **Critical issue**: Your Pneumonia class has only **1 PA view** vs **12 AP** and **5 LATERAL**. The model may be learning to associate "frontal view" → Normal and "lateral view" → Pneumonia, rather than learning actual pathological features!

### View Distribution by Class (Pilot 139)

| Label | AP | PA | LATERAL | LL |
|:---|:---:|:---:|:---:|:---:|
| **Normal** | 43 | 30 | 35 | 5 |
| **Pneumonia** | 12 | 1 | 5 | 3 |

---

## The Core Problem: Mixing Views

### Why Different Views Look Different

```
PA (Postero-Anterior)         AP (Antero-Posterior)         LATERAL
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│   Standard upright   │       │ Portable/bedside    │       │   Side projection    │
│   Heart size normal  │       │ Heart appears larger│       │   Completely different│
│   Crisp lung fields  │       │ More hazy/noisy     │       │   anatomy visible    │
│   Gold standard      │       │ Sicker patients     │       │   No frontal anatomy │
└─────────────────────┘       └─────────────────────┘       └─────────────────────┘
```

Three distinct problems arise from mixing views:

1. **LATERAL views show completely different anatomy** — the heart, spine, and retrosternal space are visible, not the bilateral lung fields. A CNN trained on frontal views will be confused by lateral anatomy.

2. **AP vs PA have subtle but systematic differences** — AP images have magnified heart silhouettes and are often taken on sicker (bedside) patients, introducing a **confounding bias**: AP → sicker → more likely pneumonia.

3. **Class-view correlation becomes a shortcut** — If pneumonia patients are disproportionately imaged with AP (because they're sicker, portable bedside X-rays), the model learns "AP appearance → Pneumonia" instead of actual lung opacities.

---

## Recommendation

### ✅ Best Strategy: **Filter to PA + AP only (frontal views)**

| Approach | Pros | Cons |
|:---|:---|:---|
| **All views (current)** | More data | View confusion, lateral noise, biased shortcuts |
| **PA only** | Most consistent, gold standard | Loses ~60-70% of data |
| **PA + AP only** ⭐ | Consistent frontal anatomy, large dataset | Minor AP/PA differences (manageable) |
| **Separate models per view** | Clean per-view learning | Complex, needs enough data per view |

### Why PA + AP Together Works

- Both are **frontal projections** — the CNN sees bilateral lung fields, heart silhouette, and mediastinum in both.
- The AP/PA differences (heart magnification, image quality) are **minor compared to PA vs LATERAL**.
- ResNet/CNNs can learn to be robust to these minor differences with data augmentation.
- You preserve the majority of your data (~60-65% of all images).

### Why You Must Remove LATERAL/LL

- Lateral views show **fundamentally different anatomy** (spine, retrosternal space).
- A single CNN cannot meaningfully learn pneumonia features from both frontal and lateral projections simultaneously.
- In research literature, nearly all pneumonia detection papers use **frontal views only**.

---

## Impact on the 1,000 Patient Dataset

### Current View Distribution (All 1,721 images)

| View | Total | Normal | Pneumonia |
|:---|:---:|:---:|:---:|
| AP | 564 | 215 | 349 |
| PA | 489 | 312 | 177 |
| LATERAL | 436 | 309 | 127 |
| LL | 145 | 58 | 87 |
| Unknown | 87 | — | — |

### After Filtering to PA + AP Only (~1,053 images)

| View | Total | Normal | Pneumonia |
|:---|:---:|:---:|:---:|
| AP | 564 | 215 | 349 |
| PA | 489 | 312 | 177 |
| **Total** | **1,053** | **527** | **526** |

> 💡 After filtering, you get a **near-perfectly balanced** dataset of ~1,053 frontal images with 527 Normal and 526 Pneumonia!

---

## Code Changes Required

Your existing code **will work as-is** with a filtered dataset — no architectural changes needed. You only need to **filter the CSV before training**.

### Quick Filter (Add to your notebook before the train/val/test split)

```python
# Filter to frontal views only (PA + AP)
df = df[df['view_position'].isin(['PA', 'AP'])].reset_index(drop=True)
print(f'After frontal-view filter: {len(df)} images')
print(df['view_position'].value_counts())
```

### Optional: Add View Position as a Feature

If you want the model to be **aware** of the projection type (useful for Phase 2 multimodal fusion), you can encode it:

```python
# One-hot encode view position
df['is_AP'] = (df['view_position'] == 'AP').astype(int)
df['is_PA'] = (df['view_position'] == 'PA').astype(int)
```

---

## Summary

| Question | Answer |
|:---|:---|
| Should I use all views? | **No** — lateral views add noise and confuse the CNN |
| Should I use PA only? | Ideal but loses too much data |
| **Best approach?** | **PA + AP only** — consistent frontal anatomy, keeps ~60% of data |
| Does my code need changes? | **Only a 1-line CSV filter** before the split — no model changes |
| Is the balanced dataset still balanced? | **Yes** — PA+AP filtering gives ~527 Normal, ~526 Pneumonia |

---

## References

- Most pneumonia detection benchmarks (CheXpert, CheXNet, NIH ChestX-ray14) use **frontal views only**.
- The MIMIC-CXR official documentation recommends filtering by `ViewPosition` for task-specific studies.
- AP projection bias towards sicker patients is well-documented in radiology literature.
