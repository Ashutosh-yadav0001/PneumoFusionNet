# Contributing to PneumoFusionNet

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## ⚠️ Data Access Requirement

> **IMPORTANT:** This project uses the **MIMIC-CXR** and **MIMIC-IV** datasets, which are credentialed-access clinical databases. You **must not** commit, share, or publish any patient data in any form. Any PR containing patient data will be rejected immediately.

To obtain data access:
1. Complete [CITI Human Subjects Research training](https://www.citiprogram.org/)
2. Sign the [PhysioNet Data Use Agreement](https://physionet.org/settings/credentialing/)
3. Request access at [physionet.org/content/mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

---

## 🚀 Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/Ashutosh-yadav0001/PneumoFusionNet.git
cd PneumoFusionNet
```

### 2. Create Environment
```bash
python -m venv venv_PneumoFusionNet
# Windows:
venv_PneumoFusionNet\Scripts\activate
# Linux / macOS:
source venv_PneumoFusionNet/bin/activate
```

### 3. Install PyTorch (CUDA)
```bash
# CUDA 11.3+
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

### 4. Install Project in Editable Mode
```bash
pip install -e ".[dev]"
```

### 5. Register Jupyter Kernel
```bash
python -m ipykernel install --user \
    --name=venv_PneumoFusionNet \
    --display-name "PneumoFusionNet"
```

---

## 📁 Where Code Lives

All primary experiment code lives in `mimic/main/`:

```
mimic/main/
├── Phase-1/     ← DenseNet-121 + CBAM image classifier notebooks
├── Phase-2/     ← Bio_ClinicalBERT + CrossAttention fusion notebooks
├── Phase-3/     ← Triple fusion (image + text + clinical metadata)
├── Scaleup/     ← Scale-up experiments on ~3,763 images
├── dataset/     ← CSV manifests and dataset build scripts
└── outputs/     ← Checkpoints and result plots (gitignored for .pth)
```

Reusable Python modules extracted from notebooks live in `src/`:

```
src/
├── models/
│   ├── vision.py        — DenseNet-121, CBAM, ImageEncoder
│   ├── text_encoder.py  — Bio_ClinicalBERT wrapper
│   └── fusion.py        — CrossAttnFusionNet, TripleFusionNet
├── data/
│   ├── dataset.py       — CXRDataset, MultimodalCXRDataset, TripleModalCXRDataset
│   └── preprocessing.py — CLAHE, anti-leakage text, TTA transforms
└── utils/
    ├── metrics.py       — evaluate, TTA eval, threshold selection
    └── training.py      — FocalLoss, Mixup, training loop helpers
```

---

## 🔄 Contribution Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** — follow the code style guide below.

3. **Verify imports work** (no notebook dependency):
   ```bash
   python -c "from src.models.fusion import TripleFusionNet; print('OK')"
   python -c "from src.data.dataset import CXRDataset; print('OK')"
   ```

4. **Run linting**:
   ```bash
   black src/ --check
   isort src/ --check
   flake8 src/ --max-line-length=100
   ```

5. **Open a Pull Request** — fill in the PR template with:
   - What changed and why
   - Which notebook the code was extracted from (if applicable)
   - Experiment results if you ran evaluations

---

## 🎨 Code Style

- **Formatter**: [Black](https://black.readthedocs.io/) with `line-length=100`
- **Imports**: [isort](https://pycbpep8.readthedocs.io/en/latest/) with Black profile
- **Docstrings**: Google-style (Args / Returns / Example blocks)
- **Type hints**: Use them in all function signatures
- **No bare `except:`** — catch specific exceptions

---

## 📝 Types of Contributions

| Type | Description |
|:---|:---|
| 🐛 **Bug fix** | Fix incorrect behaviour in `src/` modules |
| 📚 **Documentation** | Improve docstrings, READMEs, or `docs/` files |
| 🔬 **New experiment** | Add a new notebook under `mimic/main/` |
| ⚙️ **Refactor** | Extract reusable code from notebooks into `src/` |
| 📊 **Results** | Update experiment tables with new benchmark results |

---

## ❓ Questions

Open a [GitHub Issue](https://github.com/Ashutosh-yadav0001/PneumoFusionNet/issues) or email:
- **Ashutosh Yadav** — [ashutosh@op.iitg.ac.in](mailto:ashutosh@op.iitg.ac.in)

---

*Please read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.*
