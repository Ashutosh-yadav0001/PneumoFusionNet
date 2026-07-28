"""PneumoFusionNet — Data sub-package.

Import only what you need:

    from src.data.dataset import CXRDataset, MultimodalCXRDataset, TripleModalCXRDataset
    from src.data.preprocessing import (
        apply_clahe, compute_lung_bbox, build_report_text,
        get_train_transforms, get_val_transforms, get_tta_transforms,
        compute_dataset_stats,
    )
"""

__all__ = [
    # datasets
    "CXRDataset",
    "MultimodalCXRDataset",
    "TripleModalCXRDataset",
    # preprocessing
    "apply_clahe",
    "build_report_text",
    "compute_dataset_stats",
    "compute_lung_bbox",
    "get_tta_transforms",
    "get_train_transforms",
    "get_val_transforms",
]
