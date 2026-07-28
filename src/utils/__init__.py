"""PneumoFusionNet — Utils sub-package.

Import only what you need:

    from src.utils.metrics import (
        evaluate, evaluate_tta,
        find_optimal_threshold, find_clinical_threshold,
        compute_metrics, permutation_importance,
    )
    from src.utils.training import (
        FocalLoss, set_seed,
        mixup_data, mixup_embeddings, mixup_triple, mixup_criterion,
        train_one_epoch_p1, build_optimizer_p1,
    )
"""

__all__ = [
    # metrics
    "compute_metrics",
    "evaluate",
    "evaluate_tta",
    "find_clinical_threshold",
    "find_optimal_threshold",
    "permutation_importance",
    # training
    "FocalLoss",
    "build_optimizer_p1",
    "mixup_criterion",
    "mixup_data",
    "mixup_embeddings",
    "mixup_triple",
    "set_seed",
    "train_one_epoch_p1",
]
