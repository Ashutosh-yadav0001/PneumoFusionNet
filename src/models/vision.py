"""
src/models/vision.py
====================
Vision backbone module for PneumoFusionNet.

Contains:
    ChannelAttention   — CBAM channel attention branch
    SpatialAttention   — CBAM spatial attention branch
    CBAM               — Full Convolutional Block Attention Module
    EnhancedPneumoNetV4 — Phase 1 standalone DenseNet-121 classifier
    ImageEncoder        — Frozen feature extractor used in Phase 2 & 3
    build_model         — Convenience factory for Phase 1 training

Source notebooks:
    mimic/main/Phase-1/Phase-1.1v4-crossval_tta_PA.ipynb
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
    mimic/main/Phase-3/Phase-3-triple_fusion_PA_Scaleup.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# Attention Modules (CBAM)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """CBAM channel attention branch.

    Applies both average-pool and max-pool on spatial dimensions, passes
    them through a shared bottleneck MLP, and rescales input features with
    a sigmoid gate.

    Args:
        channels (int): Number of input feature-map channels.
        reduction (int): Bottleneck reduction ratio. Default: 16.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(
            self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    """CBAM spatial attention branch.

    Concatenates channel-wise average and max along dim=1, applies a
    single-channel convolution, and rescales input features with sigmoid.

    Args:
        kernel_size (int): Convolution kernel size. Default: 7.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies channel attention followed by spatial attention to
    refine intermediate CNN feature maps.

    Args:
        channels (int): Input channel dimension (1024 for DenseNet-121).
        reduction (int): Channel reduction ratio. Default: 16.

    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_att(self.channel_att(x))


# ---------------------------------------------------------------------------
# Phase 1 — Standalone Image Classifier
# ---------------------------------------------------------------------------

class EnhancedPneumoNetV4(nn.Module):
    """Phase 1 DenseNet-121 pneumonia classifier with CBAM attention.

    Architecture:
        - Backbone : TorchXRayVision DenseNet-121 (densenet121-res224-all)
        - Frozen   : conv0 → transition2 (early layers)
        - Trainable: denseblock3, transition3, denseblock4 + CBAM + head
        - Attention: CBAM on 1024-channel feature map
        - Head     : Dropout → Linear(1024→512) → BN → ReLU →
                     Dropout → Linear(512→128) → BN → ReLU →
                     Dropout → Linear(128→num_classes)

    Args:
        num_classes (int): Number of output classes. Default: 2.

    Example::

        model = EnhancedPneumoNetV4(num_classes=2)
        logits = model(images)          # images: (B, 1, 224, 224)
        features = model.get_features(images)  # (B, 1024)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.features = xrv_model.features

        # Freeze early layers; fine-tune later blocks
        freeze_layers = {
            "conv0", "norm0", "relu0", "pool0",
            "denseblock1", "transition1",
            "denseblock2", "transition2",
        }
        for name, child in self.features.named_children():
            for p in child.parameters():
                p.requires_grad = name not in freeze_layers

        self.cbam = CBAM(1024, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits.

        Args:
            x: Grayscale CXR tensor of shape (B, 1, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        f = F.relu(self.features(x), inplace=True)
        f = self.cbam(f)
        f = self.pool(f).flatten(1)
        return self.classifier(f)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1024-d pooled feature vector (no classification head).

        Args:
            x: Grayscale CXR tensor of shape (B, 1, H, W).

        Returns:
            Feature tensor of shape (B, 1024).
        """
        f = F.relu(self.features(x), inplace=True)
        f = self.cbam(f)
        return self.pool(f).flatten(1)


def build_model(num_classes: int = 2, device: torch.device = None) -> EnhancedPneumoNetV4:
    """Factory function: instantiate and move EnhancedPneumoNetV4 to device.

    Args:
        num_classes (int): Output classes. Default: 2.
        device (torch.device): Target device. Defaults to CUDA if available.

    Returns:
        EnhancedPneumoNetV4 on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedPneumoNetV4(num_classes=num_classes).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[EnhancedPneumoNetV4] Params: {total:,} total, "
        f"{trainable:,} trainable ({trainable / total * 100:.1f}%)"
    )
    return model


# ---------------------------------------------------------------------------
# Phase 2 / 3 — Frozen Image Feature Extractor
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """Frozen DenseNet-121 + CBAM feature extractor for Phase 2 & 3.

    Loads Phase 1 trained weights, freezes all parameters, and exposes a
    1024-dimensional pooled feature vector for downstream multimodal fusion.
    Also provides ``get_feature_maps()`` for Grad-CAM visualisation.

    Args:
        checkpoint_path (str): Path to Phase 1 ``.pth`` checkpoint file.

    Example::

        encoder = ImageEncoder("mimic/main/outputs/.../best_model_fold5.pth")
        features = encoder(images)        # (B, 1024)
        maps = encoder.get_feature_maps(images)  # (B, 1024, H, W)
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.features = xrv_model.features
        self.cbam = CBAM(1024)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Load only matching keys from Phase 1 checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        compatible = {
            k: v for k, v in state_dict.items()
            if k.startswith("features.") or k.startswith("cbam.")
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        print(
            f"[ImageEncoder] Loaded {len(compatible)} keys from checkpoint | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

        # Freeze all parameters — image encoder is not updated during Phase 2/3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled 1024-d feature vector.

        Args:
            x: Grayscale CXR tensor (B, 1, H, W).

        Returns:
            Tensor of shape (B, 1024).
        """
        f = F.relu(self.features(x), inplace=True)
        f = self.cbam(f)
        return self.pool(f).flatten(1)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps before pooling (for Grad-CAM).

        Args:
            x: Grayscale CXR tensor (B, 1, H, W).

        Returns:
            Tensor of shape (B, 1024, H', W').
        """
        return F.relu(self.cbam(self.features(x)), inplace=True)
