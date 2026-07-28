"""
src/models/fusion.py
====================
Multimodal fusion modules for PneumoFusionNet.

Contains:
    CrossAttnFusionNet  — Phase 2: Image + Text via cross-attention (1536-d)
    MetadataEncoder     — Phase 3: Clinical metadata MLP (16/17 → 64-d)
    TripleFusionNet     — Phase 3: Image + Text + Metadata (1600-d)

Architecture summary:
    Phase 2  →  img(1024) + cross_attn(512) = 1536-d → MLP → 2 logits
    Phase 3  →  img(1024) + cross_attn(512) + meta(64) = 1600-d → MLP → 2 logits

Source notebooks:
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
    mimic/main/Phase-3/Phase-3-triple_fusion_PA_Scaleup.ipynb
"""

import torch
import torch.nn as nn

# Default hyper-parameters (match notebook CONFIG cells)
IMG_FEAT_DIM = 1024   # DenseNet-121 + CBAM output
TXT_FEAT_DIM = 768    # Bio_ClinicalBERT last_hidden_state
ATTN_DIM     = 512    # Cross-attention projection dimension
ATTN_HEADS   = 8      # Number of attention heads
META_HIDDEN  = 128    # MetadataEncoder hidden size
META_OUT_DIM = 64     # MetadataEncoder output size


# ---------------------------------------------------------------------------
# Phase 2 — Image + Text Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttnFusionNet(nn.Module):
    """Phase 2 multimodal fusion: image features attend over text tokens.

    The image global feature vector (1024-d) is used as a single-token
    Query; the full BERT token sequence (seq_len × 768) provides Keys and
    Values.  The attended text representation (512-d) is concatenated with
    the image feature and passed through a classification MLP.

    Fusion dimension: 1024 + 512 = 1536-d.

    Args:
        img_dim  (int): Image feature dimension. Default: 1024.
        txt_dim  (int): Text token hidden dimension. Default: 768.
        attn_dim (int): Cross-attention projection size. Default: 512.
        heads    (int): Number of attention heads. Default: 8.

    Example::

        fusion = CrossAttnFusionNet()
        logits = fusion(img_feat, txt_tokens)
        # img_feat  : (B, 1024)
        # txt_tokens: (B, seq_len, 768)
        # logits    : (B, 2)
    """

    def __init__(
        self,
        img_dim:  int = IMG_FEAT_DIM,
        txt_dim:  int = TXT_FEAT_DIM,
        attn_dim: int = ATTN_DIM,
        heads:    int = ATTN_HEADS,
    ):
        super().__init__()
        self.query_proj = nn.Linear(img_dim, attn_dim)
        self.key_proj   = nn.Linear(txt_dim, attn_dim)
        self.val_proj   = nn.Linear(txt_dim, attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            attn_dim, heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(attn_dim)

        fused_dim = attn_dim + img_dim  # 1536
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        img_feat:   torch.Tensor,
        txt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            img_feat   (torch.Tensor): Pooled image features (B, img_dim).
            txt_tokens (torch.Tensor): BERT token sequence (B, seq_len, txt_dim).

        Returns:
            torch.Tensor: Class logits of shape (B, 2).
        """
        q = self.query_proj(img_feat).unsqueeze(1)   # (B, 1, attn_dim)
        k = self.key_proj(txt_tokens)                 # (B, seq, attn_dim)
        v = self.val_proj(txt_tokens)                 # (B, seq, attn_dim)
        attended, _ = self.cross_attn(q, k, v)        # (B, 1, attn_dim)
        attended = self.norm1(attended.squeeze(1))     # (B, attn_dim)
        fused = torch.cat([img_feat, attended], dim=1) # (B, 1536)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Phase 3 — Clinical Metadata Encoder
# ---------------------------------------------------------------------------

class MetadataEncoder(nn.Module):
    """Phase 3 clinical metadata MLP encoder.

    Maps a vector of 16 or 17 standardised clinical features (demographics,
    vitals, laboratory values) to a compact 64-dimensional embedding.

    Architecture: Linear(in→128) → ReLU → Dropout(0.3)
                  → Linear(128→128) → ReLU → Dropout(0.2)
                  → Linear(128→64) → ReLU

    Args:
        in_dim  (int): Number of clinical input features. Default: 17.
        hidden  (int): Hidden layer size. Default: 128.
        out_dim (int): Output embedding size. Default: 64.

    Clinical features (17 total)::

        Demographics : age, gender_M, is_deceased
        Vitals       : heart_rate, respiratory_rate, spo2,
                       systolic_bp, diastolic_bp, temperature_f
        Labs         : wbc, hemoglobin, hematocrit, creatinine,
                       crp, alk_phos, albumin
        Flag         : has_vitals

    Example::

        encoder = MetadataEncoder(in_dim=17)
        meta_emb = encoder(clinical_tensor)  # (B, 64)
    """

    def __init__(
        self,
        in_dim:  int = 17,
        hidden:  int = META_HIDDEN,
        out_dim: int = META_OUT_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )
        trainable = sum(p.numel() for p in self.net.parameters())
        print(f"[MetadataEncoder] Trainable params: {trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode clinical metadata vector.

        Args:
            x (torch.Tensor): Standardised clinical feature tensor (B, in_dim).

        Returns:
            torch.Tensor: Metadata embedding of shape (B, out_dim=64).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Phase 3 — Triple Fusion Classifier
# ---------------------------------------------------------------------------

class TripleFusionNet(nn.Module):
    """Phase 3 triple-modality fusion network.

    Fuses three complementary clinical modalities:
        1. Image features  (B, 1024)  from ImageEncoder (DenseNet-121 + CBAM)
        2. Text tokens     (B, seq, 768) from TextEncoder (Bio_ClinicalBERT)
        3. Metadata        (B, 64)    from MetadataEncoder

    Cross-attention: image queries text → attended representation (B, 512).
    Concatenation  : [img(1024) | attn(512) | meta(64)] = 1600-d.
    Classifier     : LayerNorm → Dropout → Linear chain → 2 logits.

    Args:
        img_dim  (int): Image feature dimension. Default: 1024.
        txt_dim  (int): Text token hidden size.  Default: 768.
        attn_dim (int): Cross-attention projection. Default: 512.
        heads    (int): Attention heads. Default: 8.
        meta_out (int): Metadata embedding size. Default: 64.

    Note:
        Cross-attention weights can be warm-started from a Phase 2 checkpoint
        (``query_proj``, ``key_proj``, ``val_proj``, ``cross_attn``, ``norm1``
        keys are compatible).  The classifier head differs (1536→1600 input)
        and must be re-initialised.

    Example::

        model = TripleFusionNet()
        logits = model(img_feat, txt_tokens, meta_feat)
        # (B, 2)
    """

    def __init__(
        self,
        img_dim:  int = IMG_FEAT_DIM,
        txt_dim:  int = TXT_FEAT_DIM,
        attn_dim: int = ATTN_DIM,
        heads:    int = ATTN_HEADS,
        meta_out: int = META_OUT_DIM,
    ):
        super().__init__()
        # Cross-attention components (compatible with Phase 2 weights)
        self.query_proj = nn.Linear(img_dim, attn_dim)
        self.key_proj   = nn.Linear(txt_dim, attn_dim)
        self.val_proj   = nn.Linear(txt_dim, attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            attn_dim, heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(attn_dim)

        # Triple-fusion classification head
        fused_dim = img_dim + attn_dim + meta_out  # 1024 + 512 + 64 = 1600
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.4),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        print(
            f"[TripleFusionNet] fused_dim={fused_dim} "
            f"(img={img_dim} + attn={attn_dim} + meta={meta_out})"
        )

    def forward(
        self,
        img_feat:   torch.Tensor,
        txt_tokens: torch.Tensor,
        meta_feat:  torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through triple-fusion network.

        Args:
            img_feat   (torch.Tensor): Image features (B, 1024).
            txt_tokens (torch.Tensor): BERT tokens (B, seq_len, 768).
            meta_feat  (torch.Tensor): Metadata embedding (B, 64).

        Returns:
            torch.Tensor: Class logits of shape (B, 2).
        """
        q = self.query_proj(img_feat).unsqueeze(1)    # (B, 1, 512)
        k = self.key_proj(txt_tokens)                  # (B, seq, 512)
        v = self.val_proj(txt_tokens)                  # (B, seq, 512)
        attended, _ = self.cross_attn(q, k, v)         # (B, 1, 512)
        attended = self.norm1(attended.squeeze(1))      # (B, 512)
        fused = torch.cat([img_feat, attended, meta_feat], dim=1)  # (B, 1600)
        return self.classifier(fused)

    def load_phase2_weights(self, checkpoint_path: str) -> None:
        """Warm-start cross-attention layers from a Phase 2 checkpoint.

        Loads ``query_proj``, ``key_proj``, ``val_proj``, ``cross_attn``,
        and ``norm1`` weights from a Phase 2 ``.pth`` file.  The classifier
        head is left randomly initialised (different input dimension).

        Args:
            checkpoint_path (str): Path to Phase 2 ``best_v2_model.pth``.
        """
        import torch, os
        if not os.path.exists(checkpoint_path):
            print(f"[TripleFusionNet] WARNING: checkpoint not found at {checkpoint_path}")
            return
        state = torch.load(checkpoint_path, map_location="cpu")
        fusion_state = state.get("fusion", state)
        cross_attn_keys = {
            k: v for k, v in fusion_state.items()
            if any(
                k.startswith(pfx)
                for pfx in ["query_proj", "key_proj", "val_proj", "cross_attn", "norm1"]
            )
        }
        missing, unexpected = self.load_state_dict(cross_attn_keys, strict=False)
        print(
            f"[TripleFusionNet] Warm-started {len(cross_attn_keys)} cross-attn keys "
            f"| missing={len(missing)} unexpected={len(unexpected)}"
        )
