"""
src/models/text_encoder.py
==========================
Bio_ClinicalBERT text encoder for PneumoFusionNet.

Contains:
    TextEncoder — Hugging Face AutoModel wrapper with partial fine-tuning.
                  Last 2 encoder layers (10, 11) + pooler are unfrozen;
                  all earlier layers are frozen.

Source notebooks:
    mimic/main/Scaleup/Phase-2v2-multimodal_improved_PA_Upscaled.ipynb
    mimic/main/Phase-3/Phase-3-triple_fusion_PA_Scaleup.ipynb
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Default pre-trained model name
CLINBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


class TextEncoder(nn.Module):
    """Bio_ClinicalBERT encoder with partial fine-tuning.

    Loads ``emilyalsentzer/Bio_ClinicalBERT`` (pre-trained on MIMIC-III
    clinical notes) and freezes all layers except the top 2 transformer
    encoder layers (layer 10 and 11) and the pooler head.  The unfrozen
    parameters are fine-tuned with a low learning rate (1e-5) alongside
    the fusion network.

    Returns the full ``last_hidden_state`` sequence of shape
    ``(B, seq_len, 768)`` so the cross-attention module can attend over
    individual tokens.

    Args:
        model_name (str): Hugging Face model identifier.
            Default: ``'emilyalsentzer/Bio_ClinicalBERT'``.

    Example::

        encoder = TextEncoder()
        tokens = tokenizer(texts, return_tensors='pt',
                           padding='max_length', truncation=True,
                           max_length=256)
        sequence = encoder(tokens['input_ids'], tokens['attention_mask'])
        # sequence: (B, 256, 768)
    """

    def __init__(self, model_name: str = CLINBERT_MODEL):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # Freeze all parameters first
        for p in self.bert.parameters():
            p.requires_grad = False

        # Unfreeze last 2 encoder layers and pooler for task fine-tuning
        for name, p in self.bert.named_parameters():
            if (
                "encoder.layer.10" in name
                or "encoder.layer.11" in name
                or "pooler" in name
            ):
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"[TextEncoder] Unfrozen BERT params: {trainable:,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode tokenised clinical text.

        Args:
            input_ids (torch.Tensor): Token IDs of shape (B, seq_len).
            attention_mask (torch.Tensor): Padding mask of shape (B, seq_len).

        Returns:
            torch.Tensor: Token sequence embeddings of shape (B, seq_len, 768).
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, seq_len, 768)


def build_tokenizer(model_name: str = CLINBERT_MODEL) -> AutoTokenizer:
    """Load the Bio_ClinicalBERT tokenizer.

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        AutoTokenizer pre-configured for Bio_ClinicalBERT.

    Example::

        tokenizer = build_tokenizer()
        enc = tokenizer(text, max_length=256, padding='max_length',
                        truncation=True, return_tensors='pt')
    """
    return AutoTokenizer.from_pretrained(model_name)
