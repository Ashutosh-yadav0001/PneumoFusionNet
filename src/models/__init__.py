"""PneumoFusionNet — Models sub-package.

Import only what you need:

    from src.models.vision import EnhancedPneumoNetV4, ImageEncoder, CBAM
    from src.models.text_encoder import TextEncoder, build_tokenizer
    from src.models.fusion import CrossAttnFusionNet, MetadataEncoder, TripleFusionNet

Note: Importing from this package requires ``torchxrayvision`` and
``transformers`` to be installed in your environment.
"""

__all__ = [
    # vision
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "EnhancedPneumoNetV4",
    "ImageEncoder",
    "build_model",
    # text
    "TextEncoder",
    "build_tokenizer",
    # fusion
    "CrossAttnFusionNet",
    "MetadataEncoder",
    "TripleFusionNet",
]
