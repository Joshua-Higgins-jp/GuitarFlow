import torch
from loguru import logger
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet


def build_model(
        num_classes: int,
        device: torch.device
) -> nn.Module:
    """
    Builds a fine-tuned ResNet18 classifier.

    Architecture strategy:
        - Load ResNet18 with pretrained ImageNet weights.
        - Freeze all layers to preserve learned low-level features.
        - Unfreeze layer4 (the last residual block) for domain adaptation.
        - Replace the final FC layer with a new Linear(in_features, num_classes).

    Freezing most of the backbone is appropriate for our dataset size (~max 1k images
    after balancing). Training all layers would overfit badly on this amount.
    Unfreezing layer4 is a tradeoff — the most abstract features adapt to
    guitar-specific patterns while the majority of pretrained representations are
    preserved. If val accuracy plateaus early, consider also unfreezing layer3.

    Args:
        num_classes: Number of output classes. Must match len(CLASS_NAMES).
        device: Torch device to use.

    Returns:
        ResNet18 model moved to DEVICE, with layer4 + classification head unfrozen.
    """
    model: ResNet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features: int = model.fc.in_features
    model.fc = nn.Linear(
        in_features=in_features,
        out_features=num_classes
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Built ResNet18 — {trainable:,} trainable params (layer4 + head)")

    return model.to(device)
