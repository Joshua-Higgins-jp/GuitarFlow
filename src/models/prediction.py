from pathlib import Path
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from torchvision import models, transforms
from torchvision.models import ResNet

from config.globals import ClassLabels

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CLASS_NAMES = ClassLabels.as_tuple()

IMG_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose(
    transforms=[
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
)

def load_classification_model(
        weights_path: Path,
        device: Union[torch.device, str] = DEVICE,
        class_names: Tuple[str, ...] = CLASS_NAMES
) -> nn.Module:
    """
    Rebuild the ResNet18 architecture and load saved weights.

    Args:
        weights_path: Path to the saved .pth weights file.
        device: Device to run the model on.
        class_names: Tuple of class names.

    Returns:
        Model in eval mode, moved to the appropriate device.
    """
    model: ResNet = models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(class_names))
    model.load_state_dict(state_dict=torch.load(f=weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def classification_predict(
        image: Union[Path, Image.Image],
        model: nn.Module
) -> Dict[str, float]:
    """
    Run inference on a single guitar image.

    Args:
        image: Path to the image file, or an already-loaded PIL Image.
        model: Loaded model in eval mode.

    Returns:
        Dict mapping each class name to its confidence score (0–1, sums to 1).
    """
    if isinstance(image, Path):
        pil_image = Image.open(image).convert("RGB")
        label = image.name
    else:
        pil_image = image.convert("RGB")
        label = repr(image)  # e.g. "<PIL.JpegImagePlugin ... >"

    tensor = eval_transform(pil_image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze()                  # (num_classes,)

    scores    = {name: round(probs[i].item(), 4) for i, name in enumerate(CLASS_NAMES)}
    predicted = max(scores, key=scores.get)

    logger.info(
        "{} → {}  ({})",
        label,
        predicted,
        ", ".join(f"{name}: {score:.1%}" for name, score in scores.items()),
    )

    return scores
