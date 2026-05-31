import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Run one full pass over the training set and update model weights.
    Decoupled from the model architecture so the loops can run on any nn.Module.

    Args:
        model: Classifier in train mode.
        loader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimiser: Optimiser instance (Adam).
        device: Device to run forward/backward passes on.

    Returns:
        Tuple of (average_loss, accuracy) across all batches this epoch.
    """
    model.train()
    total_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the model on a DataLoader without updating weights.
    Decoupled from the model architecture so the loops can run on any nn.Module.

    Decorated with @torch.no_grad() to disable gradient tracking for the
    entire function, reducing memory usage and speeding up inference.

    Args:
        model: Classifier to evaluate.
        loader: DataLoader to evaluate on (val or test).
        criterion: Loss function (CrossEntropyLoss).
        device: Device to run forward passes on.

    Returns:
        Tuple of (average_loss, accuracy) across all batches in the loader.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> tuple[list[int], list[int]]:
    """
    Collect ground-truth labels and model predictions from a DataLoader.
    Decoupled from the model architecture so the loops can run on any nn.Module.

    Used after training to generate the classification report and confusion
    matrix without re-running the full evaluation loop.

    Args:
        model: Trained classifier.
        loader: DataLoader to run inference on (typically the test split).
        device: Device to run forward passes on.

    Returns:
        Tuple of (all_labels, all_preds) — ground-truth class indices and
        predicted class indices, as flat integer lists.
    """
    model.eval()
    all_labels: list[int] = []
    all_preds:  list[int] = []

    for images, labels in loader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_labels, all_preds
