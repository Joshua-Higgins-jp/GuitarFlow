from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    all_labels: list[int],
    all_preds: list[int],
    output_path: Path,
    class_names: list[str],
) -> Path:
    """
    Generate and save a seaborn heatmap of the confusion matrix locally.
    Filename and filetype is passed in. Returning the Path is redundant but just in case
    we ever modify the filename setting logic internally, it could be useful.

    Args:
        all_labels:  Ground truth integer class indices.
        all_preds:   Predicted integer class indices.
        output_path: Full path to write the PNG to.
        class_names: Ordered list of class name strings for axis labels.

    Returns:
        output_path: Full path of the written image file.
    """
    cm = confusion_matrix(y_true=all_labels, y_pred=all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    fig.savefig(fname=output_path)
    plt.close(fig=fig)
    logger.info(f"Confusion matrix saved → {output_path.name}")

    return output_path
