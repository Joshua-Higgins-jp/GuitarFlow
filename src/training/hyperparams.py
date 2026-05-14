"""
Training hyperparameters for the GuitarFlow classifier.

All hyperparameters are collected in a single frozen dataclass so that:
  - There is one source of truth for every training run.
  - MLflow can log the full config in one shot via config.__dict__.
  - Accidental mutation mid-run is impossible (frozen=True raises FrozenInstanceError).

Usage:

    from training.hyperparams import TrainingConfig

    config = TrainingConfig()
    print(config.epochs)          # 20
    print(config.__dict__)        # full dict, ready for mlflow.log_params()
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """
    Frozen hyperparameter configuration for a single GuitarFlow training run.

    Frozen means no field can be reassigned after instantiation. If you need
    to experiment with a different value, instantiate a new config:

        config = TrainingConfig(epochs=30, lr=1e-3)

    All defaults are the baseline used for training-set-001.

    Fields:
        epochs:
            Total number of full passes over the training set.
            More epochs allow the model to learn finer patterns but risk
            overfitting once val loss stops improving.
            Reasonable range: 10-50 for fine-tuning a pretrained backbone.
            JUSTIFY THIS: 20 chosen as a conservative starting point for a
            small dataset (~600 images). Watch val loss — if it flattens
            before epoch 20, add early stopping.

        batch_size:
            Number of images fed to the model per gradient update step.
            Larger batches are faster but require more GPU/MPS memory and
            can generalise slightly worse on small datasets.
            Reasonable range: 8-64. Powers of 2 are conventional.
            JUSTIFY THIS: 16 chosen to fit comfortably in MPS (Apple Silicon)
            memory. Increase to 32 if training on CUDA with >4 GB VRAM.

        lr:
            Learning rate — the step size for each gradient update.
            Too high: loss diverges or oscillates. Too low: training is slow
            and may get stuck in a local minimum.
            Reasonable range: 1e-4 to 1e-2 for Adam with a pretrained backbone.
            JUSTIFY THIS: 3e-3 is on the higher end for fine-tuning. Acceptable
            here because only layer4 + the classification head are unfrozen,
            so the pretrained features are protected. If val loss is noisy,
            drop to 1e-3.

        val_split:
            Fraction of the full dataset reserved for validation (monitoring
            overfitting during training). Not used for gradient updates.
            Reasonable range: 0.10-0.20.
            JUSTIFY THIS: 0.15 gives ~90 val images at 600 total. Enough to
            get a stable accuracy signal without eating too much training data.

        test_split:
            Fraction of the full dataset reserved for final held-out evaluation.
            This split is touched exactly once — after training is complete.
            Touching it earlier causes data leakage into your reported metrics.
            Reasonable range: 0.10-0.20.
            JUSTIFY THIS: 0.15 mirrors val_split for symmetry. Gives ~90
            test images — small but sufficient for a 3-class portfolio model.

        img_size:
            Height and width (H, W) that every input image is resized to
            before being fed into the model.
            ResNet18 was pretrained on ImageNet at 224x224. Using a different
            size is valid but degrades the pretrained feature quality.
            JUSTIFY THIS: Fixed at (224, 224) to match the pretrained weights.
            Do not change unless you replace the backbone.

        random_seed:
            Seed passed to train_test_split to make the data split
            reproducible across runs. The same seed + same dataset always
            produces the same train/val/test assignment.
            JUSTIFY THIS: 42 is conventional. The specific value does not
            matter — what matters is that it is fixed and logged to MLflow
            so any run can be reproduced exactly.
    """
    epochs:      int              = 20
    batch_size:  int              = 16
    lr:          float            = 3e-3  # 0.003
    val_split:   float            = 0.15
    test_split:  float            = 0.15
    img_size:    tuple[int, int]  = (224, 224)
    random_seed: int              = 42
