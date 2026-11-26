from typing import Optional

import torch

from common.constants.torch_devices import TorchDevices


def torch_available_device(override_with: Optional[TorchDevices] = None) -> TorchDevices:
    """
    Return the override device if given, otherwise the best available torch device.

    If running in production, it is recommended to override with CPU unless a GPU is attached to the environment.

    Args:
        override_with: Device to force. If None, the device is auto-detected in the order CUDA, MPS, CPU.

    Returns:
        The override device, or the best available device.

    Raises:
        TypeError: If override_with is neither None nor a TorchDevices member.
    """
    if override_with is not None:
        if not isinstance(override_with, TorchDevices):
            raise TypeError(
                f"override_with must be a TorchDevices member or None, "
                f"got {type(override_with).__name__}"
            )
        return override_with

    if torch.cuda.is_available():
        return TorchDevices.CUDA

    if torch.backends.mps.is_available():
        return TorchDevices.MPS

    return TorchDevices.CPU
