from enum import StrEnum


class TorchDevices(StrEnum):
    """cuda, cpu, mps"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
