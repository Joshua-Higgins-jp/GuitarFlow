import torch

# Preference order: CUDA (Nvidia GPU) → MPS (Apple Silicon) → CPU.
# Training on CPU is ~10x slower but functionally correct.

# If you want to specify the device, remove the logic and just specify "device"
# For example, train on your MacBook -> mps. Cool. Then use torch.load(model.pt, map_location="cpu")
# and model.to("cpu") in the deployment if you wanna avoid GPU costs for inference.
# If you're worried about inference differences such as float errors between cpu and mps, test it out,
# or just run training on cpu and deploy with inference as cpu.

TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
