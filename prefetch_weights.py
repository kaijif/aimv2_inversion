"""Download every model the notebook uses into /work caches, so GPU jobs can run offline.
Run on a Longleaf login node:  python prefetch_weights.py"""
import os
from pathlib import Path

WORK = Path(os.environ.get("FI_WORK", "/work/users/k/a/kaiji"))
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
os.environ.setdefault("TORCH_HOME", str(WORK / "torch_cache"))

from huggingface_hub import snapshot_download

REPOS = [
    "apple/aimv2-large-patch14-224",
    "openai/clip-vit-large-patch14",
    "google/siglip-large-patch16-256",
    "facebook/dinov2-large",
    "facebook/vit-mae-large",
]
SKIP = ["*.h5", "*.msgpack", "*.onnx", "*tf_model*", "*flax_model*", "*.ot", "*.mlpackage/*", "*.mlmodel*"]
for repo in REPOS:
    path = snapshot_download(repo, ignore_patterns=SKIP)
    print(f"{repo:40s} -> {path}")

import torchvision, lpips
lpips.LPIPS(net="alex", verbose=False)   # pulls torchvision AlexNet into TORCH_HOME
lpips.LPIPS(net="vgg", verbose=False)
torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)  # semantic metric
print("all weights cached under", os.environ["HF_HOME"], "and", os.environ["TORCH_HOME"])
