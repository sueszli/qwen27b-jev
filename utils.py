# /// script
# requires-python = ">=3.12"
# dependencies = ["torch"]
# ///
import importlib.util
import os
import random
from pathlib import Path

import torch


def set_storage(weights_dir: Path) -> Path:
    (weights_dir / "tmp").mkdir(parents=True, exist_ok=True)
    hf, pt, xdg = weights_dir / "hf", weights_dir / "torch", weights_dir / "cache"
    os.environ.update({"HF_HOME": str(hf), "HF_HUB_CACHE": str(hf), "TRANSFORMERS_CACHE": str(hf), "HF_XET_CACHE": str(hf / "xet"), "HF_HUB_DISABLE_TELEMETRY": "1", "TORCH_HOME": str(pt), "TORCHINDUCTOR_CACHE_DIR": str(pt / "inductor"), "TRITON_CACHE_DIR": str(pt / "triton"), "CUDA_CACHE_PATH": str(weights_dir / "cuda"), "XDG_CACHE_HOME": str(xdg), "XDG_DATA_HOME": str(xdg), "XDG_CONFIG_HOME": str(xdg), "TMPDIR": str(weights_dir / "tmp")})
    return weights_dir


def set_seed(seed: int = 41) -> int:
    os.environ.update({"PYTHONHASHSEED": str(seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    random.seed(seed)
    if importlib.util.find_spec("numpy"):
        importlib.import_module("numpy").random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
