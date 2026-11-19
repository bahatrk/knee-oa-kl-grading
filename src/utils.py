"""Tekrarlanabilirlik, loglama ve ortam bilgisi yardımcıları."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Python, NumPy ve (varsa) PyTorch rastgelelik kaynaklarını sabitler."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def env_info() -> dict:
    """Rapora yazılacak donanım / yazılım bilgisi."""
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    libs = ["numpy", "pandas", "PIL", "imagehash", "matplotlib", "sklearn", "skimage",
            "torch", "torchvision", "timm", "optuna", "pytorch_grad_cam", "captum"]
    versions = {}
    for lib in libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "?")
        except Exception:
            versions[lib] = None
    info["libraries"] = versions
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["cuda_available"] = None
    return info


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
