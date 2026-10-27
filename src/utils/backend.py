"""Tiny array-backend shim so the fuzzy solvers run unchanged on NumPy (CPU) or PyTorch (CUDA).

Only the handful of operations the solvers need are exposed.  Importing torch is
optional – if it is missing or CUDA is unavailable we fall back to CPU without crashing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("neuroscanx")


class NumpyOps:
    name = "cpu"

    def array(self, x):
        return np.ascontiguousarray(x, dtype=np.float32)

    def host(self, x):
        return np.asarray(x)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float32)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def clip_min(self, x, v):
        return np.maximum(x, np.float32(v))

    def pow(self, x, p):
        return np.power(x, np.float32(p))

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def max_abs_diff(self, a, b):
        return float(np.abs(a - b).max())

    def to_float(self, x):
        return float(x)

    def free(self):
        pass


class TorchOps:
    def __init__(self, device: str):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.name = "cuda" if device.startswith("cuda") else "cpu"

    def array(self, x):
        t = self.torch
        return t.as_tensor(np.ascontiguousarray(x, dtype=np.float32), device=self.device)

    def host(self, x):
        return x.detach().cpu().numpy()

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.torch.float32, device=self.device)

    def zeros_like(self, x):
        return self.torch.zeros_like(x)

    def clip_min(self, x, v):
        return self.torch.clamp(x, min=float(v))

    def pow(self, x, p):
        return self.torch.pow(x, float(p))

    def exp(self, x):
        return self.torch.exp(x)

    def log(self, x):
        return self.torch.log(x)

    def max_abs_diff(self, a, b):
        return float((a - b).abs().max().item())

    def to_float(self, x):
        return float(x.item()) if hasattr(x, "item") else float(x)

    def free(self):
        if self.name == "cuda":
            self.torch.cuda.empty_cache()


@dataclass
class DeviceInfo:
    requested: str
    active: str            # "cpu" | "cuda"
    name: str
    message: str


def detect_device(requested: str = "auto") -> DeviceInfo:
    """Detect CUDA automatically; never raises."""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return DeviceInfo(requested, "cpu", "CPU", "CPU mode (forced)")
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            nm = torch.cuda.get_device_name(0)
            return DeviceInfo(requested, "cuda", nm, f"GPU mode: CUDA ({nm})")
        msg = "CUDA unavailable → falling back to CPU"
    except Exception:
        msg = "CUDA unavailable → falling back to CPU"
    return DeviceInfo(requested, "cpu", "CPU", msg)


def get_ops(device: str = "cpu"):
    if device == "cuda":
        try:
            return TorchOps("cuda")
        except Exception as e:  # pragma: no cover - depends on hardware
            log.warning("CUDA backend failed (%s) → falling back to CPU", e)
    return NumpyOps()
