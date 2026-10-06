"""Compute-backend selection for the numerical core of CALIX.

The Q-network runs its forward and backward passes with a NumPy-compatible
array library.  When an NVIDIA GPU with a CUDA 12.x driver is available
(developed and tested against CUDA 12.2), CuPy (``cupy-cuda12x``) is used so
every matrix operation executes on the GPU; otherwise the code falls back to
NumPy on the CPU with identical semantics.

Device resolution
-----------------
``"auto"`` (default) use the GPU when CuPy + a CUDA device are present,
                     else fall back to CPU
``"cuda"``           require the GPU; raise if unavailable
``"cpu"``            force the NumPy CPU backend
"""
from __future__ import annotations

import numpy as np

_CACHE: dict[str, tuple[object, bool]] = {}


def resolve_backend(device: str = "auto"):
    """Return (xp, using_gpu): the array module and whether it is the GPU."""
    key = (device or "auto").lower()
    if key in _CACHE:
        return _CACHE[key]

    xp, using_gpu = np, False
    if key in ("cuda", "gpu", "auto"):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() >= 1:
                xp, using_gpu = cp, True
        except Exception as exc:
            if key != "auto":
                raise RuntimeError(
                    f"device='{device}' requested but CUDA/CuPy is not "
                    f"available: {exc}"
                ) from exc
    result = (xp, using_gpu)
    _CACHE[key] = result
    return result


def to_host(a: object) -> np.ndarray:
    """Copy an array from the active backend back to host NumPy memory."""
    if isinstance(a, np.ndarray):
        return a
    import cupy as cp  # only reached when a CuPy array exists
    return cp.asnumpy(a)
