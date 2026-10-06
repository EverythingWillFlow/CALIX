
from __future__ import annotations

import numpy as np

from .backend import resolve_backend, to_host


class QNetwork:
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512,
                 lr: float = 1e-4, grad_clip: float = 5.0, seed: int = 0,
                 device: str = "auto"):
        self.xp, self.using_gpu = resolve_backend(device)
        xp = self.xp
        # host-side initialisation: identical initial weights on every backend
        rng = np.random.default_rng(seed)
        self._w = {
            "W1": xp.asarray((rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)).astype(np.float32)),
            "b1": xp.zeros(hidden_dim, dtype=xp.float32),
            "W2": xp.asarray((rng.standard_normal((hidden_dim, hidden_dim)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)),
            "b2": xp.zeros(hidden_dim, dtype=xp.float32),
            "W3": xp.asarray((rng.standard_normal((hidden_dim, 1)) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)),
            "b3": xp.zeros(1, dtype=xp.float32),
        }
        self.lr = lr
        self.grad_clip = grad_clip
        self._m = {k: xp.zeros_like(v) for k, v in self._w.items()}
        self._v = {k: xp.zeros_like(v) for k, v in self._w.items()}
        self._tmp = {k: xp.zeros_like(v) for k, v in self._w.items()}
        self._tmp2 = {k: xp.zeros_like(v) for k, v in self._w.items()}
        self._t = 0
        self._beta1, self._beta2, self._eps = 0.9, 0.999, 1e-8

    # -- checkpointing (host NumPy dicts) ------------------------------------
    def get_params(self) -> dict[str, np.ndarray]:
        return {k: to_host(v).copy() for k, v in self._w.items()}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        xp = self.xp
        self._w = {k: xp.asarray(np.asarray(v), dtype=xp.float32)
                   for k, v in params.items()}

    # -- forward -------------------------------------------------------------
    def forward(self, x):
        xp = self.xp
        x = xp.ascontiguousarray(x, dtype=xp.float32)
        z1 = x @ self._w["W1"] + self._w["b1"]
        a1 = xp.maximum(z1, 0.0)
        z2 = a1 @ self._w["W2"] + self._w["b2"]
        a2 = xp.maximum(z2, 0.0)
        q = a2 @ self._w["W3"] + self._w["b3"]
        return q, (x, z1, a1, z2, a2)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Host NumPy in -> host NumPy (float64) out."""
        q, _ = self.forward(np.atleast_2d(x))
        return to_host(q.ravel()).astype(np.float64)

    # -- temporal-difference update ------------------------------------------
    def td_update(self, x: np.ndarray, targets: np.ndarray) -> float:
        xp = self.xp
        x = xp.ascontiguousarray(np.atleast_2d(x), dtype=xp.float32)
        targets = xp.asarray(targets.reshape(-1, 1), dtype=xp.float32)
        B = x.shape[0]
        q, (xin, z1, a1, z2, a2) = self.forward(x)

        diff = q - targets
        loss = float(xp.mean(diff * diff, dtype=xp.float64))

        dq = (2.0 / B) * diff
        gW3 = a2.T @ dq
        gb3 = dq.sum(axis=0)
        dz2 = (dq @ self._w["W3"].T) * (z2 > 0)
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self._w["W2"].T) * (z1 > 0)
        gW1 = xin.T @ dz1
        gb1 = dz1.sum(axis=0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}

        total = float(xp.sqrt(sum(float(xp.sum(g * g, dtype=xp.float64))
                                  for g in grads.values())))
        if total > self.grad_clip:
            scale = np.float32(self.grad_clip / (total + 1e-12))
            grads = {k: g * scale for k, g in grads.items()}

        # in-place Adam (on the active backend)
        self._t += 1
        b1, b2