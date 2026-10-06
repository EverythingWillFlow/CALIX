"""Replay buffer D for state--selected-policy transitions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transition:
    state: np.ndarray                # s_t
    policy_embedding: np.ndarray     # z_t^*
    reward: float                    # r_t
    next_state: np.ndarray           # s_{t+1}
    done: bool                       # d_t
    next_candidate_embeddings: np.ndarray = field(default=None)  # (K, dim) or None at terminal


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, seed: int = 0):
        self.capacity = capacity
        self._data: list[Transition] = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._data)

    def add(self, tr: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(tr)
        else:
            self._data[self._pos] = tr
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        idx = self._rng.integers(0, len(self._data), size=min(batch_size, len(self._data)))
        return [self._data[i] for i in idx]
