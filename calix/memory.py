"""Historical memory H_t with top-k cosine retrieval and update Phi_H.

Paper specification (implementation table, "Memory" section):
- representation: text + embedding + metadata
- capacity: 1000 items, FIFO replacement
- retrieval: cosine similarity, top-5
- content: task / policy / reward / outcome + environment feedback
- update: H_{t+1} = Phi_H(H_t, s_t, c_t*, r_t) per step
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .embedding import cosine


@dataclass
class MemoryItem:
    task_id: str
    observation: str
    policy_text: str
    reward: float
    outcome: str
    feedback: str
    embedding: np.ndarray


class HistoricalMemory:
    def __init__(self, capacity: int = 1000, topk: int = 5):
        self.capacity = capacity
        self.topk = topk
        self._items: deque[MemoryItem] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    # -- update operator Phi_H ----------------------------------------------
    def update(
        self,
        task_id: str,
        observation: str,
        policy_text: str,
        reward: float,
        outcome: str,
        feedback: str,
        embedding: np.ndarray,
    ) -> None:
        """H_{t+1} = Phi_H(H_t, s_t, c_t*, r_t): append the newest record."""
        self._items.append(
            MemoryItem(
                task_id=task_id,
                observation=observation,
                policy_text=policy_text,
                reward=reward,
                outcome=outcome,
                feedback=feedback,
                embedding=embedding,
            )
        )

    # -- retrieval -----------------------------------------------------------
    def retrieve(self, query_embedding: np.ndarray, topk: int | None = None) -> list[MemoryItem]:
        """Top-k cosine-similarity retrieval against the current observation."""
        k = topk or self.topk
        if not self._items:
            return []
        scored = [(cosine(query_embedding, it.embedding), it) for it in self._items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    # -- encoder f_H ---------------------------------------------------------
    def encode(self, retrieved: list[MemoryItem]) -> np.ndarray:
        """Similarity-weighted mean of retrieved embeddings -> 768-d vector."""
        if not retrieved:
            return np.zeros_like(np.zeros(768))
        feats = np.stack([it.embedding for it in retrieved])
        return feats.mean(axis=0)
