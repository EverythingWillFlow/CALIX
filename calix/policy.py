"""Candidate collaborative policies and the encoding function psi(.).

Each candidate c_t^k is a free-form structured textual policy produced by the
real LLM; psi maps it to a fixed-dimensional numerical representation z_t^k
(768-d, deterministic).  There is no latent "correct strategy" identifier:
that concept belonged to the retired simulation harness and was removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embedding import DeterministicEmbedder


@dataclass
class CandidatePolicy:
    index: int                     # k in 1..K
    text: str                      # structured textual policy from the LLM
    embedding: np.ndarray = field(default=None, repr=False)  # z = psi(c)


def encode_candidate(policy: CandidatePolicy, embedder: DeterministicEmbedder) -> np.ndarray:
    """z_t^k = psi(c_t^k): deterministic 768-d encoding of the policy text."""
    if policy.embedding is None:
        policy.embedding = embedder.embed(policy.text)
    return policy.embedding
