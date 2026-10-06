"""Candidate collaborative policies and the encoding function psi(.).

Each candidate c_t^k is a structured textual policy; psi maps it to a
fixed-dimensional numerical representation z_t^k (768-d, deterministic).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embedding import DeterministicEmbedder


STRATEGY_ARCHETYPES = [
    "decompose-first",
    "retrieve-then-act",
    "execute-verify",
    "iterate-refine",
    "direct-answer",
    "parallel-explore",
    "consensus-vote",
    "tool-augmented",
]

# Distinct descriptor sentences per archetype.  They make the candidate-policy
# embedding z = psi(c) discriminative between strategies, which is what the
# Q-network conditions on when estimating candidate values.
STRATEGY_DESCRIPTIONS = {
    "decompose-first": "Hierarchically decompose the objective into ordered subtasks before any execution, then solve subtask by subtask.",
    "retrieve-then-act": "First retrieve and gather all relevant external knowledge and context, then act on the consolidated evidence.",
    "execute-verify": "Execute the candidate solution immediately and rely on rigorous verification loops to guarantee correctness.",
    "iterate-refine": "Produce a quick draft solution, then iteratively refine and revise it through repeated self-critique cycles.",
    "direct-answer": "Answer directly in a single straightforward step without decomposition or additional tool use.",
    "parallel-explore": "Explore multiple independent solution branches in parallel and select the most promising branch.",
    "consensus-vote": "Let all agents propose solutions independently and aggregate them through consensus voting.",
    "tool-augmented": "Augment reasoning with external tools, plugins and APIs at every decision point.",
}


@dataclass
class CandidatePolicy:
    index: int                     # k in 1..K
    text: str                      # structured textual policy
    strategy: int                  # latent strategy id
    embedding: np.ndarray = field(default=None, repr=False)  # z = psi(c)


def encode_candidate(policy: CandidatePolicy, embedder: DeterministicEmbedder) -> np.ndarray:
    """z_t^k = psi(c_t^k): deterministic 768-d encoding of the policy text."""
    if policy.embedding is None:
        policy.embedding = embedder.embed(policy.text)
    return policy.embedding
