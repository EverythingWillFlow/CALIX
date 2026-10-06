"""Behaviour-aware state encoder phi_xi and Algorithm 2 (state construction).

s_t = phi_xi(FusionState(f_G(G_t), f_H(H_t^ret), f_X(x_t)))

Implementation-table settings:
- state dimension 768, per-modality hidden projection 256
- encoders f_G / f_H / f_X project each 768-d modality to 256-d
- phi_xi maps the fused 768-d representation to the 768-d latent state
- xi is randomly initialised once (fixed seed) and kept FROZEN during
  temporal-difference optimisation of the Q-network.
"""
from __future__ import annotations

import numpy as np

from .embedding import DeterministicEmbedder
from .graph import DynamicInteractionGraph
from .memory import HistoricalMemory, MemoryItem


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class BehaviorAwareStateEncoder:
    """Frozen behaviour-aware state encoder phi_xi (parameters xi)."""

    def __init__(
        self,
        embedder: DeterministicEmbedder,
        state_dim: int = 768,
        hidden_dim: int = 256,
        seed: int = 13,
        use_graph: bool = True,
        use_behavior: bool = True,
    ):
        self.embedder = embedder
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_graph = use_graph
        self.use_behavior = use_behavior
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(768)
        # modality projections f_G, f_H, f_X : 768 -> 256
        self.W_G = rng.standard_normal((768, hidden_dim)) * scale
        self.W_H = rng.standard_normal((768, hidden_dim)) * scale
        self.W_X = rng.standard_normal((768, hidden_dim)) * scale
        # phi_xi : 768 (3 x 256 fused) -> 768
        self.W_phi = rng.standard_normal((3 * hidden_dim, state_dim)) * (
            1.0 / np.sqrt(3 * hidden_dim)
        )
        self.b_phi = np.zeros(state_dim)

    # -- Algorithm 2 ---------------------------------------------------------
    def construct(
        self,
        graph: DynamicInteractionGraph,
        memory: HistoricalMemory,
        observation_text: str,
    ) -> tuple[np.ndarray, dict]:
        """Return latent state s_t and the intermediate representations."""
        x_emb = self.embedder.embed(observation_text)
        retrieved = memory.retrieve(x_emb) if self.use_behavior else []

        h_G_raw = graph.encode() if self.use_graph else np.zeros(768)
        h_H_raw = memory.encode(retrieved) if self.use_behavior else np.zeros(768)
        h_X_raw = x_emb

        h_G = _relu(h_G_raw @ self.W_G)
        h_H = _relu(h_H_raw @ self.W_H)
        h_X = _relu(h_X_raw @ self.W_X)

        h = np.concatenate([h_G, h_H, h_X])          # FusionState
        s = _relu(h @ self.W_phi + self.b_phi)        # phi_xi projection
        aux = {
            "observation_embedding": x_emb,
            "retrieved": retrieved,
            "h_G": h_G,
            "h_H": h_H,
            "h_X": h_X,
        }
        return s, aux
