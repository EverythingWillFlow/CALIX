"""Dynamic interaction graph G_t and its evolution operator Phi_G.

Paper specification (implementation table, "Graph" section):
- nodes: agents + policy (task-level entities)
- edges: ``assigned_to`` (policy-agent relation)
- node features: 768-d embeddings (context / output features)
- aggregation: mean pooling
- evolution: G_{t+1} = Phi_G(G_t, H_{t+1}, x_{t+1}) per step
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphNode:
    name: str
    kind: str  # "agent" | "policy"
    feature: np.ndarray


@dataclass
class DynamicInteractionGraph:
    """Task-level interaction graph with per-step evolution."""

    roles: list[str]
    feature_dim: int = 768
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def initialise(self, role_features: dict[str, np.ndarray]) -> None:
        """Create one node per agent role (called at episode start)."""
        self.nodes = {
            role: GraphNode(name=role, kind="agent", feature=role_features[role].copy())
            for role in self.roles
        }
        self.edges = []

    # -- evolution operator Phi_G -------------------------------------------
    def evolve(
        self,
        policy_text: str,
        policy_feature: np.ndarray,
        agent_outputs: dict[str, np.ndarray] | None,
        observation_feature: np.ndarray,
    ) -> None:
        """G_{t+1} = Phi_G(G_t, H_{t+1}, x_{t+1}).

        Adds / refreshes the policy node, connects it to every agent via
        ``assigned_to`` edges, and updates agent node features with their
        latest intermediate-output embeddings and the new observation.
        """
        policy_node = "policy"
        self.nodes[policy_node] = GraphNode(
            name=policy_node, kind="policy", feature=policy_feature.copy()
        )
        self.edges = [(policy_node, role, "assigned_to") for role in self.roles]
        for role in self.roles:
            node = self.nodes[role]
            update = 0.5 * observation_feature
            if agent_outputs and role in agent_outputs:
                update = update + 0.5 * agent_outputs[role]
            else:
                update = update + 0.5 * policy_feature
            # exponential trace update keeps features bounded
            node.feature = 0.7 * node.feature + 0.3 * update

    # -- encoder f_G ---------------------------------------------------------
    def encode(self) -> np.ndarray:
        """Mean pooling over node features -> 768-d graph representation."""
        if not self.nodes:
            return np.zeros(self.feature_dim)
        feats = np.stack([n.feature for n in self.nodes.values()])
        return feats.mean(axis=0)
