
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import DeterministicEmbedder

ROLES = ["Planner", "Researcher", "Analyst", "Executor", "Verifier", "Integrator"]

# per-role contribution to fusion quality: removing a role scales the fused
# output's effectiveness (used by the environment's progress model in the
# leave-one-agent-out analysis).  Integrator / Executor are the most critical.
ROLE_IMPORTANCE = {
    "Planner": 0.93,
    "Researcher": 0.95,
    "Analyst": 0.96,
    "Executor": 0.90,
    "Verifier": 0.93,
    "Integrator": 0.85,
}


@dataclass
class AgentOutput:
    role: str
    text: str
    confidence: float
    embedding: np.ndarray


class CollaborativeAgentTeam:
    """Sequential execution + highest-confidence fusion."""

    def __init__(self, roles: list[str], embedder: DeterministicEmbedder, seed: int = 0):
        self.roles = roles
        self.embedder = embedder
        self._rng = np.random.default_rng(seed)

    # -- policy decomposition -------------------------------------------------
    def decompose(self, policy_text: str) -> dict[str, str]:
        """c_t* -> {u_t^1, ..., u_t^N}: role-specific sub-policies."""
        return {
            role: f"[{role}] Execute your specialised part of: {policy_text}"
            for role in self.roles
        }

    # -- execution -------------------------------------------------------------
    def execute(
        self,
        sub_policies: dict[str, str],
        strategy_match: bool,
        removed_role: str | None = None,
    ) -> tuple[list[AgentOutput], dict]:

        outputs: list[AgentOutput] = []
        active_roles = [r for r in self.roles if r != removed_role]
        for role in active_roles:
            base = 0.75 if strategy_match else 0.35
            # role criticality: Executor / Verifier carry more weight
            if role in ("Executor", "Verifier"):
                base += 0.05
            confidence = float(np.clip(base + self._rng.normal(0, 0.08), 0.05, 0.99))
            text = f"{role} intermediate result for sub-policy: {sub_policies[role][:120]}"
            outputs.append(
                AgentOutput(
                    role=role,
                    text=text,
                    confidence=confidence,
                    embedding=self.embedder.embed(text),
                )
            )
        best = max(outputs, key=lambda o: o.confidence)  # Fusion: highest confidence
        n = len(active_roles)
        info = {
            "fused_text": best.text,
            "fused_role": best.role,
            "confidence": best.confidence,
            "mean_confidence": float(np.mean([o.confidence for o in outputs])),
            "communications": n * (n - 1),   # pairwise intermediate sharing
            "num_active_agents": n,
            # fusion quality degrades when a role is missing (leave-one-out)
            "fusion_quality": ROLE_IMPORTANCE.get(removed_role, 1.0) if removed_role else 1.0,
        }
        return outputs, info
