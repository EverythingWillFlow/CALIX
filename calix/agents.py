"""Six-role collaborative agent team with real LLM execution.

Every agent output y_t^i is produced by a real LLM call using the versioned
role prompt in ``prompts/agents/<role>.txt``.  There are no template outputs,
no preset confidences and no hardcoded role weights: confidence values are
self-reported by the LLM (optional ``Confidence:`` line) and the fused
decision is produced by the real fusion prompt in ``prompts/fusion.txt``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import DeterministicEmbedder
from .llm import OpenAICompatibleLLM

ROLES = ["Planner", "Researcher", "Analyst", "Executor", "Verifier", "Integrator"]

_CONTEXT_CHARS = 600  # per-agent context forwarded to later roles


@dataclass
class AgentOutput:
    role: str
    text: str
    confidence: float | None
    embedding: np.ndarray


class CollaborativeAgentTeam:
    """Sequential real execution + LLM fusion."""

    def __init__(self, roles: list[str], embedder: DeterministicEmbedder,
                 llm: OpenAICompatibleLLM):
        self.roles = roles
        self.embedder = embedder
        self.llm = llm

    # -- policy decomposition -------------------------------------------------
    def decompose(self, policy_text: str) -> dict[str, str]:
        """c_t* -> {u_t^1, ..., u_t^N}: role-specific sub-policies (real LLM)."""
        return self.llm.decompose_policy(policy_text, self.roles)

    # -- execution -------------------------------------------------------------
    def execute(
        self,
        sub_policies: dict[str, str],
        observation: str,
        removed_role: str | None = None,
        max_tool_calls_per_role: int = 2, protocol: str = "",
    ) -> tuple[list[AgentOutput], dict]:
        """Run each active role on its sub-policy, then fuse the outputs."""
        outputs: list[AgentOutput] = []
        active_roles = [r for r in self.roles if r != removed_role]
        shared_parts: list[str] = []
        tool_calls = 0
        for role in active_roles:
            shared_context = "\n".join(shared_parts)
            text, confidence, n_tools = self.llm.run_role(
                role, sub_policies[role], observation, shared_context,
                max_tool_calls=max_tool_calls_per_role,
            )
            tool_calls += n_tools
            outputs.append(
                AgentOutput(
                    role=role,
                    text=text,
                    confidence=confidence,
                    embedding=self.embedder.embed(text),
                )
            )
            shared_parts.append(f"[{role}] {text[:_CONTEXT_CHARS]}")
        fused = self.llm.fuse_outputs(
            [{"role": o.role, "text": o.text, "confidence": o.confidence}
             for o in outputs],
            protocol=protocol,
        )
        n = len(active_roles)
        info = {
            "fused_text": fused["decision"],
            "fusion_confidence": fused["confidence"],
            "fusion_rationale": fused["rationale"],
            "mean_confidence": float(np.mean(
                [o.confidence for o in outputs if o.confidence is not None]
            )) if any(o.confidence is not None for o in outputs) else None,
            "communications": n * (n - 1),   # pairwise intermediate sharing
            "num_active_agents": n,
            "tool_calls": tool_calls,
        }
        return outputs, info
