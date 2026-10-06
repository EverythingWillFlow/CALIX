"""Composite reward r_t = alpha * r_task + beta * r_eff + lambda * r_coord.

Implementation-table settings: alpha = 1.0, beta = 0.1, lambda = 0.1,
terminal success bonus 5.0, efficiency max(0, 1 - t/T_max), coordination
max(0, 1 - C/(N x 20)).
"""
from __future__ import annotations


def compute_reward(
    success: bool,
    step: int,
    max_steps: int,
    communications: int,
    num_agents: int,
    action_success: bool = False,
    alpha: float = 1.0,
    beta: float = 0.1,
    lam: float = 0.1,
    success_bonus: float = 5.0,
    coord_budget_per_agent: int = 20,
) -> dict:

    r_task = (0.5 if action_success else 0.0) + (1.0 if success else 0.0)
    r_eff = max(0.0, 1.0 - step / max_steps)
    r_coord = max(0.0, 1.0 - communications / (num_agents * coord_budget_per_agent))
    reward = alpha * r_task + beta * r_eff + lam * r_coord
    if success:
        reward += success_bonus
    return {
        "reward": reward,
        "r_task": r_task,
        "r_eff": r_eff,
        "r_coord": r_coord,
        "bonus": success_bonus if success else 0.0,
    }
