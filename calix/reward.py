"""Composite reward r_t = alpha * r_task + beta * r_eff + lambda * r_coord.

Paper definition (Eq. 20 and the implementation table):

    r_task  = S * 1[ the terminal task success condition is met ],   S = 5.0
    r_eff   = max(0, 1 - t / T_max)
    r_coord = max(0, 1 - C_t / (N * 20))

with alpha = 1.0, beta = 0.1, lambda = 0.1 and T_max = 20.

Per-episode return bound: the success bonus contributes at most 5.0, the
efficiency term sums to at most 0.1 * sum_{t<T} (1 - t/T) <= 0.1 * T = 2.0
(1.05 for T = 20), and the coordination term to at most 0.1 * T = 2.0, so the
return is bounded by 5.0 + 1.05 + 2.0 = 8.05 for T = 20.
"""
from __future__ import annotations


def compute_reward(
    success: bool,
    step: int,
    max_steps: int,
    communications: int,
    num_agents: int,
    alpha: float = 1.0,
    beta: float = 0.1,
    lam: float = 0.1,
    success_bonus: float = 5.0,
    coord_budget_per_agent: int = 20,
) -> dict:

    r_task = success_bonus if success else 0.0
    r_eff = max(0.0, 1.0 - step / max_steps)
    r_coord = max(0.0, 1.0 - communications / (num_agents * coord_budget_per_agent))
    reward = alpha * r_task + beta * r_eff + lam * r_coord
    return {
        "reward": reward,
        "r_task": r_task,
        "r_eff": r_eff,
        "r_coord": r_coord,
        "bonus": success_bonus if success else 0.0,
    }
