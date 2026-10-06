"""Decision-provenance recording and quantitative evaluation (paper Section 5.5).

For every decision step t the provenance record P_t contains:
    s_t, C_t, {z_t^k}, {Q(s_t, z_t^k)}, c_t*, {u_t^i}, {y_t^i}, y_t,
    x_{t+1}, r_t, s_{t+1}

Metrics:
- Path Completeness (PC): share of steps with all required fields present.
- Decision Fidelity (DF): share of steps where the recorded selection agrees
  with the Q-induced selection rule reconstructed from the logged Q-values
  (greedy argmax at evaluation time).
- Trace Stability (TS): share of repeated executions whose provenance
  *structure* (field schema per step) matches the reference trace.
"""
from __future__ import annotations

import json

import numpy as np

REQUIRED_FIELDS = [
    "state", "candidates", "candidate_embeddings", "q_values",
    "selected_policy", "sub_policies", "agent_outputs", "fused_output",
    "next_observation", "reward", "next_state",
]


class ProvenanceRecorder:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, step: int, **fields) -> None:
        rec = {"step": step}
        rec.update(fields)
        self.records.append(rec)

    # -- serialisation ---------------------------------------------------------
    def to_jsonable(self) -> list[dict]:
        out = []
        for rec in self.records:
            j = {}
            for k, v in rec.items():
                if isinstance(v, np.ndarray):
                    j[k] = {"shape": list(v.shape), "l2": float(np.linalg.norm(v))}
                elif isinstance(v, (np.floating, np.integer)):
                    j[k] = v.item()
                else:
                    j[k] = v
            out.append(j)
        return out

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_jsonable(), f, indent=2, ensure_ascii=False)


# -- quantitative metrics -------------------------------------------------------

def path_completeness(records: list[dict]) -> float:
    if not records:
        return 0.0
    complete = sum(
        1 for r in records if all(f in r and r[f] is not None for f in REQUIRED_FIELDS)
    )
    return 100.0 * complete / len(records)


def decision_fidelity(records: list[dict]) -> float:
    """Reconstruct greedy selection from logged Q-values and compare."""
    if not records:
        return 0.0
    consistent = 0
    for r in records:
        q = r.get("q_values")
        sel = r.get("selected_index")
        if q is None or sel is None:
            continue
        if int(np.argmax(np.asarray(q, dtype=float))) == int(sel):
            consistent += 1
    return 100.0 * consistent / len(records)


def trace_stability(traces: list[list[dict]]) -> float:
    """Share of traces whose step-schema *vocabulary* matches the reference.

    Traces naturally differ in length (episodes terminate at different
    steps), so stability is measured on the set of distinct per-step field
    schemas: a stable provenance logger records exactly the same fields at
    every step of every execution.
    """
    if len(traces) <= 1:
        return 100.0
    ref_schema = {tuple(sorted(r.keys())) for r in traces[0]}
    consistent = 0
    for tr in traces[1:]:
        if {tuple(sorted(r.keys())) for r in tr} == ref_schema:
            consistent += 1
    return 100.0 * consistent / (len(traces) - 1)
