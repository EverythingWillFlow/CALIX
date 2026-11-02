"""Failure case and recovery analysis (paper Table: failure taxonomy).

Fully trace-derived: for every evaluation episode we record the decision
provenance, then

- a *failure event* is a step whose environment feedback marks the submitted
  action as incorrect (``step_success == False``);
- the failure *type* of each event is classified by the real LLM from the
  recorded context (policy text + environment feedback); the classifier label
  and rationale are stored in the trace for audit;
- *detection* means the system actually adapted: the policy selected at the
  step after the failure differs from the failed one;
- *recovery* means the episode still reached the success condition after a
  detected failure.

No random draws and no hardcoded probabilities are used anywhere.
"""
from __future__ import annotations

import argparse
import json
import os

from common import (ensure_output_dirs, load_benchmark, load_model_params,
                    make_env, train_calix, write_table)
from calix.calix import CALIX
from calix.config import load_config
from calix.provenance import ProvenanceRecorder

FAILURE_TYPES = [
    "planning_failure",
    "retrieval_failure",
    "execution_failure",
    "coordination_conflict",
    "verification_failure",
]

_LABELS = {
    "planning_failure": "Planning Failure",
    "retrieval_failure": "Retrieval Failure",
    "execution_failure": "Execution Failure",
    "coordination_conflict": "Coordination Conflict",
    "verification_failure": "Verification Failure",
}

_CLASSIFY_PROMPT = """You are auditing a failed decision step of a multi-agent system.

Task observation:
{observation}

Selected collaborative policy at the failed step:
{policy}

Environment feedback after execution:
{feedback}

Classify the failure into exactly one of:
1. planning_failure - the task decomposition or plan was wrong
2. retrieval_failure - insufficient, irrelevant, or incorrect information
3. execution_failure - a selected action did not produce the expected outcome
4. coordination_conflict - agents produced mutually inconsistent results
5. verification_failure - an incorrect intermediate result passed verification

Answer with exactly one line: <type> | <one-sentence justification>
"""


def classify_failure(llm, observation: str, policy: str, feedback: str) -> tuple[str, str]:
    raw = llm.chat(_CLASSIFY_PROMPT.format(
        observation=observation[:800], policy=policy[:800], feedback=feedback[:400]))
    head, _, why = raw.partition("|")
    label = head.strip().lower().replace(" ", "_")
    if label not in FAILURE_TYPES:
        # conservative fallback: mark as execution failure and keep the raw output
        return "execution_failure", f"unparseable classifier output: {raw[:120]}"
    return label, why.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--tasks", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)

    model_path = os.path.join(dirs["models"], f"qnet_seed{args.seed}.npz")
    if os.path.exists(model_path):
        model = CALIX(cfg, seed=args.seed)
        load_model_params(model, model_path)
    else:
        model, _ = train_calix(cfg, args.seed, args.episodes)

    per_type: dict[str, dict[str, int]] = {
        t: {"freq": 0, "detected": 0, "recovered": 0, "success": 0}
        for t in FAILURE_TYPES
    }
    audit_log: list[dict] = []

    for bench in cfg["benchmarks"]["names"]:
        _, test = load_benchmark(bench)
        env = make_env(test, bench, cfg)
        for ti in range(min(args.tasks, len(test))):
            recorder = ProvenanceRecorder()
            result = model.run_episode(env, task_index=ti, training=False,
                                       recorder=recorder)
            recs = recorder.records
            failed_steps = [i for i, r in enumerate(recs)
                            if not r.get("step_success", True)]
            seen_types = set()
            for i in failed_steps:
                r = recs[i]
                ftype, why = classify_failure(
                    model.llm,
                    observation=recs[0].get("next_observation", "") or "",
                    policy=r.get("selected_policy", ""),
                    feedback=r.get("next_observation", ""),
                )
                if ftype in seen_types:
                    continue
                seen_types.add(ftype)
                # detection: the next step selected a different policy
                detected = (i + 1 < len(recs)
                            and recs[i + 1].get("selected_policy") != r.get("selected_policy"))
                st = per_type[ftype]
                st["freq"] += 1
                if detected:
                    st["detected"] += 1
                    if result["success"]:
                        st["recovered"] += 1
                if result["success"]:
                    st["success"] += 1
                audit_log.append({
                    "benchmark": bench, "task_id": result["task_id"],
                    "step": i, "type": ftype, "justification": why,
                    "detected": detected, "episode_success": result["success"],
                })

    with open(os.path.join(dirs["logs"], "failure_classification_audit.json"), "w",
              encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)

    rows = []
    for ftype in FAILURE_TYPES:
        st = per_type[ftype]
        freq = max(st["freq"], 1)
        rows.append({
            "Failure Type": _LABELS[ftype],
            "Frequency": st["freq"],
            "Detection Rate": f"{100.0 * st['detected'] / freq:.1f}%",
            "Recovery Rate": f"{100.0 * st['recovered'] / max(st['detected'], 1):.1f}%",
            "Final Success": f"{100.0 * st['success'] / freq:.1f}%",
        })
    write_table(rows, os.path.join(dirs["tables"], "failure_analysis"),
                ["Failure Type", "Frequency", "Detection Rate", "Recovery Rate", "Final Success"])
    print("[run_failure] wrote failure analysis table", flush=True)


if __name__ == "__main__":
    main()
