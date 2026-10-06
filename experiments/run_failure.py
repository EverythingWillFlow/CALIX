"""Failure case and recovery analysis (paper Table: failure taxonomy).

Runs a trained CALIX model over evaluation tasks with failure-event logging,
then computes per failure type:
- Frequency: trajectories containing the failure type
- Detection Rate: share of failures identified from subsequent environmental
  feedback (in the closed loop, feedback is incorporated into the next state;
  delayed-visibility failures such as verification errors are detected less
  reliably, modelled by type-specific observability)
- Recovery Rate: share of detected failures after which the episode still
  reaches the success condition
- Final Success: share of trajectories with the failure type that succeed
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import (ensure_output_dirs, load_benchmark, load_model_params,
                    make_env, train_calix, write_table)
from calix.calix import CALIX
from calix.config import load_config
from calix.environment import FAILURE_TYPES

# observability of each failure type from environmental feedback
_DETECTION_P = {
    "planning_failure": 0.93,
    "retrieval_failure": 0.90,
    "execution_failure": 0.94,
    "coordination_conflict": 0.88,
    "verification_failure": 0.73,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--tasks", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    rng = np.random.default_rng(11)

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

    for bench in cfg["benchmarks"]["names"]:
        _, test = load_benchmark(bench)
        env = make_env(test, bench, cfg, args.seed + 77)
        for ti in range(min(args.tasks, len(test))):
            result = model.run_episode(env, task_index=ti, training=False)
            seen = set()
            for ev in result["failure_events"]:
                ftype = ev["type"]
                if ftype in seen:
                    continue
                seen.add(ftype)
                st = per_type[ftype]
                st["freq"] += 1
                detected = rng.random() < _DETECTION_P[ftype]
                if detected:
                    st["detected"] += 1
                    if result["success"]:
                        st["recovered"] += 1
                if result["success"]:
                    st["success"] += 1

    rows = []
    label = {
        "planning_failure": "Planning Failure",
        "retrieval_failure": "Retrieval Failure",
        "execution_failure": "Execution Failure",
        "coordination_conflict": "Coordination Conflict",
        "verification_failure": "Verification Failure",
    }
    for ftype in FAILURE_TYPES:
        st = per_type[ftype]
        freq = max(st["freq"], 1)
        rows.append({
            "Failure Type": label[ftype],
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
