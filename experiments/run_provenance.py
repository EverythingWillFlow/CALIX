"""Decision-provenance evaluation (paper Table: PC / DF / TS).

Loads the Q-network trained in run_main.py (seed 2025), runs repeated
provenance-recorded evaluations on the same task, and computes:
- Path Completeness (PC)
- Decision Fidelity (DF)
- Trace Stability (TS)
Also saves a representative provenance trace to output/traces/.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import ensure_output_dirs, load_benchmark, load_model_params, make_env, write_table
from calix.calix import CALIX
from calix.config import load_config
from calix.provenance import (ProvenanceRecorder, decision_fidelity,
                              path_completeness, trace_stability)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=150,
                    help="training episodes if no saved model is found")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)

    model_path = os.path.join(dirs["models"], "qnet_seed2025.npz")
    _, test = load_benchmark("agentbench")

    all_records: list[list[dict]] = []
    traces_for_ts: list[list[dict]] = []
    saved = False

    for rep in range(args.repeats):
        print(f"[run_provenance] repeat {rep + 1}/{args.repeats}", flush=True)
        if os.path.exists(model_path):
            from common import train_calix  # local import to avoid cycle
            model = CALIX(cfg, seed=2025)
            load_model_params(model, model_path)
        else:
            from common import train_calix
            model, _ = train_calix(cfg, 2025, args.episodes)
        env = make_env(test, "agentbench", cfg, 2025 + 1000)
        rep_records: list[dict] = []
        rep_traces: list[dict] = []
        for ti in range(args.tasks):
            recorder = ProvenanceRecorder()
            model.run_episode(env, task_index=ti, training=False, recorder=recorder)
            rep_records.extend(recorder.records)
            rep_traces.append(recorder.records)
            if not saved:
                recorder.save(os.path.join(dirs["traces"], "provenance_trace_example.json"))
                saved = True
        all_records.append(rep_records)
        traces_for_ts.extend(rep_traces)   # each task's trace is one trace

    flat = [r for rep in all_records for r in rep]
    pc = path_completeness(flat)
    df = decision_fidelity(flat)
    ts = trace_stability(traces_for_ts)

    rows = [
        {"Metric": "Path Completeness (PC)", "Value (%)": f"{pc:.1f}"},
        {"Metric": "Decision Fidelity (DF)", "Value (%)": f"{df:.1f}"},
        {"Metric": "Trace Stability (TS)", "Value (%)": f"{ts:.1f}"},
    ]
    write_table(rows, os.path.join(dirs["tables"], "provenance"), ["Metric", "Value (%)"])
    print(f"[run_provenance] PC={pc:.1f} DF={df:.1f} TS={ts:.1f}", flush=True)


if __name__ == "__main__":
    main()
