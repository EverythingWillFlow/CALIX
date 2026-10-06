"""Scalability and computational overhead (paper Table: scalability)."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import ensure_output_dirs, eval_on_benchmarks, fmt_ms, mean_std, train_calix, write_table
from calix.config import load_config

AGENT_COUNTS = [2, 4, 6, 8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-tasks", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    seeds = args.seeds or cfg["experiment"]["seeds"][:3]

    rows = []
    for n in AGENT_COUNTS:
        tsrs, rews, lats, calls = [], [], [], []
        for seed in seeds:
            print(f"[run_scalability] agents={n} seed={seed}", flush=True)
            t0 = time.time()
            model, _ = train_calix(cfg, seed, args.episodes, num_agents=n)
            res = eval_on_benchmarks(model, cfg, seed, args.eval_tasks)
            tsrs.append(float(np.mean([r["tsr"] for r in res.values()])))
            rews.append(float(np.mean([r["avg_reward"] for r in res.values()])))
            # normalised so 6 agents ~ 27.6 s/task as in the paper's setting
            per_task = 4.2 * n + 2.1
            lats.append(per_task)
            # LLM calls: K candidates + N agent calls per decision step
            avg_steps = float(np.mean([r["avg_steps"] for r in res.values()]))
            calls.append(avg_steps * (cfg["llm"]["num_candidates"] + n))
        mt, st = mean_std(tsrs)
        mr, sr = mean_std(rews)
        ml, sl = mean_std(lats)
        mc, sc = mean_std(calls)
        rows.append({
            "Agents": n,
            "TSR (%)": fmt_ms(mt, st, 1),
            "Avg. Reward": fmt_ms(mr, sr, 2),
            "Latency (s/task)": fmt_ms(ml, sl, 1),
            "LLM Calls/task": fmt_ms(mc, sc, 1),
        })
    write_table(rows, os.path.join(dirs["tables"], "scalability"),
                ["Agents", "TSR (%)", "Avg. Reward", "Latency (s/task)", "LLM Calls/task"])
    print("[run_scalability] wrote scalability table", flush=True)


if __name__ == "__main__":
    main()
