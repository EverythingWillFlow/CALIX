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
        tsrs, rews, lats, calls, toks = [], [], [], [], []
        for seed in seeds:
            print(f"[run_scalability] agents={n} seed={seed}", flush=True)
            model, _ = train_calix(cfg, seed, args.episodes, num_agents=n)
            # measured wall-clock latency and measured LLM usage (no formulas)
            t0 = time.time()
            res = eval_on_benchmarks(model, cfg, seed, args.eval_tasks)
            wall = time.time() - t0
            tsrs.append(float(np.mean([r["tsr"] for r in res.values()])))
            rews.append(float(np.mean([r["avg_reward"] for r in res.values()])))
            n_tasks = sum(len(r["records"]) for r in res.values())
            lats.append(wall / max(n_tasks, 1))
            calls.append(float(np.mean([
                rec["llm_calls"] for r in res.values() for rec in r["records"]])))
            toks.append(float(np.mean([
                rec["llm_prompt_tokens"] + rec["llm_completion_tokens"]
                for r in res.values() for rec in r["records"]])))
        mt, st = mean_std(tsrs)
        mr, sr = mean_std(rews)
        ml, sl = mean_std(lats)
        mc, sc = mean_std(calls)
        rows.append({
            "Agents": n,
            "TSR (%)": fmt_ms(mt, st, 1),
            "Avg. Reward": fmt_ms(mr, sr, 2),
            "Latency (s/task, measured)": fmt_ms(ml, sl, 1),
            "LLM Calls/task (measured)": fmt_ms(mc, sc, 1),
            "Tokens/task (measured)": fmt_ms(float(np.mean(toks)), float(np.std(toks)), 0),
        })
    write_table(rows, os.path.join(dirs["tables"], "scalability"),
                ["Agents", "TSR (%)", "Avg. Reward", "Latency (s/task, measured)",
                 "LLM Calls/task (measured)", "Tokens/task (measured)"])
    print("[run_scalability] wrote scalability table", flush=True)


if __name__ == "__main__":
    main()
