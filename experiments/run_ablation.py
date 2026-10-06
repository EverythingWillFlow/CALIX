"""Component ablation study (paper Table: ablation).

Variants: CALIX (full), w/o Closed-loop, w/o Graph, w/o Behavior, w/o RL.
Each variant is trained and evaluated with the same seeds and protocol as the
main experiment; Avg. TSR / Avg. Reward are reported over the four benchmark
test splits.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import ensure_output_dirs, eval_on_benchmarks, fmt_ms, mean_std, train_calix, write_table
from calix.config import load_config

VARIANTS = {
    "CALIX": {},
    "w/o Closed-loop": {"use_closed_loop": False},
    "w/o Graph": {"use_graph": False},
    "w/o Behavior": {"use_behavior": False},
    "w/o RL": {"use_rl": False},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-tasks", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    seeds = args.seeds or cfg["experiment"]["seeds"]
    eval_tasks = args.eval_tasks or cfg["execution"]["eval_tasks_per_benchmark"]
    benchmarks = cfg["benchmarks"]["names"]

    tsrs: dict[str, list[float]] = {v: [] for v in VARIANTS}
    rewards: dict[str, list[float]] = {v: [] for v in VARIANTS}

    for variant, kwargs in VARIANTS.items():
        for seed in seeds:
            print(f"[run_ablation] {variant} seed={seed}", flush=True)
            model, _ = train_calix(cfg, seed, args.episodes, **kwargs)
            res = eval_on_benchmarks(model, cfg, seed, eval_tasks)
            tsrs[variant].append(float(np.mean([r["tsr"] for r in res.values()])))
            rewards[variant].append(float(np.mean([r["avg_reward"] for r in res.values()])))

    full_m, _ = mean_std(tsrs["CALIX"])
    rows = []
    for variant in VARIANTS:
        m_t, s_t = mean_std(tsrs[variant])
        m_r, s_r = mean_std(rewards[variant])
        rows.append({
            "Method": variant,
            "Avg. TSR (%)": fmt_ms(m_t, s_t, 1),
            "Avg. Reward": fmt_ms(m_r, s_r, 2),
            "TSR Drop": "--" if variant == "CALIX" else f"-{full_m - m_t:.1f}",
        })
    write_table(rows, os.path.join(dirs["tables"], "ablation"),
                ["Method", "Avg. TSR (%)", "Avg. Reward", "TSR Drop"])
    print("[run_ablation] wrote ablation table", flush=True)


if __name__ == "__main__":
    main()
