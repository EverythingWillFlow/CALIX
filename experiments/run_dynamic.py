"""Robustness to dynamic environments (paper Table: dynamic robustness)."""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import ensure_output_dirs, eval_on_benchmarks, fmt_ms, mean_std, train_calix, write_table
from calix.config import load_config

LEVELS = [("Static", 0.0), ("Low", 0.10), ("Moderate", 0.25), ("High", 0.50), ("Severe", 0.75)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-tasks", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    seeds = args.seeds or cfg["experiment"]["seeds"][:3]
    eval_tasks = args.eval_tasks or cfg["execution"]["eval_tasks_per_benchmark"]

    stats: dict[tuple, dict[str, list[float]]] = {}
    for level, rate in LEVELS:
        for variant, kwargs in [("CALIX", {}), ("w/o CL", {"use_closed_loop": False})]:
            tsrs, rews = [], []
            for seed in seeds:
                print(f"[run_dynamic] {level} ({rate:.0%}) {variant} seed={seed}", flush=True)
                model, _ = train_calix(cfg, seed, args.episodes, change_rate=rate, **kwargs)
                res = eval_on_benchmarks(model, cfg, seed, eval_tasks, change_rate=rate)
                tsrs.append(float(np.mean([r["tsr"] for r in res.values()])))
                rews.append(float(np.mean([r["avg_reward"] for r in res.values()])))
            stats[(level, variant)] = {"tsr": tsrs, "rew": rews}

    rows = []
    for level, rate in LEVELS:
        ct, cr = mean_std(stats[(level, "CALIX")]["tsr"]), mean_std(stats[(level, "CALIX")]["rew"])
        wt, wr = mean_std(stats[(level, "w/o CL")]["tsr"]), mean_std(stats[(level, "w/o CL")]["rew"])
        rows.append({
            "Dynamic Level": level,
            "Change Rate": f"{rate:.0%}",
            "CALIX TSR": fmt_ms(*ct, 1),
            "CALIX Reward": fmt_ms(*cr, 2),
            "w/o CL TSR": fmt_ms(*wt, 1),
            "w/o CL Reward": fmt_ms(*wr, 2),
        })
    write_table(rows, os.path.join(dirs["tables"], "dynamic_robustness"),
                ["Dynamic Level", "Change Rate", "CALIX TSR", "CALIX Reward",
                 "w/o CL TSR", "w/o CL Reward"])
    print("[run_dynamic] wrote dynamic robustness table", flush=True)


if __name__ == "__main__":
    main()
