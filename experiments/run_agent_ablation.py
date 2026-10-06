"""Leave-one-agent-out analysis (paper Table: agent-level contribution)."""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import ensure_output_dirs, eval_on_benchmarks, fmt_ms, mean_std, train_calix, write_table
from calix.config import load_config


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
    roles = cfg["agents"]["roles"]

    tsrs: dict[str, list[float]] = {"CALIX": []}
    for role in roles:
        tsrs[f"w/o {role}"] = []

    for config_name, removed in [("CALIX", None)] + [(f"w/o {r}", r) for r in roles]:
        for seed in seeds:
            print(f"[run_agent_ablation] {config_name} seed={seed}", flush=True)
            model, _ = train_calix(cfg, seed, args.episodes, removed_role=removed)
            res = eval_on_benchmarks(model, cfg, seed, eval_tasks)
            tsrs[config_name].append(float(np.mean([r["tsr"] for r in res.values()])))

    full_m, _ = mean_std(tsrs["CALIX"])
    rows = []
    for config_name in tsrs:
        m, s = mean_std(tsrs[config_name])
        rows.append({
            "Configuration": config_name,
            "Avg. TSR (%)": fmt_ms(m, s, 1),
            "TSR Drop": "--" if config_name == "CALIX" else f"-{full_m - m:.1f}",
        })
    write_table(rows, os.path.join(dirs["tables"], "agent_ablation"),
                ["Configuration", "Avg. TSR (%)", "TSR Drop"])
    print("[run_agent_ablation] wrote agent ablation table", flush=True)


if __name__ == "__main__":
    main()
