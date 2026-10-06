"""Hyperparameter sensitivity analysis (paper Figure: lr / gamma / tau / K)."""
from __future__ import annotations

import argparse
import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ensure_output_dirs, eval_on_benchmarks, train_calix
from calix.config import load_config

GRIDS = {
    "learning_rate": ("qnetwork.lr", [1e-4, 5e-4, 1e-3, 5e-3], "Learning rate"),
    "gamma": ("qnetwork.gamma", [0.8, 0.9, 0.95, 0.99], "Discount factor $\\gamma$"),
    "tau_sensitivity": ("llm.selection_tau", [0.25, 0.5, 1.0, 2.0], "Selection temperature $\\tau$"),
    "k_sensitivity": ("llm.num_candidates", [1, 3, 5, 7], "Number of candidates $K$"),
}


def set_param(cfg: dict, dotted: str, value) -> dict:
    cfg = copy.deepcopy(cfg)
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-tasks", type=int, default=30)
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    dirs = ensure_output_dirs(base_cfg)
    seeds = args.seeds or base_cfg["experiment"]["seeds"][:2]
    benchmarks = base_cfg["benchmarks"]["names"]

    for fig_name, (dotted, values, xlabel) in GRIDS.items():
        tsr_means, tsr_stds, rew_means = [], [], []
        for v in values:
            cfg_v = set_param(base_cfg, dotted, v)
            per_seed_tsr, per_seed_rew = [], []
            for seed in seeds:
                print(f"[run_sensitivity] {dotted}={v} seed={seed}", flush=True)
                model, _ = train_calix(cfg_v, seed, args.episodes)
                res = eval_on_benchmarks(model, cfg_v, seed, args.eval_tasks)
                per_seed_tsr.append(float(np.mean([r["tsr"] for r in res.values()])))
                per_seed_rew.append(float(np.mean([r["avg_reward"] for r in res.values()])))
            tsr_means.append(float(np.mean(per_seed_tsr)))
            tsr_stds.append(float(np.std(per_seed_tsr)))
            rew_means.append(float(np.mean(per_seed_rew)))

        fig, ax1 = plt.subplots(figsize=(5.2, 4.0), dpi=150)
        ax2 = ax1.twinx()
        x = np.arange(len(values))
        labels = [f"{v:g}" for v in values]
        ax1.errorbar(x, tsr_means, yerr=tsr_stds, marker="o", color="#d62728",
                     linewidth=1.8, capsize=4, label="TSR")
        ax2.plot(x, rew_means, marker="s", color="#1f77b4", linewidth=1.6,
                 linestyle="--", label="Avg. reward")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("TSR (%)", color="#d62728")
        ax2.set_ylabel("Avg. reward", color="#1f77b4")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(dirs["figures"], f"{fig_name}.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"[run_sensitivity] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
