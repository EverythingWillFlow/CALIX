"""RL learning-curve figure (paper Figure: learning dynamics over 300 episodes).

Reads output/logs/learning_curves.json produced by run_main.py and plots the
mean cumulative reward +/- one standard deviation across the five seeds for
CALIX and the w/o RL variant.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ensure_output_dirs
from calix.config import load_config


def smooth(x: np.ndarray, w: int = 9) -> np.ndarray:
    if w <= 1 or len(x) < 3:
        return x
    pad = w // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    path = os.path.join(dirs["logs"], "learning_curves.json")
    with open(path, "r", encoding="utf-8") as f:
        curves = json.load(f)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    n_seeds = len(curves["calix"])
    for key, label, color in [("calix", "CALIX", "#d62728"),
                              ("wo_rl", "w/o RL (heuristic selection)", "#1f77b4")]:
        runs = curves[key].values()
        episodes = np.asarray(next(iter(runs))["checkpoints"], dtype=float)
        arr = np.stack([np.asarray(r["val_reward"], dtype=float) for r in curves[key].values()])
        mean = smooth(arr.mean(axis=0), w=3)
        std = smooth(arr.std(axis=0), w=3)
        ax.plot(episodes, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Validation reward (greedy)")
    ax.set_title(f"Learning dynamics of CALIX vs. w/o RL (mean ± SD, {n_seeds} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(dirs["figures"], "reward_curve.png")
    fig.savefig(out)
    print("[run_rl_curve] wrote", out, flush=True)


if __name__ == "__main__":
    main()
