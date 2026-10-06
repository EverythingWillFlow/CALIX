
from __future__ import annotations

import argparse
import os

import numpy as np

from common import (ROOT, ensure_output_dirs, eval_on_benchmarks, fmt_ms,
                    load_benchmark, make_env, mean_std, save_json, save_model,
                    train_calix, write_table)
from calix.baselines import BASELINE_NAMES, evaluate_baseline
from calix.calix import CALIX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--eval-tasks", type=int, default=None)
    args = ap.parse_args()

    from calix.config import load_config
    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    seeds = args.seeds or cfg["experiment"]["seeds"]
    episodes = args.episodes or cfg["execution"]["train_episodes"]
    eval_tasks = args.eval_tasks or cfg["execution"]["eval_tasks_per_benchmark"]
    benchmarks = cfg["benchmarks"]["names"]

    # per-seed TSRs: method -> benchmark -> [tsr per seed]
    tsr: dict[str, dict[str, list[float]]] = {}
    curves = {"calix": {}, "wo_rl": {}}

    for seed in seeds:
        print(f"[run_main] seed={seed}: training CALIX ({episodes} episodes)", flush=True)
        model, rewards = train_calix(cfg, seed, episodes)
        # learning-curve data: greedy validation performance during training
        # (standard evaluation protocol for RL learning curves)
        curves["calix"][str(seed)] = {
            "checkpoints": [int(ep) for ep, _, _ in model.val_history],
            "val_reward": [float(r) for _, r, _ in model.val_history],
            "val_tsr": [float(t) for _, _, t in model.val_history],
        }
        save_model(model, os.path.join(dirs["models"], f"qnet_seed{seed}.npz"))

        print(f"[run_main] seed={seed}: w/o RL reference run", flush=True)
        wo = CALIX(cfg, seed=seed, use_rl=False)
        wo_env = make_env(model.val_tasks, "agentbench", cfg, seed + 500)
        wo_res = wo.evaluate(wo_env, num_tasks=len(model.val_tasks))
        curves["wo_rl"][str(seed)] = {
            "checkpoints": [int(ep) for ep, _, _ in model.val_history],
            "val_reward": [float(wo_res["avg_reward"])] * len(model.val_history),
            "val_tsr": [float(wo_res["tsr"])] * len(model.val_history),
        }

        print(f"[run_main] seed={seed}: evaluation on {len(benchmarks)} benchmarks", flush=True)
        res = eval_on_benchmarks(model, cfg, seed, eval_tasks)
        tsr.setdefault("CALIX", {})
        for name, r in res.items():
            tsr["CALIX"].setdefault(name, []).append(r["tsr"])

        for base in BASELINE_NAMES:
            for name in benchmarks:
                _, test = load_benchmark(name)
                out = evaluate_baseline(base, test[:eval_tasks], name, seed)
                tsr.setdefault(base, {}).setdefault(name, []).append(out["tsr"])

    save_json(curves, os.path.join(dirs["logs"], "learning_curves.json"))

    # aggregate table
    rows = []
    methods = BASELINE_NAMES + ["CALIX"]
    for method in methods:
        row = {"Method": method}
        bench_means = []
        for name in benchmarks:
            m, s = mean_std(tsr[method][name])
            row[name] = fmt_ms(m, s, 1)
            bench_means.append(m)
        m, s = mean_std([np.mean(tsr[method][b]) for b in benchmarks])
        row["Avg. TSR"] = fmt_ms(float(np.mean(bench_means)), s, 1)
        rows.append(row)
    write_table(rows, os.path.join(dirs["tables"], "overall_results"),
                ["Method"] + benchmarks + ["Avg. TSR"])
    print("[run_main] wrote", os.path.join(dirs["tables"], "overall_results.csv"), flush=True)


if __name__ == "__main__":
    main()
