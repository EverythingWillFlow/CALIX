"""One-command reproduction of every CALIX experiment.

Usage:
    python experiments/run_all.py [--config configs/default.yaml] [--fast]

--fast reduces seeds and episode counts for a quick end-to-end pipeline check
(approx. 15-25 min on CPU); the default full run follows the paper protocol
(five seeds) and takes substantially longer.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(os.path.abspath(__file__))


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-u", os.path.join(EXP, script)] + extra
    print(f"\n[run_all] ===== {' '.join(cmd)} =====", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg_args = ["--config", args.config] if args.config else []
    if args.fast:
        # pipeline-validation budgets (~3-4 h on a typical CPU); every table
        # and figure is produced.  Use the full branch below for the paper
        # protocol (5 seeds, longer training).
        run("run_main.py", cfg_args + ["--episodes", "150", "--seeds", "2025", "2026", "--eval-tasks", "30"])
        run("run_ablation.py", cfg_args + ["--episodes", "100", "--seeds", "2025", "2026", "--eval-tasks", "30"])
        run("run_agent_ablation.py", cfg_args + ["--episodes", "80", "--seeds", "2025", "--eval-tasks", "30"])
        run("run_sensitivity.py", cfg_args + ["--episodes", "60", "--seeds", "2025", "--eval-tasks", "20"])
        run("run_dynamic.py", cfg_args + ["--episodes", "60", "--seeds", "2025", "--eval-tasks", "20"])
        run("run_scalability.py", cfg_args + ["--episodes", "60", "--seeds", "2025", "--eval-tasks", "20"])
    else:
        run("run_main.py", cfg_args)
        run("run_ablation.py", cfg_args)
        run("run_agent_ablation.py", cfg_args)
        run("run_sensitivity.py", cfg_args)
        run("run_dynamic.py", cfg_args)
        run("run_scalability.py", cfg_args)
    # model-dependent / figure steps
    run("run_rl_curve.py", cfg_args)
    run("run_provenance.py", cfg_args)
    run("run_failure.py", cfg_args)
    print("\n[run_all] all experiments finished; see output/", flush=True)


if __name__ == "__main__":
    main()
