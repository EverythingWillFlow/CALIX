"""
Run CALIX and all baselines on the official benchmarks, calling the
official evaluators and writing real TSR values to output/tables.

Pinned versions (must match README):
    - AgentBench : THUDM/AgentBench tag v0.2
    - GAIA       : gaia-benchmark/GAIA revision
                   682dd723ee1e1697e00360edccf2366dc8418dd9
    - WebArena   : web-arena-x/webarena tag v0.2.0
    - SWE-bench  : swebench==4.1.0

Usage:
    python experiments/run_real_benchmarks.py \
        --benchmark agentbench gaia webarena swebench \
        --seeds 2025 2026 2027 2028 2029
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from calix.environment import build_environment
from calix.baselines import build_baseline
from calix.calix import CALIX


OUT_DIR = Path("output/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("data/benchmarks")

# Official data locations after data/prepare_data.py
DATASETS = {
    "agentbench": Path("third_party/AgentBench"),
    "gaia": DATA_DIR / "gaia",
    "webarena": Path("third_party/webarena"),
    "swebench": DATA_DIR / "swebench_test.jsonl",
}

BASELINES = [
    "gpt4o_direct",
    "react",
    "camel",
    "autogen",
    "agentverse",
    "metagpt",
    "chatdev",
    "sweagent",
    "openhands",
]


def record_environment(benchmark: str):
    """Record the resolved revision for the benchmark into the log."""
    info = {"benchmark": benchmark}
    if benchmark == "agentbench":
        repo = Path("third_party/AgentBench")
        info["tag"] = "v0.2"
        info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    elif benchmark == "webarena":
        repo = Path("third_party/webarena")
        info["tag"] = "v0.2.0"
        info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    elif benchmark == "gaia":
        info["revision"] = "682dd723ee1e1697e00360edccf2366dc8418dd9"
    elif benchmark == "swebench":
        info["version"] = "4.1.0"
    return info


def load_tasks(benchmark: str):
    path = DATASETS[benchmark]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data for {benchmark} at {path}. "
            f"Run data/prepare_data.py --benchmark {benchmark} first."
        )
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
        return [json.loads(f.read_text()) for f in files]
    tasks = []
    with path.open() as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def evaluate_one(method_name: str, benchmark: str, task: dict, seed: int):
    env = build_environment(benchmark)

    if method_name == "calix":
        agent = CALIX(seed=seed)
    else:
        agent = build_baseline(method_name, seed=seed)

    observation = env.reset(task)
    trajectory = agent.run(observation, env, task)

    # Official evaluator
    result = env.evaluate(trajectory)
    return {
        "method": method_name,
        "benchmark": benchmark,
        "task_id": task.get("task_id") or task.get("instance_id"),
        "seed": seed,
        "success": bool(result["success"]),
        "raw": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        nargs="+",
        default=["agentbench", "gaia", "webarena", "swebench"],
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[2025, 2026, 2027, 2028, 2029])
    parser.add_argument("--methods", nargs="+", default=["calix"] + BASELINES)
    args = parser.parse_args()

    env_log = {b: record_environment(b) for b in args.benchmark}
    print("[environment]", json.dumps(env_log, indent=2))

    rows = []
    for benchmark in args.benchmark:
        tasks = load_tasks(benchmark)
        print(f"[{benchmark}] {len(tasks)} tasks")

        for method in args.methods:
            for seed in args.seeds:
                successes = 0
                for task in tasks:
                    try:
                        record = evaluate_one(method, benchmark, task, seed)
                    except Exception as e:
                        print(f"[error] {method} {benchmark} "
                              f"{task.get('task_id')}: {e}")
                        record = {
                            "method": method,
                            "benchmark": benchmark,
                            "task_id": task.get("task_id"),
                            "seed": seed,
                            "success": False,
                            "raw": {"error": str(e)},
                        }
                    rows.append(record)
                    successes += int(record["success"])

                tsr = successes / max(len(tasks), 1)
                print(f"[{benchmark}] {method} seed={seed} TSR={tsr:.4f}")

    df = pd.DataFrame(rows)
    per_run = df.groupby(["benchmark", "method", "seed"])["success"] \
                .mean().reset_index()
    summary = per_run.groupby(["benchmark", "method"])["success"] \
                     .agg(["mean", "std"]).reset_index()
    summary.to_csv(OUT_DIR / "real_results.csv", index=False)
    print(f"Saved real results to {OUT_DIR / 'real_results.csv'}")


if __name__ == "__main__":
    main()