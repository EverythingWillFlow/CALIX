"""End-to-end smoke test against the REAL LLM endpoint.

Runs one CALIX episode and one task for each in-repo baseline, prints the
actual model output, the environment verdict and the token usage, so the
estimated cost of the pilot run can be checked before launching it.

Usage:
    export CALIX_OPENAI_API_KEY=...
    export CALIX_OPENAI_BASE_URL=https://<your-relay>/v1
    python experiments/run_smoke.py --config configs/pilot.yaml --tasks 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calix.baselines import evaluate_baseline  # noqa: E402
from calix.calix import CALIX  # noqa: E402
from calix.config import load_config  # noqa: E402
from common import ensure_output_dirs, load_benchmark, make_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--baselines", nargs="*", default=["gpt4o_direct", "react"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirs = ensure_output_dirs(cfg)
    benchmark = args.benchmark or cfg["benchmarks"]["names"][0]
    _, test = load_benchmark(benchmark)
    print(f"[smoke] benchmark={benchmark} eval_pool={len(test)} tasks")

    env = make_env(test, benchmark, cfg)
    t0 = time.time()
    model = CALIX(cfg, seed=cfg["experiment"]["seeds"][0])
    res = model.evaluate(env, num_tasks=args.tasks, record_provenance=True,
                         provenance_path=os.path.join(
                             dirs["traces"], "smoke_provenance.json"))
    calix_time = time.time() - t0
    print(f"[smoke] CALIX: {json.dumps(res['records'], ensure_ascii=False, indent=2)}")
    print(f"[smoke] CALIX usage: {res['llm_usage']} in {calix_time:.1f}s")

    for base in args.baselines:
        t1 = time.time()
        out = evaluate_baseline(base, make_env(test, benchmark, cfg),
                                args.tasks, cfg["experiment"]["seeds"][0],
                                model=cfg["llm"]["model"], tools=model.tools)
        print(f"[smoke] {base}: {json.dumps(out['records'], ensure_ascii=False, indent=2)}"
              f" usage={out['llm_usage']} in {time.time() - t1:.1f}s")

    total = res["llm_usage"]
    print("\n[smoke] extrapolated pilot cost (1 CALIX task):")
    calls_per_task = total["calls"]
    print(f"  LLM calls / CALIX task = {calls_per_task}")
    # pilot: 40 train episodes (1 task each) + 15 eval tasks + baselines
    episodes = cfg["execution"]["train_episodes"]
    eval_tasks = cfg["execution"]["eval_tasks_per_benchmark"]
    est_calls = (episodes + eval_tasks) * calls_per_task
    print(f"  estimated calls for pilot (train {episodes} + eval {eval_tasks}) "
          f"= {est_calls}")


if __name__ == "__main__":
    main()
