"""Shared helpers for CALIX experiment scripts."""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calix.config import load_config  # noqa: E402
from calix.environment import BenchmarkEnv  # noqa: E402
from calix.calix import CALIX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "data", "benchmarks")


def ensure_output_dirs(cfg: dict) -> dict:
    out = os.path.join(ROOT, cfg["experiment"]["output_dir"])
    dirs = {
        "root": out,
        "tables": os.path.join(out, "tables"),
        "figures": os.path.join(out, "figures"),
        "traces": os.path.join(out, "traces"),
        "logs": os.path.join(out, "logs"),
        "models": os.path.join(out, "models"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def load_benchmark(name: str) -> tuple[list[dict], list[dict]]:
    tasks = BenchmarkEnv.load_tasks(os.path.join(BENCH_DIR, f"{name}.json"))
    train = [t for t in tasks if t["split"] == "train"]
    test = [t for t in tasks if t["split"] == "test"]
    return train, test


def make_env(tasks: list[dict], benchmark: str, cfg: dict, seed: int,
             change_rate: float = 0.0) -> BenchmarkEnv:
    return BenchmarkEnv(
        tasks, benchmark,
        max_steps=cfg["execution"]["max_rounds"],
        change_rate=change_rate,
        num_strategies=cfg["openai"]["num_strategies"],
        seed=seed,
    )


def train_calix(cfg: dict, seed: int, episodes: int, benchmark: str = "agentbench",
                change_rate: float = 0.0, val_fraction: float = 0.15,
                val_every: int = 25, **calix_kwargs) -> tuple[CALIX, list[float]]:



    train, _ = load_benchmark(benchmark)
    train = sorted(train, key=lambda t: t.get("required_progress", 1.0))
    n_val = max(5, int(len(train) * val_fraction)) if val_fraction > 0 else 0
    if n_val:
        stride = max(1, len(train) // n_val)
        val_idx = set(range(0, len(train), stride))
        val_tasks = [t for i, t in enumerate(train) if i in val_idx][:n_val]
        core_tasks = [t for i, t in enumerate(train) if i not in val_idx]
    else:
        val_tasks, core_tasks = [], train
    env = make_env(core_tasks, benchmark, cfg, seed, change_rate=change_rate)
    model = CALIX(cfg, seed=seed, **calix_kwargs)
    rewards = []
    best_score, best_params = -np.inf, None
    model.val_history = []               # (episode, val avg_reward, val TSR)
    model.val_tasks = val_tasks
    for ep in range(1, episodes + 1):
        rewards.append(model.run_episode(env, training=True)["cum_reward"])
        if n_val and ep % val_every == 0:
            val_env = make_env(val_tasks, benchmark, cfg, seed + 500)
            val_res = model.evaluate(val_env, num_tasks=len(val_tasks))
            score = val_res["avg_reward"]
            model.val_history.append((ep, score, val_res["tsr"]))
            if score > best_score:
                best_score = score
                best_params = model.q_net.get_params()
    if best_params is not None:
        model.q_net.set_params(best_params)
        model.target_q_net.copy_from(model.q_net)
    return model, rewards


def eval_on_benchmarks(model: CALIX, cfg: dict, seed: int, num_tasks: int,
                       change_rate: float = 0.0,
                       benchmarks: list[str] | None = None) -> dict:
    """Evaluate a trained model on each benchmark's held-out test split."""
    results = {}
    for name in (benchmarks or cfg["benchmarks"]["names"]):
        _, test = load_benchmark(name)
        env = make_env(test, name, cfg, seed + 1000, change_rate=change_rate)
        res = model.evaluate(env, num_tasks=min(num_tasks, len(test)))
        results[name] = res
    return results


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def fmt_ms(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def write_table(rows: list[dict], path_base: str, fieldnames: list[str] | None = None) -> None:
    """Write results as CSV and as a Markdown table."""
    if not rows:
        return
    fields = fieldnames or list(rows[0].keys())
    with open(path_base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(path_base + ".md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |\n")


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_model(model: CALIX, path: str) -> None:
    np.savez(path, **{f"q_{k}": v for k, v in model.q_net.get_params().items()})


def load_model_params(model: CALIX, path: str) -> None:
    data = np.load(path)
    params = model.q_net.get_params()
    for k in params:
        params[k] = data[f"q_{k}"]
    model.q_net.set_params(params)
    model.target_q_net.copy_from(model.q_net)
