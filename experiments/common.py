"""Shared helpers for CALIX experiment scripts (real-benchmark pipeline).

Task pools
----------
Benchmark task files in ``data/benchmarks/`` are JSON/JSONL lists.  A task may
carry an explicit ``pool`` field (``"train"`` / ``"eval"``); when it is absent
the split is derived deterministically from the task id (SHA-256 modulo 5,
first 3/5 train, rest eval) so every method sees the same disjoint pools on
every machine and every run.

Task-level records
------------------
Every evaluation appends one JSON object per (method, benchmark, seed, task)
to ``output/traces/task_records.jsonl``.  Aggregate tables are always
computed from these records, never edited by hand.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calix.config import load_config  # noqa: E402
from calix.environment import build_environment, load_task_file, RealEnvironment  # noqa: E402
from calix.calix import CALIX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "data", "benchmarks")
RECORDS_PATH = None  # resolved lazily from the active config's output dir
_TASK_CACHE: dict[str, dict] = {}


def _task_context(benchmark: str, task_id: str) -> tuple[str, str]:
    """Return the benchmark task's observation and public split."""
    key = f"{benchmark}:{task_id}"
    if key not in _TASK_CACHE:
        path = os.path.join(BENCH_DIR, f"{benchmark}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    task = json.loads(line)
                    _TASK_CACHE[f"{benchmark}:{task.get('task_id')}"] = task
    task = _TASK_CACHE.get(key, {})
    # Keep the published 60/40 train/test ratio stable across runs.
    digest = hashlib.sha256(f"{benchmark}:{task_id}:calix-split".encode()).digest()
    split = "test" if int.from_bytes(digest[:4], "little") % 5 < 2 else "train"
    return task.get("initial_observation", ""), split


def _record_payload(method: str, benchmark: str, seed: int, record: dict) -> dict:
    """Normalize one result to the published task-record schema."""
    task_id = record.get("task_id", "")
    observation, split = _task_context(benchmark, task_id)
    payload = dict(record)
    payload.setdefault("initial_observation", observation)
    payload.setdefault("metadata", {"split": split})
    payload["metadata"] = {"split": (payload["metadata"] or {}).get("split", split)}
    payload.setdefault("answer", "")
    payload.setdefault("shell_used", False)
    payload.setdefault("shell_available", None)
    return {"method": method, "benchmark": benchmark, "seed": seed, **payload}


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
    write_manifest(cfg, dirs)
    return dirs


# --------------------------------------------------------------------------- #
# run manifest -- the process data a reviewer needs to reproduce a table
# --------------------------------------------------------------------------- #
_MANIFEST_WRITTEN = {"path": None}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> str:
    """SHA-256 over the repository's own Python sources.

    Two runs that report the same fingerprint executed the same code.  This is
    what lets a reviewer tie a number in the paper to a commit rather than to a
    claim.
    """
    h = hashlib.sha256()
    for base in ("calix", "experiments"):
        root = os.path.join(ROOT, base)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
                h.update(rel.encode())
                h.update(_sha256_file(full).encode())
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", ROOT, *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def git_state() -> dict:
    sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    return {
        "commit": sha[:12] or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty_files": len([l for l in dirty.splitlines() if l.strip()]),
        "clean": not dirty.strip(),
    }


def write_manifest(cfg: dict, dirs: dict, extra: dict | None = None) -> str:
    """Record the exact conditions a run started under.

    Written once per process to ``<output>/manifests/``.  The API key is never
    stored, only a short digest, so a manifest can be published without leaking
    credentials while still proving which account produced the run.
    """
    if _MANIFEST_WRITTEN["path"] and extra is None:
        return _MANIFEST_WRITTEN["path"]

    key = os.environ.get("CALIX_OPENAI_API_KEY", "")
    cfg_path = cfg.get("_path") or os.environ.get("CALIX_CONFIG", "")
    llm = cfg.get("llm", {}) or {}
    manifest = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "script": os.path.basename(sys.argv[0]) if sys.argv else None,
        "argv": [a for a in sys.argv] if extra is None else None,
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "git": git_state(),
        "code_fingerprint": code_fingerprint(),
        "config": {
            "path": cfg_path or None,
            "sha256": (_sha256_file(cfg_path)
                       if cfg_path and os.path.exists(cfg_path) else None),
            "experiment": cfg.get("experiment", {}),
            "execution": cfg.get("execution", {}),
            "benchmarks": cfg.get("benchmarks", {}),
        },
        "llm": {
            "base_url": (os.environ.get("CALIX_OPENAI_BASE_URL")
                         or llm.get("base_url")),
            "model": (os.environ.get("CALIX_MODEL") or llm.get("model")),
            "embedding_model": (os.environ.get("CALIX_EMBED_MODEL")
                                or llm.get("embed_model")),
            "key_sha256_12": (hashlib.sha256(key.encode()).hexdigest()[:12]
                              if key else None),
        },
        "records": {
            "path": os.path.relpath(
                os.path.join(dirs["traces"], "task_records.jsonl"), ROOT),
        },
    }
    if extra:
        manifest["extra"] = extra

    mdir = os.path.join(dirs["root"], "manifests")
    os.makedirs(mdir, exist_ok=True)
    stamp = manifest["timestamp_utc"].replace(":", "").replace("+", "Z")
    name = f"{stamp}_{manifest['script'] or 'run'}.json"
    path = os.path.join(mdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    latest = os.path.join(mdir, "latest.json")
    with open(latest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    if extra is None:
        _MANIFEST_WRITTEN["path"] = path
    return path


def _task_pool(task: dict) -> str:
    if task.get("pool") in ("train", "eval"):
        return task["pool"]
    digest = hashlib.sha256(f"{task['task_id']}:calix-split".encode()).digest()
    return "train" if int.from_bytes(digest[:4], "little") % 5 < 3 else "eval"


def load_benchmark(name: str) -> tuple[list[dict], list[dict]]:
    """Return (train_pool, eval_pool) for a benchmark.

    A benchmark made of several named subsets (AgentBench: ``dbbench`` and
    ``knowledgegraph``) is split **within each subset** and then interleaved, so
    that any prefix of the eval pool covers every subset.  Without this a
    2-task run would only ever sample dbbench, because ids sort that way.
    """
    tasks = load_task_file(os.path.join(BENCH_DIR, f"{name}.json"))
    tasks = sorted(tasks, key=lambda t: t["task_id"])

    groups: dict[str, list[dict]] = {}
    for t in tasks:
        groups.setdefault((t.get("metadata") or {}).get("subset") or "", []).append(t)
    if len(groups) > 1:
        names = sorted(groups)
        pools = {k: ([t for t in v if _task_pool(t) == "train"],
                     [t for t in v if _task_pool(t) == "eval"])
                 for k, v in groups.items()}
        train, test = [], []
        for i in range(max(len(p[0]) for p in pools.values())):
            for k in names:
                if i < len(pools[k][0]):
                    train.append(pools[k][0][i])
        for i in range(max(len(p[1]) for p in pools.values())):
            for k in names:
                if i < len(pools[k][1]):
                    test.append(pools[k][1][i])
        return train, test

    train = [t for t in tasks if _task_pool(t) == "train"]
    test = [t for t in tasks if _task_pool(t) == "eval"]
    return train, test


def make_env(tasks: list[dict], benchmark: str, cfg: dict) -> RealEnvironment:
    """Build the benchmark environment.

    For the Docker-based benchmarks the runtime address comes from
    ``benchmarks.executors.<name>`` in the config (or from
    ``$CALIX_EXECUTOR_<NAME>``), which is how a run is pointed at a remote
    runtime, e.g. a Colab VM exposing ``tools/executor_bridge.py``.
    """
    executors = (cfg.get("benchmarks", {}) or {}).get("executors", {}) or {}
    return build_environment(
        benchmark, tasks, max_steps=cfg["execution"]["max_rounds"],
        executor_url=executors.get(benchmark))


def append_records(cfg: dict, method: str, benchmark: str, seed: int,
                   records: list[dict]) -> str:
    """Append task-level records; returns the records file path."""
    dirs = ensure_output_dirs(cfg)
    path = os.path.join(dirs["traces"], "task_records.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_payload(method, benchmark, seed, r),
                               ensure_ascii=False) + "\n")
    return path


def make_record_appender(cfg: dict, method: str, benchmark: str, seed: int):
    """Return a thread-safe callable that persists one record *immediately*.

    ``append_records`` writes a whole cell at the end.  That was fine until a
    350-task cell was killed mid-way: every task it had already graded was lost
    with it, because the rows only existed in memory.  Long runs must be able
    to die at any point without losing finished work -- ``token burned, no row
    written`` is exactly the loss a resumable benchmark cannot afford.

    Every call appends a single line and fsyncs it before returning, so a
    SIGKILL at the worst possible instant loses at most the task in flight.
    """
    dirs = ensure_output_dirs(cfg)
    path = os.path.join(dirs["traces"], "task_records.jsonl")
    lock = threading.Lock()

    def append(record: dict) -> None:
        line = json.dumps(_record_payload(method, benchmark, seed, record),
                          ensure_ascii=False) + "\n"
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    return append


def train_calix(cfg: dict, seed: int, episodes: int, benchmark: str | None = None,
                val_fraction: float = 0.15,
                val_every: int = 25, **calix_kwargs) -> tuple[CALIX, list[float]]:
    benchmark = benchmark or cfg["benchmarks"]["names"][0]
    train, _ = load_benchmark(benchmark)
    if not train:
        raise RuntimeError(
            f"No training tasks for benchmark '{benchmark}'. "
            "Check data/benchmarks and the pool split.")
    train = sorted(train, key=lambda t: t["task_id"])
    n_val = max(3, int(len(train) * val_fraction)) if val_fraction > 0 else 0
    if n_val:
        stride = max(1, len(train) // n_val)
        val_idx = set(range(0, len(train), stride))
        val_tasks = [t for i, t in enumerate(train) if i in val_idx][:n_val]
        core_tasks = [t for i, t in enumerate(train) if i not in val_idx]
    else:
        val_tasks, core_tasks = [], train

    env = make_env(core_tasks, benchmark, cfg)
    model = CALIX(cfg, seed=seed, **calix_kwargs)
    rewards: list[float] = []
    best_score, best_params = -np.inf, None
    model.val_history = []               # (episode, val avg_reward, val TSR)
    model.val_tasks = val_tasks
    for ep in range(1, episodes + 1):
        task_index = (ep - 1) % max(1, len(core_tasks))
        result = model.run_episode(env, task_index=task_index, training=True)
        rewards.append(result["cum_reward"])
        if n_val and ep % val_every == 0:
            val_env = make_env(val_tasks, benchmark, cfg)
            val_res = model.evaluate(val_env, num_tasks=len(val_tasks))
            score = val_res["avg_reward"]
            model.val_history.append((ep, score, val_res["tsr"]))
            if score > best_score:
                best_score = score
                best_params = model.q_net.get_params()
        if ep % 5 == 0:
            print(f"    [train] episode {ep}/{episodes} "
                  f"cum_reward={rewards[-1]:.2f}", flush=True)
    if best_params is not None:
        model.q_net.set_params(best_params)
        model.target_q_net.copy_from(model.q_net)
    return model, rewards


def eval_on_benchmarks(model: CALIX, cfg: dict, seed: int, num_tasks: int,
                       method: str = "CALIX",
                       benchmarks: list[str] | None = None) -> dict:
    """Evaluate a trained model on each benchmark's held-out eval pool and
    append task-level records."""
    results = {}
    for name in (benchmarks or cfg["benchmarks"]["names"]):
        _, test = load_benchmark(name)
        env = make_env(test, name, cfg)
        res = model.evaluate(env, num_tasks=min(num_tasks, len(test)))
        append_records(cfg, method, name, seed, res["records"])
        results[name] = res
    return results


def mean_std(values: list[float], label: str = "values") -> tuple[float, float]:
    """Mean and standard deviation across independent runs (seeds).

    A deviation across seeds means nothing with fewer than two seeds, and
    printing ``0.00`` there would tell a reader the number is more stable than
    it is.  The honest statistic for a single run is an interval over the tasks
    it actually measured, which is what :func:`mean_ci` returns.  Override with
    ``CALIX_ALLOW_SINGLE_SEED=1`` only for smoke tests that never reach a table.
    """
    if len(values) < 2 and not os.environ.get("CALIX_ALLOW_SINGLE_SEED"):
        raise ValueError(
            f"{label}: a standard deviation across seeds needs at least 2 "
            f"seeds, got {len(values)}. Report mean_ci(successes, n) for a "
            f"within-run interval over tasks, or run the remaining seeds.")
    return float(np.mean(values)), float(np.std(values, ddof=1))


def fmt_ms(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a success rate, returned in percent.

    Preferred over the normal approximation because several cells are small
    (n = 11 on SWE-bench) and some sit near 0%, where the normal interval would
    run below zero.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p * 100.0, max(0.0, centre - half) * 100.0, min(1.0, centre + half) * 100.0


def mean_ci(successes: int, n: int, z: float = 1.96) -> str:
    """Format a task success rate with its within-run 95% interval."""
    p, lo, hi = wilson_ci(successes, n, z)
    if n == 0:
        return "n/a"
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


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
