"""Run CALIX and baselines on the official benchmarks with task-level records.

Writes one JSON record per (method, benchmark, seed, task) to
output/traces/task_records.jsonl and a mean±SD summary to
output/tables/real_results.csv.  No synthetic data: every record comes from a
real environment interaction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import sys
import time

import numpy as np

# Hard safety net: no socket may block forever.  A hung HTTPS read once
# stalled a long run for hours even though individual clients set their own
# timeouts, so enforce a global ceiling for any code path that forgot to.
socket.setdefaulttimeout(180)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (ensure_output_dirs, load_benchmark,  # noqa: E402
                    make_env, make_record_appender, train_calix)
from calix.baselines import BASELINE_NAMES, evaluate_baseline  # noqa: E402
from calix.calix import CALIX  # noqa: E402
from calix.config import load_config  # noqa: E402
from calix.websearch import build_toolkit  # noqa: E402


def _probe_llm() -> str | None:
    """One cheap reachability check; returns an error string, or None if fine.

    On 2026-10-29 an endpoint outage ran unnoticed through a full 10-method
    GAIA expansion: 229 of 350 tasks came back aborted, the exit status was
    still 0 and the summary still printed TSR values.  Nothing was lost to the
    dedupe rule, but the entire budget went into writing rows that only said
    "unreachable".  Probing up front turns that into a one-line refusal; the
    same probe gates each retry round so a dead endpoint is never retried into.
    """
    from calix.llm import OpenAICompatibleLLM

    model = os.environ.get("CALIX_MODEL") or "gpt-4o"
    # One deliberate attempt, not six with backoff: a health check that waits
    # out the full retry policy defeats its own purpose.
    saved = os.environ.get("CALIX_LLM_RETRIES")
    os.environ["CALIX_LLM_RETRIES"] = "1"
    try:
        OpenAICompatibleLLM(model=model).chat("Reply with the word ok.",
                                              max_tokens=8)
        return None
    except Exception as exc:  # noqa: BLE001 - the point is to report anything
        return f"{type(exc).__name__}: {str(exc)[:180]}"
    finally:
        if saved is None:
            os.environ.pop("CALIX_LLM_RETRIES", None)
        else:
            os.environ["CALIX_LLM_RETRIES"] = saved


_FATAL_STATUS_RE = re.compile(r"HTTP (401|402|403)\b")


def _is_fatal(error: str | None) -> bool:
    """True when the endpoint refused us for a reason waiting cannot fix.

    ``_probe_llm`` returns a string so it can be logged anywhere; this reads
    that string back.  A 402 (balance exhausted) does not heal with time, so
    every layer that would otherwise sleep and retry checks this first.
    """
    return bool(error) and bool(_FATAL_STATUS_RE.search(error))


def _fatal_from_records(records) -> str | None:
    """The first fatal refusal among aborted rows, if any.

    Read from the rows themselves rather than a fresh probe: the aborted tasks
    already carry the endpoint's own words, and probing again would spend one
    more (doomed) call to learn what the log already says.
    """
    for r in records or []:
        if r.get("aborted") and _is_fatal(r.get("error", "")):
            return str(r.get("error", ""))[:120]
    return None


def _require_llm() -> None:
    """Refuse to start against a dead endpoint (see ``_probe_llm``)."""
    error = _probe_llm()
    if error is None:
        return
    if _is_fatal(error):
        raise SystemExit(
            f"[run_benchmarks] refusing to start: the endpoint is refusing "
            f"requests ({error}).\n"
            f"  Waiting cannot fix this: top up the balance or put a working "
            f"key in .env, then re-run the same command -- every task already "
            f"measured is kept and only the gaps will be run.")
    raise SystemExit(
        f"[run_benchmarks] refusing to start: the inference endpoint is "
        f"unreachable ({error}).\n"
        f"  A run started now would write rows that record nothing, and "
        f"they would be indistinguishable from real work in the summary. "
        f"Pass --allow-unreachable to override.")


def _wait_for_endpoint(budget: float, interval: float = 60.0) -> bool:
    """Block until the endpoint answers again, or ``budget`` seconds run out.

    Retrying the instant a task aborts is what wasted the 2026-10-29 run:
    against an ongoing outage the retry is a guaranteed loss that costs the
    same half hour as the original attempt.  Wait for recovery instead.
    """
    deadline = time.monotonic() + budget
    while True:
        error = _probe_llm()
        if error is None:
            return True
        if _is_fatal(error):
            # A revoked key or an empty balance is not an outage: sleeping
            # through the budget only delays the message the user needs.
            print(f"[run_benchmarks] endpoint refused us and waiting will not "
                  f"help: {error}\n"
                  f"  Fix the account (top up / replace the key in .env) and "
                  f"re-run -- already measured tasks are kept.", flush=True)
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"[run_benchmarks] endpoint still unreachable after "
                  f"{budget:.0f}s: {error}", flush=True)
            return False
        nap = min(interval, remaining)
        print(f"[run_benchmarks] endpoint unreachable ({error}); "
              f"retrying in {nap:.0f}s", flush=True)
        time.sleep(nap)


def _run_cell(cfg, method, benchmark, seed, test, n_tasks, episodes,
              shared_tools, env_factory, workers, calix_model=None,
              on_record=None):
    """Evaluate one (method, benchmark, seed) cell; returns (records, tsr).

    ``on_record``, when given, is called with each record the moment its task
    finishes, so a killed run keeps everything it had already measured.
    """
    if method.lower() == "calix":
        if calix_model:
            # Evaluate the policy that was actually trained earlier instead of
            # training a fresh one here.  Without this, ``--episodes 0`` would
            # hand a randomly initialised Q-network to the benchmark and the
            # CALIX column would be meaningless.
            from common import load_model_params
            model = CALIX(cfg, seed=seed, tools=shared_tools)
            load_model_params(model, calix_model)
            print(f"[run_benchmarks] loaded CALIX policy from {calix_model}",
                  flush=True)
        else:
            model, _ = train_calix(cfg, seed, episodes, benchmark=benchmark,
                                   tools=shared_tools)
        env = env_factory()
        res = model.evaluate(env, num_tasks=n_tasks, workers=workers,
                             env_factory=env_factory, on_record=on_record)
        return res["records"], res["tsr"]
    env = env_factory()
    out = evaluate_baseline(method, env, n_tasks, seed,
                            model=cfg["llm"]["model"], tools=shared_tools,
                            workers=workers, env_factory=env_factory,
                            on_record=on_record)
    return out["records"], out["tsr"]


def _retry_aborted(cfg, method, benchmark, seed, test, records, tsr, episodes,
                   shared_tools, make_factory, args, on_record=None):
    """Re-run the tasks that aborted, after waiting for the endpoint to return.

    An abort means the infrastructure failed, not that the method answered
    wrongly, so the row is genuinely worth another attempt -- but only once the
    endpoint answers again.  Retrying straight away, as the 2026-10-29 run did,
    just spends another half hour to record the same absence.
    """
    for attempt in range(args.retry_aborted):
        missing = {r["task_id"] for r in records if r.get("aborted")}
        if not missing:
            break
        wait = min(args.retry_wait_cap, args.retry_wait * (2 ** attempt))
        if _is_fatal(_probe_llm()):
            # Balance exhausted / key revoked: the 47 aborted tasks are not
            # coming back by waiting, and the next method would abort too.
            print(f"[run_benchmarks] {benchmark}/{method}: the endpoint is "
                  f"refusing requests and waiting will not fix it; skipping "
                  f"the remaining retries.", flush=True)
            break
        print(f"[run_benchmarks] {benchmark}/{method}: {len(missing)} tasks "
              f"aborted; waiting up to {wait:.0f}s for the endpoint before "
              f"retry {attempt + 1}/{args.retry_aborted}", flush=True)
        if not _wait_for_endpoint(wait):
            break
        subset = [t for t in test if t["task_id"] in missing]
        try:
            retry_records, _ = _run_cell(
                cfg, method, benchmark, seed, subset, len(subset),
                episodes, shared_tools, make_factory(subset),
                args.workers, calix_model=args.calix_model,
                on_record=on_record)
        except (EnvironmentError, NotImplementedError) as exc:
            print(f"[run_benchmarks] retry {attempt + 1} could not run "
                  f"either: {exc}", flush=True)
            break
        best = {r["task_id"]: r for r in records}
        for r in retry_records:
            # A graded row always beats an aborted one, regardless of which
            # ran later; between two graded rows the later run wins.
            if not r.get("aborted") or best.get(r["task_id"], {}).get("aborted"):
                best[r["task_id"]] = r
        records = list(best.values())
        graded = [r for r in records if not r.get("aborted")]
        tsr = sum(1 for r in graded if r["success"]) / max(len(graded), 1) * 100
    return records, tsr


def _completed_tasks(cfg, method: str, benchmark: str) -> set:
    """Task ids with a *graded* row already on disk for (method, benchmark).

    An aborted row is a missing measurement, not a completed one, so it does
    not count as done: that is what lets a restarted run pick up exactly the
    tasks an outage left behind.
    """
    path = os.path.join(ensure_output_dirs(cfg)["traces"],
                        "task_records.jsonl")
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("method") == method and r.get("benchmark") == benchmark
                    and not r.get("aborted")):
                done.add(r["task_id"])
    return done


def _graded_rows(cfg, method: str, benchmark: str, seed) -> list[dict]:
    """The graded task records already on disk for one (method, benchmark).

    The task records, not the summary CSV, are what a run actually produced.
    When ``--resume`` skips a finished cell it still has to be reported: on
    2026-10-29 a resumed GAIA sweep rewrote ``real_results.csv`` from the rows
    *this invocation* measured and turned eight fully measured methods into
    ``unavailable``, purely because resume had nothing left to do for them.
    Later duplicates win, matching the audit's de-duplication rule.
    """
    path = os.path.join(ensure_output_dirs(cfg)["traces"],
                        "task_records.jsonl")
    best: dict[str, dict] = {}
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (r.get("method") != method or r.get("benchmark") != benchmark
                    or r.get("aborted") or "task_id" not in r):
                continue
            if seed is not None and r.get("seed") not in (None, seed):
                continue
            best[r["task_id"]] = r
    return list(best.values())


def _load_previous(path: str) -> dict:
    """Read existing summary rows, keyed by (benchmark, method)."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["benchmark"], row["method"])] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--eval-tasks", type=int, default=None)
    ap.add_argument("--task-ids", nargs="+", default=None,
                    help="run only these task ids (smoke tests); a benchmark "
                         "with no match keeps its normal eval pool")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel task workers (tasks are independent)")
    ap.add_argument("--skip-unavailable", action="store_true",
                    help="record and continue when a benchmark runtime is "
                         "unreachable (for matrix/smoke runs); the default is "
                         "to abort so a dead runtime is never read as 0%% TSR")
    ap.add_argument("--merge-results", action="store_true",
                    help="when real_results.csv already exists, keep the rows "
                         "for benchmarks this invocation did not run instead of "
                         "overwriting them (for re-running one benchmark)")
    ap.add_argument("--output-dir", default=None,
                    help="override experiment.output_dir; needed when several "
                         "benchmarks are run as concurrent processes, so that "
                         "each one writes its own tables/ and traces/ instead "
                         "of racing on the same real_results.csv")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="override execution.max_rounds; browser tasks need a "
                         "much larger budget than QA tasks, and the right "
                         "value is a property of the benchmark, not the config")
    ap.add_argument("--allow-unreachable", action="store_true",
                    help="start even when the inference endpoint cannot be "
                    "reached (see _probe_llm); by default the run refuses, "
                    "because rows written against a dead endpoint record "
                    "nothing and look like completed work")
    ap.add_argument("--retry-aborted", type=int, default=2,
                    help="re-run tasks that aborted this many times. An "
                    "aborted task is a missing measurement, not a wrong "
                    "answer, so it is worth another attempt -- but only once "
                    "the endpoint answers again, hence the wait below")
    ap.add_argument("--retry-wait", type=float, default=300.0,
                    help="seconds to wait for the endpoint to recover before "
                    "the first retry round; doubles each round")
    ap.add_argument("--retry-wait-cap", type=float, default=3600.0,
                    help="upper bound on a single wait between retry rounds")
    ap.add_argument("--resume", action="store_true",
                    help="skip (method, task) cells that already have a graded "
                         "row in this output dir's traces. A killed run can be "
                         "restarted with the same command and continues where "
                         "it stopped instead of starting over")
    ap.add_argument("--max-abort-fraction", type=float, default=0.15,
                    help="exit non-zero when a larger share of a cell's tasks "
                    "aborted. A mostly-aborted cell is not a measurement, and "
                    "exit status is the only signal a scheduler will see")
    ap.add_argument("--calix-model", default=None,
                    help="path to a saved q-network (.npz); when method calix is "
                         "run, evaluate this policy instead of training a new "
                         "one, so CALIX and the baselines share one task set, "
                         "one tool budget and one env instance")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_rounds:
        cfg["execution"]["max_rounds"] = args.max_rounds
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if not args.allow_unreachable:
        _require_llm()
    dirs = ensure_output_dirs(cfg)
    benchmarks = args.benchmark or cfg["benchmarks"]["names"]
    seeds = args.seeds or cfg["experiment"]["seeds"]
    methods = args.methods or cfg.get("baselines") or BASELINE_NAMES
    episodes = args.episodes or cfg["execution"]["train_episodes"]
    eval_tasks = args.eval_tasks or cfg["execution"]["eval_tasks_per_benchmark"]

    # one toolkit shared by CALIX and every baseline: identical tool access
    shared_tools = build_toolkit(cfg)

    summary_rows = []
    integrity: list[str] = []
    # Set when the endpoint refuses us for good (empty balance, revoked key).
    # The remaining methods would abort identically, so the loop stops instead
    # of filling the table with rows that only record the refusal.
    fatal_error: str | None = None
    for benchmark in benchmarks:
        train, test = load_benchmark(benchmark)
        if args.task_ids:
            wanted = [t for t in test if t["task_id"] in set(args.task_ids)]
            if wanted:
                test = wanted
            else:
                print(f"[run_benchmarks] no --task-ids match in {benchmark}'s "
                      f"eval pool; using the pool as-is", flush=True)
        print(f"[run_benchmarks] {benchmark}: {len(train)} train / {len(test)} eval tasks")

        for method in methods:
            if fatal_error is not None:
                summary_rows.append({
                    "benchmark": benchmark, "method": method, "seed": seeds[0],
                    "tsr": None, "n_tasks": 0,
                    "error": f"not run: endpoint refused ({fatal_error})"})
                continue

            def make_factory(tasks):
                # Bind explicitly: a lambda closing over the loop variable
                # would silently re-target on the next retry round.
                return lambda: make_env(tasks, benchmark, cfg)

            for seed in seeds:
                if args.resume:
                    # ``--eval-tasks`` is the goal line, not the amount of new
                    # work: the cell aims at test[:eval_tasks], and resume runs
                    # only the tasks inside that window that are not graded
                    # yet.  Restarting the same command therefore continues the
                    # goal instead of silently extending past it.
                    scope = test[:min(eval_tasks, len(test))]
                    done = _completed_tasks(cfg, method, benchmark)
                    subset = [t for t in scope if t["task_id"] not in done]
                    skipped = len(scope) - len(subset)
                    if skipped:
                        print(f"[run_benchmarks] {benchmark}/{method}: "
                              f"{skipped}/{len(scope)} already graded, "
                              f"continuing with {len(subset)}", flush=True)
                    if not subset:
                        print(f"[run_benchmarks] {benchmark}/{method}: all "
                              f"{len(scope)} tasks graded, nothing to do",
                              flush=True)
                        # Report what is already measured instead of dropping
                        # the cell from the summary (see ``_graded_rows``).
                        prior = _graded_rows(cfg, method, benchmark, seed)
                        summary_rows.append({
                            "benchmark": benchmark, "method": method,
                            "seed": seed,
                            "tsr": (sum(1 for r in prior if r["success"])
                                    / max(len(prior), 1) * 100),
                            "n_tasks": len(prior), "aborted": 0,
                            "successes": sum(1 for r in prior if r["success"]),
                        })
                        continue
                else:
                    subset = test
                n_tasks = len(subset) if args.resume else min(eval_tasks,
                                                              len(subset))
                on_record = make_record_appender(cfg, method, benchmark, seed)
                try:
                    records, tsr = _run_cell(
                        cfg, method, benchmark, seed, subset, n_tasks,
                        episodes, shared_tools, make_factory(subset),
                        args.workers, calix_model=args.calix_model,
                        on_record=on_record)
                    records, tsr = _retry_aborted(
                        cfg, method, benchmark, seed, test, records, tsr,
                        episodes, shared_tools, make_factory, args,
                        on_record=on_record)
                except (EnvironmentError, NotImplementedError) as exc:
                    msg = f"{type(exc).__name__}: {str(exc)[:180]}"
                    print(f"[run_benchmarks] UNAVAILABLE {benchmark}/{method}: "
                          f"{msg}", flush=True)
                    if not args.skip_unavailable:
                        raise
                    summary_rows.append({
                        "benchmark": benchmark, "method": method, "seed": seed,
                        "tsr": None, "n_tasks": 0, "successes": 0,
                        "error": msg})
                    continue
                aborted = [r for r in records if r.get("aborted")]
                if (records and len(aborted) / len(records)
                        > args.max_abort_fraction):
                    integrity.append(
                        f"{benchmark}/{method} seed={seed}: "
                        f"{len(aborted)}/{len(records)} tasks aborted")
                    if fatal_error is None:
                        fatal_error = _fatal_from_records(records)
                # Records were already persisted line by line as each task
                # finished (``make_record_appender``), so a kill here loses
                # nothing.  Do NOT append them again -- that would write every
                # row a second time.
                summary_rows.append({
                    "benchmark": benchmark, "method": method, "seed": seed,
                    "tsr": tsr, "n_tasks": len(records),
                    "aborted": len(aborted),
                    "successes": sum(r["success"] for r in records),
                })
                print(f"[run_benchmarks] {benchmark} {method} seed={seed} "
                      f"TSR={tsr:.1f}", flush=True)

    # mean ± SD across seeds, computed from the task-level records
    out_path = os.path.join(dirs["tables"], "real_results.csv")
    merged = _load_previous(out_path) if args.merge_results else {}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "method", "mean_tsr", "std_tsr", "n_seeds",
                    "status"])
        for benchmark in benchmarks:
            for method in methods:
                rows = [r for r in summary_rows
                        if r["benchmark"] == benchmark and r["method"] == method]
                vals = [r["tsr"] for r in rows if r.get("tsr") is not None]
                if vals:
                    # A TSR computed over mostly-aborted rows is a number, but
                    # not a measurement: on 2026-10-29 sweagent showed 62.5%
                    # off 8 surviving tasks and calix 0.0% off none, both
                    # marked "ok".  The status column now says so.
                    n_all = sum(r.get("n_tasks") or 0 for r in rows)
                    n_ab = sum(r.get("aborted") or 0 for r in rows)
                    if n_all and n_ab / n_all > args.max_abort_fraction:
                        status = f"incomplete: {n_ab}/{n_all} aborted"
                    else:
                        status = "ok"
                    w.writerow([benchmark, method, f"{np.mean(vals):.2f}",
                                f"{np.std(vals):.2f}", len(vals), status])
                else:
                    err = next((r.get("error", "no records") for r in rows), "")
                    w.writerow([benchmark, method, "", "", 0,
                                f"unavailable: {err}"])
        for (benchmark, method), row in sorted(merged.items()):
            if benchmark in benchmarks and method in methods:
                continue          # this invocation owns the cell
            w.writerow([benchmark, method, row["mean_tsr"], row["std_tsr"],
                        row["n_seeds"], row["status"]])
    print(f"[run_benchmarks] saved {out_path}")

    # Exit status is the only part of a run a scheduler, a Makefile or a cron
    # wrapper will ever look at.  It has to mean "this cell was measured", not
    # "this process ended".
    if integrity:
        print("\n[run_benchmarks] INCOMPLETE - the following cells mostly did "
              "not measure anything:", file=sys.stderr)
        for line in integrity:
            print(f"  - {line}", file=sys.stderr)
        if fatal_error is not None and _is_fatal(fatal_error):
            print("  The endpoint is refusing requests (empty balance / bad "
                  "key), so waiting cannot help: top up or replace the key in "
                  ".env, then re-run the same command -- finished tasks are "
                  "kept.", file=sys.stderr)
        else:
            print("  Fill the gaps once the endpoint recovers; "
                  "tools/resume_plan.py prints the task ids.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
