
"""
Dynamic-environment robustness evaluation for CALIX.

This runner supports genuine dynamic environments through
a plugin interface. It deliberately does NOT simulate
environment changes by perturbing predefined strategies.

A dynamic plugin must provide:

1. build_environment(config)
   Returns an instance of a RealEnvironment subclass.

2. run_calix_evaluation(env, config)
   Executes the actual CALIX evaluation procedure and
   returns task-level execution records.

3. The returned environment must implement dynamic_signature().
   This method should identify the actual runtime state,
   including relevant tool availability, task-success
   conditions, or environment configuration.

Each returned record must contain:
    task_id
    seed
    success
    dynamic_signature

The runner rejects records that do not demonstrate
at least two distinct dynamic environment states.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from calix.environment import RealEnvironment


logger = logging.getLogger("CALIX.DynamicEvaluation")


def load_config(path: Path) -> dict[str, Any]:
    """Load the dynamic experiment configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("The experiment configuration must be a mapping.")

    return config


def load_plugin(module_name: str):
    """Import the user-provided dynamic evaluation plugin."""
    try:
        plugin = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import dynamic evaluation plugin: {module_name}"
        ) from exc

    required_functions = (
        "build_environment",
        "run_calix_evaluation",
    )

    for name in required_functions:
        if not callable(getattr(plugin, name, None)):
            raise AttributeError(
                f"Plugin '{module_name}' must implement {name}()."
            )

    return plugin


def validate_environment(env: Any) -> None:
    """Verify that the plugin provides a CALIX environment."""
    if not isinstance(env, RealEnvironment):
        raise TypeError(
            "The dynamic runtime must be an instance of "
            "calix.environment.RealEnvironment."
        )

    if not callable(getattr(env, "dynamic_signature", None)):
        raise TypeError(
            "The dynamic environment must implement "
            "dynamic_signature() to identify actual runtime changes."
        )


def validate_records(records: Any) -> list[dict[str, Any]]:
    """Validate task-level results and reject synthetic placeholders."""
    if records is None or isinstance(records, (str, bytes, dict)):
        raise ValueError(
            "run_calix_evaluation() must return an iterable "
            "of task-level evaluation records."
        )

    try:
        records = list(records)
    except TypeError as exc:
        raise ValueError("Invalid evaluation records.") from exc

    if not records:
        raise ValueError("No task-level evaluation records were returned.")

    required_fields = {
        "task_id",
        "seed",
        "success",
        "dynamic_signature",
    }

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record {index} must be a dictionary.")

        missing = required_fields - record.keys()
        if missing:
            raise ValueError(
                f"Record {index} is missing fields: {sorted(missing)}"
            )

        if not isinstance(record["success"], bool):
            raise TypeError(
                f"Record {index}: success must be a Boolean value."
            )

        if not str(record["task_id"]).strip():
            raise ValueError(f"Record {index} has an empty task_id.")

        if record["dynamic_signature"] in (None, ""):
            raise ValueError(
                f"Record {index} has an empty dynamic_signature."
            )

    signatures = {
        str(record["dynamic_signature"])
        for record in records
    }

    if len(signatures) < 2:
        raise RuntimeError(
            "Dynamic evaluation was rejected: only one distinct "
            "environment signature was observed. At least two "
            "genuinely different runtime states must be recorded."
        )

    return records


def summarize_records(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Generate an auditable evaluation summary."""
    total = len(records)
    successful = sum(
        1 for record in records if record["success"]
    )

    by_change_rate = defaultdict(
        lambda: {"success": 0, "total": 0}
    )

    for record in records:
        rate = str(record.get("change_rate", "not_specified"))

        by_change_rate[rate]["total"] += 1
        by_change_rate[rate]["success"] += int(
            record["success"]
        )

    grouped_results = {}

    for rate, values in by_change_rate.items():
        grouped_results[rate] = {
            "success": values["success"],
            "total": values["total"],
            "tsr": round(
                100.0 * values["success"] / values["total"], 4
            ),
        }

    return {
        "experiment": "dynamic_environment_robustness",
        "total_instances": total,
        "successful_instances": successful,
        "tsr": round(100.0 * successful / total, 4),
        "seeds": sorted({
            record["seed"] for record in records
        }),
        "unique_tasks": len({
            str(record["task_id"]) for record in records
        }),
        "unique_dynamic_states": len({
            str(record["dynamic_signature"])
            for record in records
        }),
        "results_by_change_rate": grouped_results,
    }


def save_results(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Save task-level records and the aggregate summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_path = output_path.with_suffix(".jsonl")
    summary_path = output_path.with_suffix(".summary.json")

    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Task-level records saved to: %s", records_path)
    logger.info("Summary saved to: %s", summary_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CALIX on a genuine dynamic environment."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the dynamic experiment YAML configuration.",
    )

    parser.add_argument(
        "--plugin",
        required=True,
        help="Python module implementing the dynamic environment "
             "and CALIX evaluation functions.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/dynamic/results"),
        help="Output prefix for task-level records and summary.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        config = load_config(args.config)
        plugin = load_plugin(args.plugin)

        logger.info("Building dynamic environment...")
        env = plugin.build_environment(config)

        validate_environment(env)

        logger.info(
            "Dynamic environment: %s",
            type(env).__name__,
        )

        logger.info("Starting CALIX dynamic evaluation...")

        raw_records = plugin.run_calix_evaluation(
            env,
            config,
        )

        records = validate_records(raw_records)
        summary = summarize_records(records)

        save_results(
            records,
            summary,
            args.output,
        )

        logger.info(
            "Dynamic evaluation completed: TSR = %.2f%%",
            summary["tsr"],
        )

        return 0

    except (
        FileNotFoundError,
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        logger.error("Dynamic evaluation failed: %s", exc)
        return 2

    except Exception:
        logger.exception("Unexpected dynamic evaluation failure.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
