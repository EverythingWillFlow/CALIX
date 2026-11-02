"""Dynamic-environment robustness.

IMPORTANT: this experiment is NOT runnable against the official static
benchmarks (AgentBench v0.2, GAIA, WebArena v0.2.0, SWE-bench v4.1.0), whose
task outcomes do not change over time.  The previous implementation perturbed
a synthetic "required strategy" with a configurable change rate; that harness
has been removed because it did not measure any real environment.

A genuine dynamic-environment evaluation requires a runtime whose success
conditions or available tools actually change during or across episodes
(e.g. a WebArena deployment whose site content is refreshed between episodes).
When such a runtime is available, implement it as a RealEnvironment subclass
and re-enable this script.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "run_dynamic requires a genuinely dynamic environment. The official "
        "benchmarks are static; the former simulated change-rate harness was "
        "removed because its results were synthetic. Provide a dynamic "
        "RealEnvironment implementation before running this experiment."
    )


if __name__ == "__main__":
    main()
