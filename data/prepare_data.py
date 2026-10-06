"""
Official benchmark data download and preprocessing.

Pinned versions (verified):
    - AgentBench : THUDM/AgentBench tag v0.2
    - GAIA       : gaia-benchmark/GAIA revision
                   682dd723ee1e1697e00360edccf2366dc8418dd9
    - WebArena   : web-arena-x/webarena tag v0.2.0
    - SWE-bench  : swebench==4.1.0 (PyPI)

Usage:
    python data/prepare_data.py --benchmark agentbench
    python data/prepare_data.py --benchmark gaia
    python data/prepare_data.py --benchmark webarena
    python data/prepare_data.py --benchmark swebench
    python data/prepare_data.py --benchmark all
"""

import argparse
import os
import subprocess
from pathlib import Path

DATA_DIR = Path("data/benchmarks")
THIRD_PARTY = Path("third_party")
DATA_DIR.mkdir(parents=True, exist_ok=True)
THIRD_PARTY.mkdir(parents=True, exist_ok=True)

# Pinned versions -----------------------------------------------------------
AGENTBENCH_TAG = "v0.2"
WEBARENA_TAG = "v0.2.0"
GAIA_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"
SWEBENCH_VERSION = "4.1.0"


def _run(cmd, cwd=None):
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def prepare_agentbench():
    """Clone AgentBench at tag v0.2 (original protocol, not FC)."""
    repo = THIRD_PARTY / "AgentBench"
    if not repo.exists():
        _run(["git", "clone", "https://github.com/THUDM/AgentBench.git", str(repo)])
    _run(["git", "fetch", "--all", "--tags"], cwd=repo)
    _run(["git", "checkout", AGENTBENCH_TAG], cwd=repo)

    # Record the resolved SHA in the experiment log
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    print(f"AgentBench checked out at {AGENTBENCH_TAG} ({sha})")
    print("Start the official workers manually with the v0.2 instructions.")


def prepare_gaia():
    """Download GAIA at the pinned HuggingFace revision (Parquet update)."""
    from huggingface_hub import snapshot_download
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("Please set HF_TOKEN before downloading GAIA.")

    local = snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        revision=GAIA_REVISION,
        local_dir=str(DATA_DIR / "gaia"),
    )
    print(f"GAIA downloaded at revision {GAIA_REVISION} to {local}")


def prepare_webarena():
    """Clone WebArena at tag v0.2.0 and start the official Docker sites."""
    repo = THIRD_PARTY / "webarena"
    if not repo.exists():
        _run(["git", "clone", "https://github.com/web-arena-x/webarena.git", str(repo)])
    _run(["git", "fetch", "--all", "--tags"], cwd=repo)
    _run(["git", "checkout", WEBARENA_TAG], cwd=repo)

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    print(f"WebArena checked out at {WEBARENA_TAG} ({sha})")

    setup = repo / "environment_docker" / "setup.sh"
    if setup.exists():
        _run(["bash", str(setup)], cwd=repo)

    compose = repo / "environment_docker" / "docker-compose.yml"
    if compose.exists():
        _run(["docker", "compose", "-f", str(compose), "up", "-d"], cwd=repo)
    print("WebArena Docker sites are running.")


def prepare_swebench():
    """Install the pinned SWE-bench harness and download the test split."""
    _run(["pip", "install", f"swebench=={SWEBENCH_VERSION}"])

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench", split="test")
    out = DATA_DIR / "swebench_test.jsonl"
    ds.to_json(out)
    print(f"SWE-bench: {len(ds)} tasks written to {out}")

    try:
        _run([
            "python", "-m", "swebench.harness.prepare_images",
            "--dataset", "princeton-nlp/SWE-bench",
            "--split", "test",
        ])
    except subprocess.CalledProcessError as e:
        print(f"[warning] Docker image preparation failed: {e}")
        print("You can still run evaluation if the images are already cached.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["agentbench", "gaia", "webarena", "swebench", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.benchmark in ("agentbench", "all"):
        prepare_agentbench()
    if args.benchmark in ("gaia", "all"):
        prepare_gaia()
    if args.benchmark in ("webarena", "all"):
        prepare_webarena()
    if args.benchmark in ("swebench", "all"):
        prepare_swebench()


if __name__ == "__main__":
    main()