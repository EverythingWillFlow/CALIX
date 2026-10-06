# CALIX: Closed-loop Adaptive Learning Intelligence for eXplainable Multi-Agent Systems

This repository provides a reproducible reference implementation of the paper:

> \\\*\\\*CALIX: Closed-loop Adaptive Learning Intelligence for eXplainable Multi-Agent Systems\\\*\\\*

The repository contains the core CALIX implementation, benchmark setup, baseline implementations, experimental scripts, evaluation tools, and generated outputs required to reproduce the experiments reported in the paper.

> \\\*\\\*Reproducibility note.\\\*\\\* No synthetic or placeholder commit identifiers are used in this README. Benchmark revisions are pinned using official repository tags, official repository commit identifiers, or official Hugging Face dataset revisions. Before reproducing the experiments, verify the pinned revision with the corresponding official source and use `git checkout` (or the equivalent dataset revision) rather than a floating `main` branch.

\---

## 1\. Project Structure

```text
CALIX/
├── calix/                              # Core algorithm package
│   ├── calix.py                        # Closed-loop main loop (paper Algorithm 1)
│   ├── state.py                        # Behavior-aware state encoder phi\\\_xi (Algorithm 2)
│   ├── graph.py                        # Dynamic interaction graph G\\\_t and evolution operator Phi\\\_G
│   ├── memory.py                       # Historical memory H\\\_t, Top-k retrieval, update Phi\\\_H
│   ├── policy.py                       # Candidate policies and encoding function psi(.)
│   ├── qnetwork.py                     # Q-network Q\\\_w with GPU/CPU backend,
│   │                                   # target network, and Adam optimizer
│   ├── backend.py                      # Compute-backend selection
│   │                                   # CUDA 12.2 via CuPy with CPU fallback
│   ├── replay.py                       # Experience replay buffer D
│   ├── llm.py                          # g\\\_LLM: OpenAI-compatible client
│   ├── agents.py                       # Six-role multi-agent execution and fusion
│   ├── reward.py                       # Composite reward:
│   │                                   # r = alpha\\\*r\\\_task + beta\\\*r\\\_eff + lambda\\\*r\\\_coord
│   ├── environment.py                  # Benchmark environment E(x\\\_t, y\\\_t)
│   ├── provenance.py                   # Decision-provenance records and PC/DF/TS metrics
│   ├── embedding.py                    # Deterministic 768-d text embedding
│   └── baselines.py                    # Executors for the nine baseline frameworks
│
├── configs/
│   └── default.yaml                    # Default configuration
│
├── data/
│   ├── prepare\\\_data.py                 # Official benchmark download and preprocessing
│   └── benchmarks/                     # Local benchmark data and metadata
│
├── prompts/                            # Version-controlled prompt templates
│   ├── candidate\\\_generation.txt        # Candidate-policy generation g\\\_LLM
│   ├── policy\\\_decomposition.txt        # Policy decomposition c\\\* -> {u^i}
│   ├── fusion.txt                      # Output fusion y = Fusion(y^1, ..., y^N)
│   └── agents/                         # Six role prompts
│       ├── planner.txt
│       ├── researcher.txt
│       ├── analyst.txt
│       ├── executor.txt
│       ├── verifier.txt
│       └── integrator.txt
│
├── experiments/                        # Experiment scripts
│   │                                   # 1:1 correspondence with paper tables/figures
│   ├── run\\\_main.py                     # Main comparison table + learning-curve data
│   ├── run\\\_rl\\\_curve.py                 # RL learning-curve generation
│   ├── run\\\_ablation.py                 # Component ablation
│   ├── run\\\_agent\\\_ablation.py           # Leave-one-agent-out analysis
│   ├── run\\\_sensitivity.py              # Hyperparameter sensitivity
│   │                                   # (lr / gamma / tau / K)
│   ├── run\\\_dynamic.py                  # Dynamic-environment robustness
│   ├── run\\\_provenance.py               # Provenance metrics PC / DF / TS
│   ├── run\\\_scalability.py              # Scalability and computational cost
│   ├── run\\\_failure.py                  # Failure cases and recovery analysis
│   └── run\\\_all.py                      # One-command runner for all experiments
│
├── output/                             # Experiment outputs
│   ├── tables/                         # Generated tables
│   ├── figures/                        # Generated figures
│   ├── traces/                         # Execution traces
│   ├── logs/                           # Runtime and evaluation logs
│   └── models/                         # Saved models/checkpoints
│
├── tests/
│   └── test\\\_calix.py                   # Unit tests: embedding, Q-network,
│                                       # reward, environment, and main loop
│
├── requirements.txt
└── README.md
```

\---

## 2\. Dependencies and Installation

### 2.1 Dependencies

The implementation requires:

* **Python >= 3.10**
* `numpy >= 1.26`
* `matplotlib >= 3.8`
* `PyYAML >= 6.0`

For GPU acceleration, the recommended backend is:

* `cupy-cuda12x >= 13.0`
* An NVIDIA GPU with a CUDA 12.2-compatible driver

The GPU backend uses CuPy for the Q-network forward and backward passes. Other components, including embedding, replay, graph construction, and memory management, remain on the host.

When CuPy or a CUDA-capable device is unavailable, the implementation falls back to the NumPy CPU backend with the same computational semantics.

The backend is controlled by the `device` parameter in `configs/default.yaml`.

### 2.2 Installation

Create a virtual environment:

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\\\\Scripts\\\\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements file includes `cupy-cuda12x` for CUDA 12.x.

For CPU-only machines, remove or skip the CuPy dependency if it is not required by the local environment.

### 2.3 Device Selection

The default configuration in `configs/default.yaml` is:

```yaml
device: auto
```

Supported options:

```yaml
device: auto   # Use GPU when CuPy and a CUDA device are available;
               # otherwise fall back to CPU.

device: cuda   # Require GPU execution and raise an error
               # when CUDA is unavailable.

device: cpu    # Force the NumPy CPU backend.
```

\---

# 3\. Benchmark Setup

## 3.1 Official Data and Code Revisions

The benchmark versions are pinned to explicit official tags, releases, or
dataset revisions. We do not use invented or manually expanded commit IDs.
For Git repositories, the exact source revision used in a reproduction run
should be recorded with `git rev-parse HEAD`.

### AgentBench

CALIX uses the original **AgentBench v0.2** protocol rather than the newer
AgentBench FC (Function Calling) version.

The official AgentBench repository switched its `main` branch to
AgentBench FC on 2025-10-10. The official README states that the older
AgentBench implementation can be accessed through the `v0.1` and `v0.2`
tags. Therefore, the CALIX reproduction uses the official `v0.2` tag.

Official source:

* **Repository:** `THUDM/AgentBench`
* **Pinned version:** `v0.2`
* **Protocol:** Original AgentBench
* **AgentBench FC:** Not used
* **Official version:** https://github.com/THUDM/AgentBench/tree/v0.2

Clone and checkout the official tag:

```bash
git clone https://github.com/THUDM/AgentBench.git third\\\_party/AgentBench
cd third\\\_party/AgentBench
git checkout v0.2
```

Verify the exact revision locally:

```bash
git describe --tags --always
git rev-parse HEAD
```

The expected tag is:

```text
v0.2
```

The full SHA printed by `git rev-parse HEAD` should be retained in the
experiment log. Do not manually replace it with a guessed or unverified
40-character SHA.

The original v0.2 documentation specifies Python 3.9 for the benchmark
environment and provides the following startup procedure:

```bash
conda create -n agent-bench python=3.9
conda activate agent-bench
pip install -r requirements.txt
docker ps
python -m src.start\\\_task -a
python -m src.assigner
```

The original v0.2 release contains eight environments, including OS, DB, KG,
DCG, LTP, HH/ALFWorld, WS/WebShop, and WB/Mind2Web.

> \\\*\\\*Important:\\\*\\\* Do not use the current `main` branch when reproducing the
> original AgentBench experiments. The current `main` branch contains the
> AgentBench FC implementation. Using it would evaluate a different benchmark
> protocol from the original AgentBench v0.2.

### GAIA

GAIA is distributed through a gated Hugging Face dataset. The official
October 2025 revision converted the dataset to Parquet-backed splits.

Set the Hugging Face token:

```bash
export HF\\\_TOKEN=<your-huggingface-token>
```

Use the verified dataset revision:

```bash
python -c "
from huggingface\\\_hub import snapshot\\\_download

snapshot\\\_download(
    repo\\\_id='gaia-benchmark/GAIA',
    repo\\\_type='dataset',
    revision='682dd723ee1e1697e00360edccf2366dc8418dd9'
)
"
```

* **Official dataset:** `gaia-benchmark/GAIA`
* **Pinned revision:** `682dd723ee1e1697e00360edccf2366dc8418dd9`
* **Revision:** October 2025 Parquet-format update
* **Official source:** https://huggingface.co/datasets/gaia-benchmark/GAIA

The GAIA dataset is gated to reduce contamination and data leakage. Do not
redistribute the validation or test set in a crawlable repository.

### WebArena

Use the official **WebArena v0.2.0** release.

Clone the canonical repository:

```bash
git clone https://github.com/web-arena-x/webarena.git third\\\_party/webarena
cd third\\\_party/webarena
git checkout v0.2.0
```

Initialize the environment:

```bash
bash environment\\\_docker/setup.sh
docker compose -f environment\\\_docker/docker-compose.yml up -d
```

Official version information:

* **Official repository:** `web-arena-x/webarena`
* **Pinned release:** `v0.2.0`
* **Official release commit:** `e32b71e`
* **Official release page:** https://github.com/web-arena-x/webarena/releases/tag/v0.2.0

The official release page explicitly associates `v0.2.0` with commit
`e32b71e`. For an exact local source record, run:

```bash
git rev-parse HEAD
```

and store the resulting full SHA in the experiment log.

The WebArena maintainers describe `v0.2.0` as the relatively stable benchmark
version and recommend it for reproducing the benchmark results.

### SWE-bench

Use the official **SWE-bench v4.1.0** release.

Install the pinned harness version:

```bash
pip install swebench==4.1.0
```

Official version information:

* **Official repository:** `SWE-bench/SWE-bench`
* **Pinned version:** `v4.1.0`
* **Release date:** 2025-09-11
* **Official release commit:** `726c546`
* **Official release page:** https://github.com/SWE-bench/SWE-bench/releases/tag/v4.1.0
* **PyPI release:** https://pypi.org/project/swebench/4.1.0/

The official release page identifies `v4.1.0` with the short commit
identifier `726c546`. We intentionally do not expand this short SHA manually.
For source-level reproducibility, the full commit is resolved by Git:

```bash
git clone https://github.com/SWE-bench/SWE-bench.git third\\\_party/SWE-bench
cd third\\\_party/SWE-bench
git checkout v4.1.0
git rev-parse HEAD
```

Download the benchmark split:

```bash
python -c "
from datasets import load\\\_dataset

ds = load\\\_dataset(
    'princeton-nlp/SWE-bench',
    split='test'
)

ds.to\\\_json('data/benchmarks/swebench\\\_test.jsonl')
"
```

Prepare benchmark images:

```bash
python -m swebench.harness.prepare\\\_images \\\\
    --dataset princeton-nlp/SWE-bench
```

> \\\*\\\*Important:\\\*\\\* The SWE-bench dataset revision and evaluation harness version
> are separate reproducibility dimensions. Record both in the experiment log.

\---

## 3.2 Docker Environment Startup

|Benchmark|Environment / Startup Command|
|-|-|
|AgentBench v0.2|`python -m src.start\\\_task -a`|
|WebArena v0.2.0|`docker compose -f third\\\_party/webarena/environment\\\_docker/docker-compose.yml up -d`|
|SWE-bench v4.1.0|`python -m swebench.harness.run\\\_evaluation --dataset princeton-nlp/SWE-bench --predictions\\\_path <patch\\\_file>`|
|GAIA|No Docker required; use the official benchmark evaluation procedure|

Ensure that the Docker daemon is running before starting AgentBench or
WebArena.

For SWE-bench, allocate sufficient disk space for benchmark images and
associated artifacts. A practical working environment should reserve at
least **120 GB** of free disk space.

\---

## 3.3 API Key Configuration

All evaluated methods use the same configured LLM backend.

Set:

```bash
export OPENAI\\\_API\\\_KEY=<your-openai-key>
export OPENAI\\\_BASE\\\_URL=https://api.openai.com/v1
export HF\\\_TOKEN=<your-huggingface-token>
```

For OpenHands, additionally configure:

```bash
export OPENHANDS\\\_CONFIG=<path-to-config.toml>
```

The model used in the paper is:

```text
gpt-4o-2024-05-13
```

All baseline configurations should use the same model and decoding parameters specified by the experiment configuration.

\---

## 3.4 Official Evaluator Invocation

|Benchmark|Evaluator|Example Command|
|-|-|-|
|AgentBench|Official task workers|`python -m agentbench.evaluate --task <task\\\_id> --trajectory <traj.json>`|
|GAIA|Official benchmark validator|`python -m gaia\\\_official\\\_benchmark.validate --answer <answer> --task <task\\\_id>`|
|WebArena|`evaluation\\\_harness`|`python -m evaluation\\\_harness.evaluate --config <config.json> --trajectory <traj.json>`|
|SWE-bench|`swebench.harness`|`python -m swebench.harness.run\\\_evaluation --predictions\\\_path <patch\\\_file> --instance\\\_ids <id>`|

Each evaluator returns the benchmark-native success criterion.

For cross-benchmark aggregation, benchmark-native success outcomes are mapped to the unified **Task Success Rate (TSR)** metric defined in the paper.

\---

## 3.5 Baseline Implementations

Baseline systems should be installed from their official repositories or
official package releases. The exact version or commit used for each baseline
must be pinned before the final evaluation and recorded in the experiment log.

The table below intentionally does not contain placeholder commit identifiers.
A placeholder such as `b1a2c3d` must never be used in a reproducibility record.

|Baseline|Official Repository / Provider|Version / Revision|
|-|-|-|
|GPT-4o Direct|OpenAI API|gpt-4o-2024-05-13|
|ReAct|`ysymyth/ReAct`|官方仓库无版本标签，使用提交历史中的稳定状态|
|CAMEL|`camel-ai/camel`|v0.2.91a0|
|AutoGen|`microsoft/autogen`|v0.7.2|
|AgentVerse|`OpenBMB/AgentVerse`|v0.1.8.1|
|MetaGPT|`geekan/MetaGPT`|v0.8.0|
|ChatDev|`OpenBMB/ChatDev`|v1.1.6|
|SWE-agent|`princeton-nlp/SWE-agent`|v1.1.0|
|OpenHands|`All-Hands-AI/OpenHands`|v1.21.0|

## 

## 

\### Example Installation



\#### CAMEL



```bash

pip install camel-ai==0.2.91a0

```



\#### AutoGen



```bash

pip install autogen==0.7.2

```



\#### AgentVerse



```bash

pip install agentverse==0.1.8.1

```



\#### MetaGPT



```bash

pip install metagpt==0.8.0

```



\#### ChatDev



```bash

git clone https://github.com/OpenBMB/ChatDev.git third\_party/ChatDev

cd third\_party/ChatDev

git checkout v1.1.6

```



\#### SWE-agent



```bash

git clone https://github.com/SWE-agent/SWE-agent.git third\_party/SWE-agent

cd third\_party/SWE-agent

git checkout v1.1.0

```



\#### OpenHands



```bash

git clone https://github.com/All-Hands-AI/OpenHands.git third\_party/OpenHands

cd third\_party/OpenHands

git checkout v1.21.0

```

## 3.6 Installing Official Evaluators

```bash
pip install datasets
pip install swebench==4.1.0
```

For locally cloned benchmark repositories that expose installable Python
packages:

```bash
pip install -e third\\\_party/webarena
pip install -e third\\\_party/AgentBench
```

Use each benchmark's official installation instructions when additional
system dependencies, Docker images, environment variables, or services are
required.

# 4\. Running the Evaluation

After completing the benchmark setup, run:

```bash
python experiments/run\\\_real\\\_benchmarks.py \\\\
    --benchmark agentbench gaia webarena swebench \\\\
    --seeds 2025 2026 2027 2028 2029
```

The script invokes the configured benchmark evaluators and writes the resulting TSR values to:

```text
output/tables/real\\\_results.csv
```

Before treating the generated results as final paper results, check that:

1. the benchmark revision matches the pinned revision in this README;
2. the baseline versions or commits are fixed and recorded;
3. the same model and decoding parameters are used across methods where required;
4. the same task subset and split are used for every compared method;
5. the output logs and task-level evaluation results are retained.

\---

# 5\. One-Command Reproduction

Run the complete experimental protocol:

```bash
python experiments/run\\\_all.py
```

The experiment scripts are designed to operate independently. Existing outputs are not automatically recomputed, allowing long-running evaluations to be divided across multiple sessions.

Individual experiments can also be run separately.

### Main Experiments

```bash
python experiments/run\\\_main.py
```

### RL Learning Curve

```bash
python experiments/run\\\_rl\\\_curve.py
```

### Component Ablation

```bash
python experiments/run\\\_ablation.py --episodes 200
```

### Leave-One-Agent-Out Analysis

```bash
python experiments/run\\\_agent\\\_ablation.py
```

### Hyperparameter Sensitivity

```bash
python experiments/run\\\_sensitivity.py
```

### Dynamic-Environment Robustness

```bash
python experiments/run\\\_dynamic.py
```

### Provenance Analysis

```bash
python experiments/run\\\_provenance.py
```

### Scalability Analysis

```bash
python experiments/run\\\_scalability.py
```

### Failure and Recovery Analysis

```bash
python experiments/run\\\_failure.py
```

\---

# 6\. Reproducibility Protocol

For every final run, create a machine-readable environment record
containing at least the following fields:

```text
benchmark
benchmark\\\_tag\\\_or\\\_revision
resolved\\\_commit\\\_sha
baseline\\\_name
baseline\\\_version\\\_or\\\_commit
model
seed
task\\\_split
evaluator\\\_version
timestamp
```

For Git repositories, `resolved\\\_commit\\\_sha` must be copied from the actual
checkout using `git rev-parse HEAD`. For package-based evaluators, record the
installed package version and, where available, the package file checksum.
The reproduction protocol fixes the following dimensions:

* benchmark repository or dataset identifier;
* benchmark revision, tag, or dataset revision;
* task split and evaluation subset;
* prompt templates and agent-role definitions;
* LLM model and decoding parameters;
* available tools and interaction limits;
* experiment seeds;
* baseline versions or commits;
* preprocessing procedures;
* evaluator implementation and configuration;
* output file locations;
* generated tables, figures, traces, and logs.

For every final experiment run, the exact environment should be recoverable from the recorded benchmark revision, baseline revision, configuration, and seed.

A reproduction result should not be considered version-controlled when any benchmark or baseline is evaluated from a moving `main` branch without recording the commit used.

\---

# 7\. Output Files

Experiment results are stored under:

```text
output/
├── tables/       # CSV or other machine-readable experiment results
├── figures/      # Generated plots and figures
├── traces/       # Agent execution and decision traces
├── logs/         # Runtime and evaluation logs
└── models/       # Saved model parameters/checkpoints
```

The generated outputs are intended to support reconstruction of the tables and figures reported in the paper.

For auditability, retain the task-level outputs and evaluation logs used to produce aggregate metrics rather than storing only the final summary tables.

\---

# 8\. Version Verification

For every Git-based dependency, resolve the exact revision from the actual
checkout:

```bash
git fetch --all --tags
git checkout <PINNED\\\_TAG\\\_OR\\\_COMMIT>
git rev-parse HEAD
```

Record the resulting full SHA in the experiment log.

### AgentBench

```bash
cd third\\\_party/AgentBench
git checkout v0.2
git describe --tags --always
git rev-parse HEAD
```

### WebArena

```bash
cd third\\\_party/webarena
git checkout v0.2.0
git rev-parse HEAD
```

The official release page identifies `v0.2.0` with short commit `e32b71e`.

### SWE-bench

```bash
cd third\\\_party/SWE-bench
git checkout v4.1.0
git rev-parse HEAD
```

The official release page identifies `v4.1.0` with short commit `726c546`.

### GAIA

```bash
python -c "
from huggingface\\\_hub import HfApi

repo = HfApi().dataset\\\_info(
    'gaia-benchmark/GAIA',
    revision='682dd723ee1e1697e00360edccf2366dc8418dd9'
)

print(repo.sha)
"
```

This verification procedure prevents a README from depending on a manually
constructed SHA. Official tags/releases identify the intended benchmark
version, while the actual local checkout determines the exact full revision.

\---

# 9\. Citation

If you use CALIX in your research, please cite the corresponding paper:

```bibtex
@article{calix,
  title   = {CALIX: Closed-loop Adaptive Learning Intelligence for eXplainable Multi-Agent Systems},
  author  = {Author Names},
  journal = {Journal/Conference Name},
  year    = {2026}
}
```

\---

# 10\. License

Please refer to the repository license file for the applicable terms of use.

Benchmark datasets, benchmark environments, and third-party baseline frameworks remain subject to their respective licenses and access conditions. Follow the official terms for each external resource.

