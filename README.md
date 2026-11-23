# CALIX: A Closed-Loop Adaptive Learning Framework for Traceable LLM-Driven Multi-Agent Collaboration

Repository: **https://github.com/EverythingWillFlow/CALIX.git**

| Item | Verified value |
| - | - |
| Python | 3.10 (`.venv`) |
| Python packages | numpy 2.5.4, matplotlib 3.11.2, PyYAML 6.0.3, pyarrow 26.0.0 |
| Container runtime (optional) | Colima + Docker (`--vm-type vz --vz-rosetta`) |
| SWE-bench harness | `swebench==2.1.0`, run in a container |
| LLM endpoint | any OpenAI-compatible `/v1/chat/completions` gateway |
| Unit tests | `python -m unittest discover -s tests` → 34 passed |

Reproducible reference implementation of the paper:

> **CALIX: Closed-loop Adaptive Learning Intelligence for eXplainable Multi-Agent Systems**

## Status: what runs today

| Benchmark | Requires | Status |
| - | - | - |
| GAIA (validation split, 40 tasks) | real LLM endpoint | ✅ runs |
| SWE-bench (40 tasks) | Docker + official images | ✅ runs |
| AgentBench v0.2 — dbbench (50 tasks) | Docker (any CPU arch) | ✅ runs |
| WebArena v0.2.0 (40 tasks) | upstream prebuilt site images | ✅ runs |

AgentBench data comes from the pinned upstream tag `v0.2`, rebuilt with
`python data/prepare_agentbench.py` .
Every task carries its official gold; nothing is scored by a local guess:

- **dbbench** — table rebuilt exactly like the official `build_init_sql` (one
  database per task, all columns TEXT, rows inserted verbatim); `Action:
  Operation` executes one statement in a stock `mysql:8.0` container and
  returns raw output; INSERT/UPDATE/DELETE scored with the official md5-of-rows
  query against `answer_md5`, else against `label`. Verified: gold SQL
  reproduces `answer_md5`, a no-op submission does not.
- **knowledgegraph** — official environment answers `get_relations` /
  `get_neighbors` against a Freebase graph served by Virtuoso (not
  redistributable), so the agent uses its own reasoning and is scored against
  the official gold. Every record carries `protocol:
  gold_answer_scoring_no_graphdb`; this deviation must be stated in the paper.

All nine baselines are implemented in-repo as faithful in-house
reimplementations (paper Table 5: "in-house (repo)"), all driven by the same
LLM client, web toolkit and budget: `gpt4o_direct`, `react`, `camel`,
`autogen`, `agentverse`, `metagpt`, `chatdev`, `sweagent`, `openhands`.

### Remote runtime (e.g. Colab)

AgentBench, WebArena and SWE-bench need Docker; nothing here emulates them.
Start the official runtime on a Docker-capable machine and point CALIX at a bridge:

```bash
# Colab side: bridge + tunnel, prints the exports for the Mac
python tools/colab_serve.py --tunnel cloudflared \
    --set agentbench=agentbench --set webarena=webarena \
    --set swebench=swebench_shell

# Mac side (paste what it printed, then verify before spending)
export CALIX_EXECUTOR_SWEBENCH=https://<tunnel>/swebench
export CALIX_BRIDGE_TOKEN=<printed token>
python tools/remote_check.py
python experiments/run_benchmarks.py --config configs/colab.yaml \
    --benchmark gaia swebench --eval-tasks 15 --workers 8
```

### Local runtime (Apple Silicon)

```bash
brew install colima docker
colima start --vm-type vz --vz-rosetta --arch aarch64 --cpu 8 --memory 16 --disk 200

export CALIX_DOCKER_PLATFORM=linux/amd64     # official images are x86_64
python tools/executor_bridge.py --host 127.0.0.1 --port 8000 --set swebench=swebench_shell
export CALIX_EXECUTOR_SWEBENCH=http://127.0.0.1:8000/swebench
python tools/remote_check.py
```

The bridge serves `POST /<bench>/reset`, `POST /<bench>/step`,
`POST /<bench>/shell` (SWE-agent / OpenHands need a real shell) and
`GET /health`. When the runtime is missing or an adapter is unwired it answers
**HTTP 501** with the concrete install command; it never returns a fabricated
observation or score. Address resolves from `$CALIX_EXECUTOR_<NAME>`, then
`benchmarks.executors.<name>`, then the task file's `executor_url`.

`run_dynamic.py` : the official benchmarks are static,
so no genuine dynamic-environment robustness measurement exists.

---

## 0. Credentials (do this first)

CALIX talks to a real OpenAI-compatible chat-completions endpoint. No mock mode:
without credentials every experiment script fails immediately.

```bash
cp .env.example .env
open -e .env              # or: nano .env
```

File (git-ignored, never committed): **`/Users/sara/calix/CALIX/.env`**

```dotenv
CALIX_OPENAI_API_KEY=sk-...                       # required
CALIX_OPENAI_BASE_URL=https://api.openai.com/v1   # must end in /v1
# CALIX_MODEL=gpt-4o                              # optional, overrides configs/*.yaml
```

Verify before spending (one `max_tokens=1` call):

```bash
python tools/check_key.py
python tools/check_key.py --list-models   # if your gateway uses other model names
```

Notes:
- `CALIX_OPENAI_API_KEY` falls back to `OPENAI_API_KEY`; `CALIX_OPENAI_BASE_URL`
  falls back to `OPENAI_BASE_URL`. Real env vars override `.env`.
- `.env` is loaded automatically by `calix/env.py::load_dotenv`; no
  `python-dotenv` dependency needed.
- Never paste a key into chat, an issue, or a commit. Revoke it if it leaks.

---

## 1. Project Structure

```text
CALIX/
├── calix/                  # Core algorithm package
│   ├── calix.py            # Closed-loop main loop (Algorithm 1)
│   ├── state.py            # Behavior-aware state encoder phi_xi (Algorithm 2)
│   ├── graph.py            # Dynamic interaction graph G_t and Phi_G
│   ├── memory.py           # Historical memory H_t, Top-k retrieval, Phi_H
│   ├── policy.py           # Candidate policies and psi(.)
│   ├── qnetwork.py         # Q-network Q_w, target net, Adam
│   ├── backend.py          # CUDA 12.2 via CuPy, CPU fallback
│   ├── replay.py           # Experience replay buffer D
│   ├── llm.py              # g_LLM: OpenAI-compatible client
│   ├── agents.py           # Six-role multi-agent execution and fusion
│   ├── reward.py           # r = alpha*r_task + beta*r_eff + lambda*r_coord
│   ├── environment.py      # Benchmark environment E(x_t, y_t)
│   ├── provenance.py       # Decision-provenance + PC/DF/TS metrics
│   ├── embedding.py        # Deterministic 768-d text embedding
│   └── baselines.py        # Executors for the nine baselines
├── configs/                # default.yaml (+ pilot/colab/etc.)
├── data/                   # prepare_data.py, prepare_*.py, benchmarks/
├── prompts/                # Version-controlled prompt templates + agents/
├── experiments/            # 1:1 with paper tables/figures (run_main, run_rl_curve,
│                           #   run_ablation, run_agent_ablation, run_sensitivity,
│                           #   run_dynamic, run_provenance, run_scalability,
│                           #   run_failure, run_all)
├── output/                 # tables/ figures/ traces/ logs/ models/
├── tests/test_calix.py     # embedding, Q-net, reward, env, main loop
├── requirements.txt
└── README.md
```

---

## 2. Dependencies and Installation

**Required:**
- Python >= 3.10 (verified 3.13.12)
- `numpy >= 1.26` (2.5.4), `matplotlib >= 3.8` (3.11.2), `PyYAML >= 6.0` (6.0.3)
- `huggingface_hub >= 0.24`, `datasets >= 2.20`, `pyarrow` (benchmark prep)

**Optional (Docker benchmarks):** Colima + Docker (Apple Silicon:
`--vm-type vz --vz-rosetta`); Linux/x86_64 runs official images natively.
`swebench==2.1.0` for the official harness (containerized). Newer `swebench`
releases abort with `KeyError: 'image'` against `princeton-nlp/SWE-bench`; pin 2.1.0.

**GPU (optional):** `cupy-cuda12x >= 13.0` + CUDA 12.2 driver. Backend is
controlled by `device` in `configs/default.yaml`: `auto` (GPU if available else
CPU), `cuda` (require GPU), `cpu` (force NumPy). When CuPy/CUDA is absent it
falls back to NumPy with identical semantics.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # add -i https://pypi.tuna.tsinghua.edu.cn/simple if needed
```

`cupy-cuda12x` is skipped automatically on macOS.

---

## 3. Benchmark Setup

Benchmark versions are pinned to explicit official tags/revisions; no invented
or hand-expanded commit IDs. Record the exact local revision with
`git rev-parse HEAD`.

### AgentBench — `v0.2`
CALIX uses original AgentBench v0.2, not the newer FC version (the official
repo moved `main` to FC). The older implementation is reachable via the `v0.2`
tag.

```bash
git clone https://github.com/THUDM/AgentBench.git third_party/AgentBench
cd third_party/AgentBench && git checkout v0.2
git rev-parse HEAD          # record in the experiment log
```

Do not use the current `main` branch — it is the FC protocol, a different
benchmark from v0.2.

### GAIA — `gaia-benchmark/GAIA`
Gated Hugging Face dataset (Parquet splits). Set `HF_TOKEN`, then download the
pinned revision:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('gaia-benchmark/GAIA', repo_type='dataset', \
revision='682dd723ee1e1697e00360edccf2366dc8418dd9')"
```

Do not redistribute the validation/test split in a crawlable repo.

### WebArena — `v0.2.0`
```bash
git clone https://github.com/web-arena-x/webarena.git third_party/webarena
cd third_party/webarena && git checkout v0.2.0
git rev-parse HEAD           # official release commit: e32b71e
```

The site containers are prebuilt image tarballs only; `v0.2.0` ships no
`setup.sh`/`docker-compose.yml`/Dockerfile for the five sites.
`tools/webarena_local.sh` loads them with the pinned upstream commands and
**refuses to invent a site** when a tarball is missing:

```bash
export WEBARENA_TAR_DIR=/path/to/tarballs
bash tools/webarena_local.sh --site all      # load + start + health check
bash tools/webarena_local.sh --status
```

Until tarballs are obtained, WebArena tasks are recorded `unavailable` — never
silently skipped or counted as a 0/1.

Local self-hosting: the same images are on Docker Hub under `webarenaimages`;
`tools/webarena_local.sh --site all` pulls by name. Then:

```bash
python tools/webarena_prepare.py --exclude-sites map wikipedia --sync-dataset
python tools/verify_webarena.py          # needs the separate wa-venv below
```

`browser_env` needs WebArena's own pinned stack (Playwright 1.32.1, gymnasium,
beartype 0.12, `openai<1`), conflicting with the main env — use a separate
`uv` venv:

```bash
uv venv --python 3.11 wa-venv
uv pip install --python wa-venv/bin/python playwright==1.32.1 gymnasium Pillow \
    tiktoken lxml "beartype==0.12.0" flask nltk aiolimiter "openai<1" \
    text-generation evaluate types-tqdm requests
wa-venv/bin/playwright install chromium
wa-venv/bin/python tools/executor_bridge.py --port 8000 \
    --set agentbench=agentbench_dbbench --set webarena=webarena \
    --set swebench=swebench_shell
```

Apple-Silicon fixes applied automatically by `webarena_local.sh`:
- **PHP 8.1 SIGSEGV under Rosetta** (PCRE JIT) → writes `pcre.jit=0` +
  `opcache.enable=0` and restarts `php-fpm`.
- **Magento base URL in DB** → rewrites secure/unsecure base URL to the local port.
- **Base-URL UPDATE race** → re-runs until both rows carry the local URL.

`map` and `wikipedia` have no released image; their 5 tasks are excluded and
listed in `webarena.excluded.json`. 

### SWE-bench — `v4.1.0`
```bash
pip install swebench==4.1.0
python -c "from datasets import load_dataset; \
ds = load_dataset('princeton-nlp/SWE-bench', split='test'); \
ds.to_json('data/benchmarks/swebench_test.jsonl')"
python -m swebench.harness.prepare_images --dataset princeton-nlp/SWE-bench
```

Official repo `SWE-bench/SWE-bench`, release tag `v4.1.0` (commit `726c546`).
Record both the dataset revision and the harness version in the log.

### Docker startup (macOS / Linux / Colab share one bridge)
```bash
brew install colima docker
colima start --vm-type vz --vz-rosetta --arch aarch64 --cpu 8 --memory 16 --disk 200
export CALIX_DOCKER_PLATFORM=linux/amd64

python tools/executor_bridge.py --host 127.0.0.1 --port 8000 \
    --set agentbench=agentbench_dbbench --set webarena=webarena \
    --set swebench=swebench_shell

export CALIX_EXECUTOR_AGENTBENCH=http://127.0.0.1:8000/agentbench
export CALIX_EXECUTOR_WEBARENA=http://127.0.0.1:8000/webarena
export CALIX_EXECUTOR_SWEBENCH=http://127.0.0.1:8000/swebench
python tools/remote_check.py
```

| Benchmark | Runtime | Status |
| - | - | - |
| GAIA (40) | none (local official normalization) | ✅ runs |
| SWE-bench (40) | per-instance image + official harness | ✅ runs |
| AgentBench dbbench (50) | `mysql:8.0` + official data | ✅ runs, built & scored |
| WebArena (40) | official images + official `evaluator_router` | ✅ runs |

SWE-bench instance images: `swebench/sweb.eval.x86_64.<id with "__" → "_1776_">`.
Reserve ~4 GB per instance (~165 GB for 40); the official project recommends ≥ 120 GB free.

### API Key Configuration
```bash
export OPENAI_API_KEY=<your-openai-key>
export OPENAI_BASE_URL=https://api.openai.com/v1
export HF_TOKEN=<your-huggingface-token>
# OpenHands:
export OPENHANDS_CONFIG=<path-to-config.toml>
```
Paper model: `gpt-4o-2024-05-13`. All methods use the same model and decoding params.

### Official Evaluator Invocation
| Benchmark | Evaluator | Command |
| - | - | - |
| AgentBench | Official workers | `python -m agentbench.evaluate --task <id> --trajectory <traj.json>` |
| GAIA | Official validator | `python -m gaia_official_benchmark.validate --answer <answer> --task <id>` |
| WebArena | `evaluation_harness` | `python -m evaluation_harness.evaluate --config <c> --trajectory <traj.json>` |
| SWE-bench | `swebench.harness` | `python -m swebench.harness.run_evaluation --predictions_path <p> --instance_ids <id>` |

Native success outcomes map to the unified **Task Success Rate (TSR)**.

### Baseline Versions
| Baseline | Repo / Provider | Version |
| - | - | - |
| GPT-4o Direct | OpenAI API | gpt-4o-2024-05-13 |
| ReAct | `ysymyth/ReAct` | pin a stable commit, record full SHA |
| CAMEL | `camel-ai/camel` | v0.2.91a0 |
| AutoGen | `microsoft/autogen` | v0.7.2 |
| AgentVerse | `OpenBMB/AgentVerse` | v0.1.8.1 |
| MetaGPT | `geekan/MetaGPT` | v0.8.0 |
| ChatDev | `OpenBMB/ChatDev` | v1.1.6 |
| SWE-agent | `princeton-nlp/SWE-agent` | v1.1.0 |
| OpenHands | `All-Hands-AI/OpenHands` | v1.21.0 |

```bash
pip install camel-ai==0.2.91a0 autogen==0.7.2 agentverse==0.1.8.1 metagpt==0.8.0
git clone https://github.com/OpenBMB/ChatDev.git third_party/ChatDev && (cd third_party/ChatDev && git checkout v1.1.6)
git clone https://github.com/SWE-agent/SWE-agent.git third_party/SWE-agent && (cd third_party/SWE-agent && git checkout v1.1.0)
git clone https://github.com/All-Hands-AI/OpenHands.git third_party/OpenHands && (cd third_party/OpenHands && git checkout v1.21.0)
```

SWE-bench scoring uses the official harness in a container:
```bash
bash tools/swebench_harness_container.sh <instance_id> predictions.jsonl
# -> output/swebench/<model>.calix_<instance>.json
```
The script pins `swebench==2.1.0`; override with `SWEBENCH_VERSION=...`.
For cloned benchmark packages: `pip install -e third_party/webarena third_party/AgentBench`.

---

## 4. Running the Evaluation

### 4.0 Credentials
```bash
export CALIX_OPENAI_API_KEY=<your-key>
export CALIX_OPENAI_BASE_URL=https://<your-endpoint>/v1
```
`.env` is git-ignored; without a key every script fails immediately, no offline/synthetic fallback.

### 4.1 Smoke test
```bash
python experiments/run_smoke.py --config configs/pilot.yaml --tasks 1
```
One real CALIX episode + one task per baseline, prints actual output and verdict, extrapolates cost.

### 4.2 Pilot and matrix
```bash
python experiments/run_benchmarks.py --config configs/pilot.yaml \
    --benchmark gaia --seeds 2025 --methods gpt4o_direct react calix --workers 8
# all 9 baselines on GAIA (~1.5–2 h, 8 workers):
python experiments/run_benchmarks.py --config configs/colab.yaml \
    --benchmark gaia --eval-tasks 15 --workers 8
# full matrix (4×9), skip unwired runtimes instead of aborting:
python experiments/run_benchmarks.py --config configs/colab.yaml \
    --benchmark gaia agentbench webarena swebench --eval-tasks 1 \
    --episodes 0 --workers 3 --skip-unavailable
```

Output: `output/<run>/tables/real_results.csv`. A row `status = unavailable: ...`
means the runtime was missing and **no number was invented**. Render as a paper
table (unavailable → `–`, never `0`):
```bash
python tools/matrix_table.py        # -> output/<run>/tables/real_results.md
```
Re-run a single benchmark and keep the rest with `--merge-results`.

### 4.3 Positive controls (does the matrix prove anything?)
An all-zero matrix is indistinguishable from a scorer that always returns 0.
```bash
python tools/verify_scoring.py
```
Each control submits a known-good input (gold answer / reference SQL+label /
reference URL+stop) and requires success. For SWE-bench, the harness report
itself is the evidence: `completed_instances: 1` with `error_ids: []` means the
container built and tests ran. A non-empty `error_ids` is an *environment* error
and must never be counted as a 0. A 1-task smoke matrix is a plumbing check,
not a result — fill Table 7 from full eval pools.

### 4.4 Measurement integrity — defects found by auditing runs
Each was invisible in the score and only showed in provenance:

| # | Defect | Guard |
|---|---|---|
| 1 | Each runtime only recognizes its own terminal syntax; CALIX graded 0 for phrasing | `RealEnvironment.submission_protocol()` injected into observation + fusion prompt |
| 2 | Bridge answered `/reset` with no `ok`; client read `data.get("ok", False)` | absence of `error` + presence of `observation` ⇒ ready; 2 regression tests |
| 3 | One task raising aborted the whole cell, wrote zero rows | per-task isolation; instance recorded `aborted: true` |
| 4 | Infrastructure outage counted as `success: false` | `matrix_audit.py` excludes `aborted` rows |
| 5 | Duplicated `(method, task)` rows; older won de-dup | `(mtime, line)` comparison; later line = later run |
| 6 | Summary rewritten from the current invocation only | skipped cells recomputed from task records; `rebuild_summary.py` |
| 7 | Bookkeeping rows tagged `index-<i>` hid gaps | every row carries the real `task_id`; regression test asserts ids |

The task records are the source of truth; the summary CSV is a view of them.
Transcription policy: missing capability → `n/a`, present-but-unused → `0.0`,
missing runtime → `–`. When *every* method scores 0 on a benchmark, suspect the
measurement first.

### 4.5 Resumable runs (a kill loses at most one task)
- **Per-task persistence** — `common.make_record_appender` fsyncs one line to
  `task_records.jsonl` as each task finishes.
- **Goal-line resume** — `run_benchmarks.py --resume` runs only `(method, task)`
  cells without a graded row; `--eval-tasks` is the goal line, not new work.
  Re-running a finished cell returns in ~1 s with zero new rows.

`tools/resume_gaia.sh` wraps both into an idempotent entry point.
`tools/resume_bridge_benchs.sh` does the same for Docker benchmarks (bridge on
`127.0.0.1:8010`). Pitfalls: the agentbench backend must be `agentbench_dbbench`
(not bare `agentbench`); an ambient `http_proxy` hijacks `127.0.0.1` probes — use
`curl --noproxy '*'` / `NO_PROXY`.

### 4.6 A refusal is not a downtime
Terminal failures (401/402/403) stop the run and keep what was measured;
transient (SSL EOF, 429, timeout) wait and retry. `calix/llm.py` raises
`FatalEndpointError` for 401/402/403; the start gate refuses to run (zero calls,
one line to top up), `_retry_aborted` gives up, remaining methods are skipped,
`real_results.csv` writes `incomplete: N/M aborted`, and the process exits
non-zero.

### 4.7 Auditing the tables
```bash
.venv/bin/python tools/table_audit.py          # -> output/tables/PROVENANCE_AUDIT.md
```
Per-cell verdicts: `BACKED` (record + agrees + complete), `INCOMPLETE`,
`CONTRADICTED`, `UNSOURCED`. Exits non-zero unless every cell is `BACKED`.
Catches (a) tables with no producer — e.g. `dynamic_robustness.md` while
`run_dynamic.py` raises `NotImplementedError` by design: remove, don't backfill;
(b) transcribed deviations — a headline TSR with identical mean but drifting std
across tables indicates manual entry, not independent runs.

```bash
.venv/bin/python tools/paper_tables.py
```
Regenerates (not audits) tables from records: `overall_results.md` (rate + 95%
Wilson interval + task count), `computational_cost.md`, and
`appendix_task_records.csv` (the per-task file a reviewer recomputes from).
Both quote that CSV's SHA-256, the commit, and the graded-task count in their
footer. De-dup by `(benchmark, method, task_id)`, keeping the later measurement.
Running the audit after regeneration takes verdicts to 10/12 `BACKED`, 0
contradicted, 0 unsourced (the 2 `INCOMPLETE` are real gaps).



### 4.8 Statistics that match the sample
Tables used to report `mean ± std` as if across seeds. Now:
- `common.mean_std()` refuses fewer than two values (no fake `0.00`).
- `common.mean_ci(successes, n)` renders a Wilson score interval
  (`52.73 [39.79, 65.31]`), used because several cells are small (n=11) or near 0%.

### 4.9 Process manifest
`common.write_manifest()` writes `output/<run>/manifests/latest.json` per process:
`git.commit`/`clean`/`dirty_files`, `code_fingerprint` (SHA-256 over `calix/` +
`experiments/`), `config.path`/`sha256`, `python.version`/`numpy`/`platform`,
`llm.base_url`/`model`/`key_sha256_12`. The API key is never written — only a
12-hex digest, so a manifest can be published while proving which account ran it.

### 4.10 Full paper protocol
```bash
python experiments/run_all.py --config configs/default.yaml
# -> output/tables/real_results.csv, output/traces/task_records.jsonl
```
Before treating results as final: benchmark revision matches the pin; baseline
versions/commits fixed and recorded; same model/decoding across methods; same
task subset/split per method; logs and task-level results retained.

---

## 5. One-Command Reproduction
```bash
python experiments/run_all.py
```
Scripts run independently; existing outputs are not auto-recomputed, so long
evaluations can be split across sessions.

```bash
python experiments/run_main.py          # main comparison + learning-curve data
python experiments/run_rl_curve.py      # RL learning curve
python experiments/run_ablation.py --episodes 200   # component ablation
python experiments/run_agent_ablation.py            # leave-one-agent-out
python experiments/run_sensitivity.py                # lr / gamma / tau / K
python experiments/run_dynamic.py                   # dynamic robustness 
python experiments/run_provenance.py                # PC / DF / TS
python experiments/run_scalability.py               # cost / scalability
python experiments/run_failure.py                   # failure + recovery
```

---

## 6. Reproducibility Protocol
For every final run, create a machine-readable environment record with at least:
`benchmark`, `benchmark_tag_or_revision`, `resolved_commit_sha`,
`baseline_name`, `baseline_version_or_commit`, `model`, `seed`, `task_split`,
`evaluator_version`, `timestamp`. For Git deps copy `resolved_commit_sha` from
`git rev-parse HEAD`; for packages record the installed version and checksum.

The protocol fixes: benchmark repo/dataset id; revision/tag; task split;
prompt + role templates; LLM model + decoding; tools + interaction limits;
seeds; baseline versions/commits; preprocessing; evaluator config; output
locations; generated tables/figures/traces/logs. A result is not
version-controlled while any benchmark/baseline runs off a moving `main` branch.

### Statistics
```bash
python tools/records_to_runs.py
python tools/check_results.py --runs output/traces/runs.csv \
    --target CALIX --vs gpt4o_direct react
```
`check_results.py --audit` verifies every reported TSR is an admissible
`successes / total`.

---

## 7. Output Files
```text
output/
├── tables/    # CSV / MD experiment results
├── figures/   # plots and figures
├── traces/    # agent execution + decision traces (task_records.jsonl)
├── logs/      # runtime + evaluation logs
└── models/    # saved checkpoints
```
Retain task-level outputs and logs, not just summary tables, for auditability.

---

## 8. Version Verification
For every Git dependency:
```bash
git fetch --all --tags
git checkout <PINNED_TAG_OR_COMMIT>
git rev-parse HEAD          # record the full SHA
```

- **AgentBench** — `THUDM/AgentBench`, tag `v0.2`.
- **WebArena** — `web-arena-x/webarena`, tag `v0.2.0` (commit `e32b71e`).
- **SWE-bench** — `princeton-nlp/SWE-bench` (test split), harness `swebench==2.1.0`,
  images pinned per instance by the harness.
- **GAIA** — `gaia-benchmark/GAIA`, revision
  `682dd723ee1e1697e00360edccf2366dc8418dd9`.

This repo:
```bash
git clone https://github.com/EverythingWillFlow/CALIX.git
cd CALIX && git rev-parse HEAD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests     # 34 tests, all must pass
```

The README never depends on a manually constructed SHA: tags identify the
intended version, the local checkout determines the exact revision.

---

## 9. Citation
```bibtex
@article{calix,
  title   = {CALIX: A Closed-Loop Adaptive Learning Framework for Traceable LLM-Driven Multi-Agent Collaboration},
  author  = {Author Names},
  journal = {Journal/Conference Name},
  year    = {2026}
}
```

---


