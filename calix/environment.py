"""Real benchmark environments for CALIX (no simulation).

Unified interface consumed by the CALIX closed loop:

    obs = env.reset(task_index)                 -> {"task_id": str, "text": str}
    obs, info, done = env.step(action_text)     -> next observation dict,
                                                   info (with "success" /
                                                   "step_success"), done flag

Environments
------------
GAIAEnvironment
    Real GAIA question answering scored with the official GAIA answer
    normalisation and exact-match rule.  Requires tasks that carry a gold
    ``answer`` field (i.e. the *validation* split; the test split has no
    public answers and cannot be scored locally).

AgentBenchHTTPEnvironment / WebArenaHTTPEnvironment
    Talk to the official benchmark runtime over HTTP.  The official
    AgentBench v0.2 task services and WebArena v0.2.0 Docker sites must be
    running locally (see README section 3).  A thin bridge service exposes
    them through the per-task ``executor_url`` recorded in
    ``data/benchmarks/<name>.json``.

SWEBenchEnvironment
    Collects the model-generated patch and, at episode end, has it scored by
    the official ``swebench.harness.run_evaluation``.  If a bridge is
    configured (``$CALIX_EXECUTOR_SWEBENCH``) the harness runs on the machine
    that owns Docker; the local invocation is only a fallback.

No environment in this module fabricates observations, rewards, or success
signals.  When the required runtime is unavailable the environment raises a
clear error instead of returning synthetic feedback.
"""
from __future__ import annotations

import json
import os
import re
import string
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


# --------------------------------------------------------------------------- #
#  GAIA official answer scoring (mirrors the public GAIA question_scorer)
# --------------------------------------------------------------------------- #
def _normalize_number_str(number_str: str) -> str:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return str(float(number_str))
    except ValueError:
        return number_str


def _split_string(s: str, char_list: list[str]) -> list[str]:
    pattern = f"[{''.join(re.escape(c) for c in char_list)}]"
    return [part.strip() for part in re.split(pattern, s) if part.strip()]


def _normalize_str(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    for article in ("a", "an", "the"):
        text = re.sub(rf"\b{article}\b", " ", text)
    return " ".join(text.split())


def gaia_exact_match(answer: str, gold: str) -> bool:
    """Official GAIA scoring rule (comma lists, numbers, normalised strings)."""
    if not answer or not gold:
        return False
    answer = answer.strip()
    gold = gold.strip()
    if "," in gold:
        gold_parts = _split_string(gold, [","])
        answer_parts = _split_string(answer, [",", ";"])
        if len(gold_parts) != len(answer_parts):
            return False
        return sorted(_normalize_str(p) for p in gold_parts) == \
            sorted(_normalize_str(p) for p in answer_parts)
    gold_norm = _normalize_number_str(gold)
    ans_norm = _normalize_number_str(answer)
    try:
        return float(ans_norm) == float(gold_norm)
    except ValueError:
        return _normalize_str(ans_norm) == _normalize_str(gold_norm)


# --------------------------------------------------------------------------- #
#  Base class
# --------------------------------------------------------------------------- #
class RealEnvironment(ABC):
    def __init__(self, tasks: list[dict], max_steps: int):
        self.tasks = tasks
        self.max_steps = max_steps
        self.task: dict[str, Any] | None = None

    @abstractmethod
    def reset(self, task_index: int) -> dict:
        ...

    @abstractmethod
    def step(self, action: str) -> tuple[dict, dict, bool]:
        """Return (observation, info, done). ``info['success']`` is the
        terminal benchmark success flag, available when ``done`` is True."""
        ...

    def shell(self, command: str) -> str:
        """Execute ``command`` inside the benchmark's own sandbox.

        Only execution-oriented environments (SWE-bench, AgentBench, WebArena)
        provide one.  Those without a sandbox raise ``NotImplementedError``;
        callers must record that fact instead of inventing command output.
        """
        raise NotImplementedError(
            f"The {self.name} environment provides no execution sandbox, so "
            "shell/code actions cannot be executed here.")

    def submission_protocol(self) -> str:
        """How this runtime expects a terminal answer to be phrased.

        Every official runtime only recognises its own submission syntax: the
        AgentBench bridge scores nothing unless the reply ends with
        ``Final Answer: [...]``, WebArena needs ``stop [answer]``, SWE-bench
        grades a ```diff fence.  Baseline templates spell that out in their own
        prompt (``prompts/baselines/*.txt``); the same sentence has to reach a
        method whose output is produced by a fusion module, or the 0.0 it
        records means "the aggregation step never learned the protocol" rather
        than "the question was too hard".  Returns "" when the runtime accepts
        plain text (GAIA).
        """
        return ""

    def __len__(self) -> int:
        return len(self.tasks)


# --------------------------------------------------------------------------- #
#  GAIA (no external runtime required; needs gold answers = validation split)
# --------------------------------------------------------------------------- #
class GAIAEnvironment(RealEnvironment):
    name = "gaia"

    def reset(self, task_index: int) -> dict:
        self.task = self.tasks[task_index % len(self.tasks)]
        if not self.task.get("answer"):
            raise ValueError(
                f"GAIA task {self.task.get('task_id')} has no gold answer. "
                "The test split cannot be scored locally; rebuild the task "
                "file from the validation split (see data/prepare_data.py)."
            )
        self._attempts = 0
        self._success = False
        return {"task_id": self.task["task_id"], "text": self.task["question"]}

    def step(self, action: str) -> tuple[dict, dict, bool]:
        self._attempts += 1
        final = extract_final_answer(action)
        correct = gaia_exact_match(final, self.task["answer"])
        done = correct or self._attempts >= self.max_steps
        self._success = correct
        if correct:
            feedback = "Environment feedback: the submitted answer is correct."
        elif done:
            feedback = ("Environment feedback: the interaction budget is "
                        "exhausted and the answer remains incorrect.")
        else:
            feedback = ("Environment feedback: the submitted answer is "
                        "incorrect. Reconsider the strategy and try again.")
        obs = {"task_id": self.task["task_id"], "text": feedback}
        info = {"success": bool(done and correct), "step_success": correct}
        return obs, info, done


_FINAL_RE = re.compile(
    r"(?:final\s*answer|answer)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: str) -> str:
    """Extract the answer portion of an agent output (``FINAL ANSWER: x``)."""
    matches = _FINAL_RE.findall(text)
    if matches:
        return matches[-1].strip().strip('"').strip()
    return text.strip()


_SWEBENCH_GOLD: dict[str, str] | None = None


def swebench_gold(instance: str) -> str:
    """The official gold patch for one SWE-bench instance, or "".

    It lives in ``data/benchmarks/swebench.gold.jsonl``, deliberately *not* in
    the task objects, because a task object is what gets rendered into a
    prompt: a gold patch inside it would be handed to the very agent it is
    supposed to grade.  Only the audit trail and the scoring control read this.
    """
    global _SWEBENCH_GOLD
    if _SWEBENCH_GOLD is None:
        loaded: dict[str, str] = {}
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data", "benchmarks", "swebench.gold.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    loaded[row.get("instance_id", "")] = row.get("patch", "")
        _SWEBENCH_GOLD = loaded
    key = instance.replace("swebench-", "", 1)
    return _SWEBENCH_GOLD.get(key, "")


def gold_of(task: dict) -> str:
    """Best-effort gold value for the audit trail.

    Gold lives in a different place per benchmark: GAIA on the task itself,
    AgentBench dbbench in ``metadata.label`` (read) or ``metadata.answer_md5``
    (write), knowledgegraph in ``metadata.answer``, WebArena inside
    ``metadata.eval`` / ``metadata.source_fields.eval``.  This is *only* for the
    trace; scoring is always done by the official scorer, never by this string.
    """
    md = task.get("metadata") or {}
    for value in (task.get("answer"), md.get("label"), md.get("answer"),
                  md.get("answer_md5"), md.get("reference_sql")):
        if value in (None, "", [], {}):
            continue
        return str(value)[:200]
    ev = md.get("eval") or (md.get("source_fields") or {}).get("eval") or {}
    if isinstance(ev, dict) and ev:
        for key in ("reference_answers", "reference_url"):
            if ev.get(key):
                return str(ev[key])[:200]
        if ev.get("eval_types"):
            return f"<{','.join(ev['eval_types'])}>"
    tid = str(task.get("task_id", ""))
    if tid.startswith("swebench-"):
        # the gold patch itself is too long for a trace and lives outside the
        # task objects anyway; recording that it exists and how big it is
        # already fixes the empty-gold hole in the SWE-bench rows.
        patch = swebench_gold(tid)
        return f"<gold patch: {len(patch)} chars>" if patch else ""
    if md.get("instance_id"):
        return f"<patch: {md['instance_id']}>"
    return ""


# --------------------------------------------------------------------------- #
#  AgentBench v0.2 / WebArena v0.2.0 via the official runtime over HTTP
# --------------------------------------------------------------------------- #
def _post_json(url: str, payload: dict, timeout: float, label: str) -> dict:
    """POST ``payload`` to the runtime bridge and decode the JSON reply.

    The bridge answers 501 when the backend exposes no sandbox for the
    requested action and 500 with a message for a genuine runtime failure.
    Those two must not be flattened into "the runtime is unreachable": the
    first means *this baseline cannot run on this benchmark*, the second means
    the environment broke.  Only a transport failure is reported as
    unreachable.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", "")
        except Exception:                                    # noqa: BLE001
            detail = ""
        detail = detail or e.reason or "no detail"
        if e.code == 501:
            raise NotImplementedError(f"{label}: {detail}") from e
        raise EnvironmentError(
            f"{label}: the runtime answered HTTP {e.code}: {detail}") from e
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise EnvironmentError(
            f"{label}: runtime not reachable at {url}: {e}") from e


class _OfficialHTTPEnvironment(RealEnvironment):
    """Generic adapter for the official benchmark runtime bridge service.

    The bridge (started together with the official Docker services) must
    accept ``POST <executor_url>/reset`` and ``POST <executor_url>/step``
    and return JSON with ``observation``, ``done`` and ``success`` fields.

    The bridge address is resolved in this order, so a remote runtime (for
    example a Colab VM reachable through a tunnel) can be pointed to without
    editing the task files:

    1. ``$CALIX_EXECUTOR_<BENCHMARK>``   (e.g. CALIX_EXECUTOR_AGENTBENCH)
    2. ``benchmarks.executors.<name>``   in configs/*.yaml
    3. ``executor_url`` recorded in the task file
    """

    timeout = 60

    def __init__(self, tasks: list[dict], max_steps: int,
                 executor_url: str | None = None):
        super().__init__(tasks, max_steps)
        self.executor_override = executor_url or ""

    def _base_url(self) -> str:
        url = (os.environ.get(f"CALIX_EXECUTOR_{self.name.upper()}", "")
               or self.executor_override
               or (self.task or {}).get("executor_url", ""))
        if not url:
            raise EnvironmentError(
                f"No runtime address for {self.name}. Set "
                f"$CALIX_EXECUTOR_{self.name.upper()} (or "
                f"benchmarks.executors.{self.name} in the config) to the "
                f"bridge URL; see tools/executor_bridge.py.")
        return url.rstrip("/")

    def _post(self, url: str, payload: dict) -> dict:
        return _post_json(url, payload, self.timeout, f"Official {self.name}")

    def reset(self, task_index: int) -> dict:
        self.task = self.tasks[task_index % len(self.tasks)]
        url = self._base_url()
        data = self._post(f"{url}/reset",
                          {"task_id": self.task["task_id"]})
        return {"task_id": self.task["task_id"],
                "text": str(data.get("observation", ""))}

    def step(self, action: str) -> tuple[dict, dict, bool]:
        url = self._base_url()
        data = self._post(f"{url}/step",
                          {"task_id": self.task["task_id"], "action": action})
        done = bool(data.get("done", False))
        obs = {"task_id": self.task["task_id"],
               "text": str(data.get("observation", ""))}
        info = {"success": bool(data.get("success", False)) if done else False,
                "step_success": bool(data.get("step_success", False))}
        return obs, info, done

    def shell(self, command: str) -> str:
        """Run ``command`` in the remote runtime (SWE-agent / OpenHands)."""
        url = self._base_url()
        data = self._post(f"{url}/shell",
                          {"task_id": self.task["task_id"], "command": command})
        if not data.get("ok", False):
            raise NotImplementedError(str(data.get("error", "shell refused")))
        return str(data.get("output", ""))


class AgentBenchHTTPEnvironment(_OfficialHTTPEnvironment):
    name = "agentbench"

    def submission_protocol(self) -> str:
        return (
            "Submission format required by this runtime: to operate, reply "
            "with\nAction: Operation\n```sql\n<one statement>\n```\nand when "
            "you are ready to commit the answer, end with exactly\nAction: "
            "Answer\nFinal Answer: [\"ANSWER1\", \"ANSWER2\", ...]\nNothing "
            "else is scored.")


class WebArenaHTTPEnvironment(_OfficialHTTPEnvironment):
    name = "webarena"

    def submission_protocol(self) -> str:
        return (
            "Submission format required by this runtime: emit one browser "
            "action per turn (``click [id]``, ``type [id] [text]``, "
            "``goto [url]``, ``scroll [direction]``), and when you want to "
            "submit, reply with exactly\nstop [<the answer, or nothing when "
            "the task is about reaching a page state>]\nNothing else ends the "
            "episode.")


# --------------------------------------------------------------------------- #
#  SWE-bench v4.1.0 (official harness; requires Docker)
# --------------------------------------------------------------------------- #
class SWEBenchEnvironment(RealEnvironment):
    name = "swebench"

    # an instance image is ~4 GB, so the first reset of an instance can be a
    # full docker pull; the bridge gives up after 1800 s on its own.
    reset_timeout = 1800

    def __init__(self, tasks: list[dict], max_steps: int,
                 executor_url: str | None = None):
        super().__init__(tasks, max_steps)
        # the bridge that owns the Docker container for this instance
        self.executor_override = executor_url or ""
        self._container_note = ""
        self._container_ready = False
        self._container_error = ""

    def _bridge_url(self) -> str:
        url = (os.environ.get("CALIX_EXECUTOR_SWEBENCH", "")
               or self.executor_override
               or (self.task or {}).get("executor_url", ""))
        if not url:
            raise NotImplementedError(
                "No SWE-bench runtime bridge is configured, so no shell is "
                "available. Set $CALIX_EXECUTOR_SWEBENCH (see "
                "tools/executor_bridge.py); without it the environment can "
                "only score a final patch through the official harness.")
        return url.rstrip("/")

    def shell(self, command: str) -> str:
        """Execute ``command`` inside the instance container (SWE-agent style)."""
        if not self._container_ready:
            # NotImplementedError is the signal the baselines catch to fall
            # back to "no shell for you"; anything else would abort the run.
            raise NotImplementedError(
                f"no container is running for {self.task['task_id']}"
                + (f" ({self._container_error})" if self._container_error
                   else ""))
        data = _post_json(f"{self._bridge_url()}/shell",
                          {"task_id": self.task["task_id"],
                           "command": command}, 300, "SWE-bench")
        if not data.get("ok", False):
            raise NotImplementedError(str(data.get("error", "shell refused")))
        return str(data.get("output", ""))

    def reset(self, task_index: int) -> dict:
        self.task = self.tasks[task_index % len(self.tasks)]
        sf = self.task.get("metadata", {}).get("source_fields", {})
        self._problem = sf.get("problem_statement") or self.task.get(
            "initial_observation", "")
        self._patch = ""
        self._attempts = 0
        self._last_harness_note = ""
        self._container_note = ""
        # The container has to exist before any shell command can run. Doing it
        # here (rather than lazily on the first shell call) is also what turns
        # "this instance's image is unavailable" into a loud failure: without
        # this reset, every method silently falls back to writing a patch blind
        # and the resulting 0.0 is indistinguishable from a real attempt.
        url = self._bridge_url_or_none()
        if url:
            try:
                data = _post_json(f"{url}/reset",
                                  {"task_id": self.task["task_id"]},
                                  self.reset_timeout, "SWE-bench")
                # The bridge answers {"observation", "done"} here (see the
                # protocol comment at the top of executor_bridge.py); there is
                # no "ok" field on /reset, only an "error" field on failure.
                # Checking a missing "ok" made *every* instance look like a
                # container failure, so every method was silently scored on a
                # patch it had written blind -- exactly the hole this flag
                # exists to expose. (Regression test: test_swebench_reset_uses_bridge)
                if data.get("error"):
                    raise EnvironmentError(str(data["error"])[:300])
                self._container_note = str(data.get("observation", "")).strip()
                self._container_ready = bool(self._container_note)
            except (EnvironmentError, OSError) as exc:
                # No container means no shell, so the instance is measured
                # *without repository exploration*. It is still measured: the
                # official harness builds from the local env image and grades
                # the submitted patch. Reporting the cell as `unavailable`
                # here would throw away a real score, and reporting it as a
                # plain 0.0 would hide that exploration was impossible -- so
                # record which it was and let the table say so.
                self._container_note = ""
                self._container_ready = False
                self._container_error = f"{type(exc).__name__}: {exc}"
        text = (f"Repository: {sf.get('repo', '?')} (base {sf.get('base_commit', '?')})\n\n"
                f"Problem statement:\n{self._problem}\n\n"
                "Produce a unified-diff patch that resolves the issue.")
        if self._container_note:
            text = f"{self._container_note}\n\n{text}"
        return {"task_id": self.task["task_id"], "text": text}

    def step(self, action: str) -> tuple[dict, dict, bool]:
        self._attempts += 1
        patch = extract_patch(action)
        if patch:
            self._patch = patch
        done = bool(patch) or self._attempts >= self.max_steps
        if not done:
            obs = {"task_id": self.task["task_id"],
                   "text": ("Environment feedback: no unified-diff patch was "
                            "found in your last output. Provide the patch in a "
                            "```diff fenced block.")}
            return obs, {"success": False, "step_success": False}, False
        success = False
        if self._patch:
            success = self._run_official_harness(self._patch)
        obs = {"task_id": self.task["task_id"],
               "text": ("Environment feedback: patch evaluated by the official "
                        f"harness. {self._last_harness_note}").strip()
               if self._patch else
               "Environment feedback: interaction budget exhausted without a patch."}
        return obs, {"success": success, "step_success": success}, True

    def submission_protocol(self) -> str:
        return (
            "Submission format required by this runtime: put the whole "
            "solution in one fenced unified diff, i.e.\n```diff\n<"
            "unified diff>\n```\nOnly that block is graded by the official "
            "harness.")

    def _bridge_url_or_none(self) -> str:
        return (os.environ.get("CALIX_EXECUTOR_SWEBENCH", "")
                or self.executor_override
                or (self.task or {}).get("executor_url", "")
                or "").rstrip("/")

    def _run_official_harness(self, patch: str) -> bool:
        """Score ``patch`` with the official harness.

        Preference order matters: if a bridge is configured, the harness runs
        on the machine that owns Docker (the bridge container), so this process
        never needs the ``swebench`` package.  Only when no bridge is set do we
        fall back to invoking the harness locally.
        """
        bridge = self._bridge_url_or_none()
        if bridge:
            return self._run_harness_via_bridge(bridge, patch)
        return self._run_harness_locally(patch)

    def _run_harness_via_bridge(self, bridge: str, patch: str) -> bool:
        """A harness that cannot be reached means "unresolved", not "crash".

        The official harness runs the repository's real test suite inside a
        container; a slow or lost run used to raise and abort the whole cell,
        so one stalled instance destroyed the other ten rows already recorded.
        A harness we cannot talk to scores the instance unresolved and says so
        in ``_last_harness_note``.
        """
        timeout = int(os.environ.get("CALIX_SWEBENCH_HARNESS_TIMEOUT", "1800"))
        try:
            data = _post_json(f"{bridge}/step",
                              {"task_id": self.task["task_id"],
                               "action": f"```diff\n{patch}\n```"},
                              timeout, "SWE-bench harness")
        except EnvironmentError as exc:
            self._last_harness_note = (
                f"official harness unreachable ({exc}); instance scored "
                "unresolved")
            return False
        self._last_harness_note = str(data.get("observation", ""))
        return bool(data.get("success", False))

    def _run_harness_locally(self, patch: str) -> bool:
        os.makedirs("output/swebench", exist_ok=True)
        instance_id = self.task["task_id"].replace("swebench-", "", 1)
        pred_path = os.path.abspath("output/swebench/preds.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "instance_id": instance_id,
                "model_name_or_path": "calix",
                "model_patch": patch,
            }) + "\n")
        report_path = os.path.abspath(f"output/swebench/report_{instance_id}.json")
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench",
            "--predictions_path", pred_path,
            "--instance_ids", instance_id,
            "--max_workers", "1",
            "--run_id", f"calix_{instance_id}",
            "--report_dir", os.path.abspath("output/swebench"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise EnvironmentError(
                "SWE-bench official harness failed (Docker required). "
                f"stderr: {result.stderr[-400:]}"
            )
        # the harness writes <run_id>.json reports; locate ours
        for fname in os.listdir("output/swebench"):
            if fname.endswith(".json") and instance_id in fname and "report" not in fname:
                with open(os.path.join("output/swebench", fname)) as f:
                    report = json.load(f)
                resolved = report.get("resolved_ids") or report.get("resolved") or []
                if isinstance(resolved, list):
                    return instance_id in resolved
        raise EnvironmentError(
            f"SWE-bench harness produced no parseable report ({report_path}).")


_DIFF_RE = re.compile(r"```diff\s*(.*?)```", re.DOTALL)


def extract_patch(text: str) -> str:
    m = _DIFF_RE.search(text)
    if m:
        return m.group(1).strip()
    if "diff --git" in text:
        return text[text.index("diff --git"):].strip()
    return ""


# --------------------------------------------------------------------------- #
def load_task_file(path: str) -> list[dict]:
    """Load a benchmark task file (JSON array or JSONL)."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def build_environment(benchmark: str, tasks: list[dict],
                      max_steps: int = 20,
                      executor_url: str | None = None) -> RealEnvironment:
    if benchmark == "gaia":
        return GAIAEnvironment(tasks, max_steps)
    if benchmark == "agentbench":
        return AgentBenchHTTPEnvironment(tasks, max_steps, executor_url)
    if benchmark == "webarena":
        return WebArenaHTTPEnvironment(tasks, max_steps, executor_url)
    if benchmark in ("swebench", "swe_bench"):
        return SWEBenchEnvironment(tasks, max_steps, executor_url)
    raise ValueError(f"Unknown benchmark: {benchmark}")
