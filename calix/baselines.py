"""Baseline executors (real LLM calls only, no synthetic results).

All nine baselines are *in-house reimplementations* executed through this
repository's own LLM client and its own web toolkit, exactly as stated in the
paper (Table 5: "in-house (repo)").  Every baseline therefore

  * calls the same model through ``OpenAICompatibleLLM``,
  * uses the same ``WebToolkit`` instance (same backends, same per-task
    budget), and
  * is scored by the same environment and the same success criterion.

The coordination mechanism of each framework is reproduced from its published
description; the prompt templates live in ``prompts/baselines/``:

  gpt4o_direct  single direct answer attempt (no collaboration)
  react         Thought -> Action loop with environment feedback
  camel         instructor/assistant role-playing pair
  autogen       group chat with a manager selecting the next speaker
  agentverse    role recruitment -> expert proposals -> group decision
  metagpt       SOP assembly line: PM -> architect -> engineer -> QA
  chatdev       phase chain: design -> code -> review -> test
  sweagent      shell-style action loop against the execution environment
  openhands     CodeAct loop (program execution) with environment feedback

``sweagent`` and ``openhands`` need a real shell.  When the active environment
provides one (SWE-bench through the Colab bridge, AgentBench, WebArena) their
commands are executed there; when it does not (GAIA, which offers no execution
sandbox) they fall back to the web tools and this is recorded per task in
``shell_available`` so the difference is visible in the logs instead of being
hidden.
"""
from __future__ import annotations

import threading

from .environment import RealEnvironment, extract_final_answer, gold_of
from .llm import OpenAICompatibleLLM
from .watchdog import CircuitBreaker, TaskTimeout, task_deadline
from .websearch import WebToolkit


def _task_ids(env) -> list[str]:
    """Real task ids behind positional indices (same rule as in calix.py).

    Bookkeeping rows must name the task they stand for, otherwise ``--resume``
    cannot find the gap again.
    """
    tasks = getattr(env, "tasks", None) or []
    return [str(t.get("task_id", "")) for t in tasks]

BASELINE_NAMES = [
    "gpt4o_direct",
    "react",
    "camel",
    "autogen",
    "agentverse",
    "metagpt",
    "chatdev",
    "sweagent",
    "openhands",
]

# Official references, used for provenance records and for Table 5 only.
OFFICIAL_REFERENCE = {
    "camel": "https://github.com/camel-ai/camel",
    "autogen": "https://github.com/microsoft/autogen",
    "agentverse": "https://github.com/OpenBMB/AgentVerse",
    "metagpt": "https://github.com/geekan/MetaGPT",
    "chatdev": "https://github.com/OpenBMB/ChatDev",
    "sweagent": "https://github.com/SWE-agent/SWE-agent",
    "openhands": "https://github.com/All-Hands-AI/OpenHands",
}


# --------------------------------------------------------------------------- #
#  Shared context
# --------------------------------------------------------------------------- #
class _Ctx:
    """Everything one baseline run needs; shared by every runner."""

    def __init__(self, llm: OpenAICompatibleLLM, env: RealEnvironment,
                 obs: dict, max_steps: int):
        self.llm = llm
        self.env = env
        self.tools = llm.tools
        self.obs = obs
        self.max_steps = max(1, int(max_steps))
        self.shell_available: bool | None = None
        self.shell_used = False
        if self.tools is not None:
            self.tools.reset_budget()

    @property
    def budget(self) -> int:
        if self.tools is None:
            return 0
        return self.tools.max_search_calls + self.tools.max_fetch_calls

    def ask(self, prompt: str):
        """One LLM turn, with the shared toolkit attached."""
        return self.llm.chat_with_tools(prompt, max_tool_calls=self.budget)

    def shell(self, command: str) -> str:
        """Execute ``command`` in the environment, if it offers a shell.

        Raises ``NotImplementedError`` when the environment has no execution
        sandbox; the caller records that fact and continues without one.
        """
        out = self.env.shell(command)
        self.shell_available = True
        self.shell_used = True
        return out

    def submit(self, answer: str, tool_calls: int = 0):
        """Submit ``answer`` to the environment; one interaction step."""
        action = (extract_final_answer(answer)
                  if self.env.name == "gaia" else answer)
        _, info, done = self.env.step(action)
        return bool(done and info.get("success", False))

    def tool_calls(self) -> int:
        return int(self.llm.tool_calls) if self.llm is not None else 0


def _runtime_ready(env) -> bool | None:
    """Was the environment's runtime (container / browser / DB) actually up?

    Only SWE-bench can answer False today: its container is what lets an agent
    explore the repository, and when the image is missing the task is still
    scored, just without exploration.  Recording it keeps that 0.0 from being
    read as a normal attempt.
    """
    return getattr(env, "_container_ready", None)


def _result(ctx: _Ctx, success: bool, steps: int, tool_calls: int,
            answer: str) -> dict:
    return {"success": bool(success), "steps": int(steps),
            "tool_calls": int(tool_calls), "answer": answer,
            "shell_used": bool(ctx.shell_used),
            "shell_available": ctx.shell_available,
            "runtime_ready": _runtime_ready(ctx.env)}


# --------------------------------------------------------------------------- #
#  1. GPT-4o Direct
# --------------------------------------------------------------------------- #
def _run_direct(ctx: _Ctx) -> dict:
    prompt = (
        "Solve the following task. Give your final answer on a line starting "
        f"with 'FINAL ANSWER:'.\n\n{ctx.obs['text']}"
    )
    answer, _ = ctx.ask(prompt)
    ok = ctx.submit(answer)
    return _result(ctx, ok, 1, ctx.tool_calls(), answer)


# --------------------------------------------------------------------------- #
#  2. ReAct
# --------------------------------------------------------------------------- #
def _run_react(ctx: _Ctx) -> dict:
    tools = ctx.tools
    history: list[str] = []
    total_tools = 0
    steps = 0
    extra = ctx.budget
    for it in range(ctx.max_steps + extra):
        prompt = (
            "You are a ReAct agent. Think step by step, then act.\n"
            "Format:\nThought: <reasoning>\nAction: <your current best answer "
            "or next command>\n\n"
            "You may also act with a web tool by writing exactly one of:\n"
            "Action: SEARCH[<query>]\nAction: FETCH[<url>]\n"
            "Tool actions return an observation and do not submit an answer.\n\n"
            f"Task:\n{ctx.obs['text']}\n\nHistory:\n"
            f"{chr(10).join(history) or '(empty)'}"
        )
        out = ctx.llm.chat(prompt)
        action = out.split("Action:", 1)[-1].strip() if "Action:" in out else out
        if tools is not None:
            res = tools.dispatch(action)
            if res is not None:
                total_tools += 1
                ctx.llm.tool_calls += 1
                history.append(f"Action {it}: {action}")
                history.append(f"Observation {it}: {res[:1500]}")
                continue
        if steps >= ctx.max_steps:
            break
        steps += 1
        ok = ctx.submit(action)
        history.append(f"Thought/Action {it}: {out[:400]}")
        history.append(f"Observation {it}: submitted (correct={ok})")
        if ok:
            return _result(ctx, True, steps, total_tools, action)
    return _result(ctx, False, steps, total_tools,
                   history[-1] if history else "")


# --------------------------------------------------------------------------- #
#  3. CAMEL (instructor / assistant role play)
# --------------------------------------------------------------------------- #
def _run_camel(ctx: _Ctx) -> dict:
    history: list[str] = []
    last_answer = ""
    base_user = ctx.llm.render_prompt("baselines/camel_instructor.txt",
                                      task=ctx.obs["text"])
    for _ in range(ctx.max_steps):
        user_prompt = base_user + "\n\nConversation so far:\n" + (
            "\n".join(history[-6:]) or "(empty)")
        instruction = ctx.llm.chat(user_prompt)
        if "TASK_DONE" in instruction.upper():
            break
        instruction = (instruction.split("Instruction:", 1)[-1].strip()
                       if "Instruction:" in instruction else instruction.strip())
        out, _ = ctx.ask(ctx.llm.render_prompt(
            "baselines/camel_assistant.txt", task=ctx.obs["text"],
            history="\n".join(history[-6:]) or "(empty)", message=instruction))
        history.append(f"USER: {instruction}")
        history.append(f"ASSISTANT: {out[:1200]}")
        if "Answer:" in out:
            last_answer = out.split("Answer:", 1)[-1].strip()
    ok = ctx.submit(last_answer)
    return _result(ctx, ok, 1, ctx.tool_calls(), last_answer)


# --------------------------------------------------------------------------- #
#  4. AutoGen (group chat with a manager)
# --------------------------------------------------------------------------- #
_AUTOGEN_ROLES = ["Planner", "Researcher", "Engineer", "Critic"]


def _run_autogen(ctx: _Ctx) -> dict:
    transcript: list[str] = []
    final = ""
    for _ in range(ctx.max_steps * len(_AUTOGEN_ROLES)):
        choice = ctx.llm.chat(ctx.llm.render_prompt(
            "baselines/autogen_moderator.txt", task=ctx.obs["text"],
            transcript="\n".join(transcript[-12:]) or "(empty)",
            roles=", ".join(_AUTOGEN_ROLES + ["TERMINATE"])))
        nxt = (choice.split("Next:", 1)[-1].strip()
               if "Next:" in choice else "").strip()
        if not nxt or nxt.upper().startswith("TERMINATE"):
            break
        role = next((r for r in _AUTOGEN_ROLES
                     if r.lower() in nxt.lower()), _AUTOGEN_ROLES[0])
        out, _ = ctx.ask(ctx.llm.render_prompt(
            "baselines/autogen_agent.txt", role=role, task=ctx.obs["text"],
            transcript="\n".join(transcript[-12:]) or "(empty)"))
        transcript.append(f"{role}: {out[:1200]}")
        if "Final Answer:" in out:
            final = out.split("Final Answer:", 1)[-1].strip()
            break
    ok = ctx.submit(final)
    return _result(ctx, ok, 1, ctx.tool_calls(), final)


# --------------------------------------------------------------------------- #
#  5. AgentVerse (recruitment -> proposals -> decision)
# --------------------------------------------------------------------------- #
def _run_agentverse(ctx: _Ctx) -> dict:
    recruit = ctx.llm.chat(ctx.llm.render_prompt(
        "baselines/agentverse_recruit.txt", task=ctx.obs["text"]))
    roles = []
    for line in recruit.splitlines():
        if line.lower().startswith("role:"):
            body = line[5:].strip()
            name, _, duty = body.partition("-")
            roles.append((name.strip() or "Expert", duty.strip() or "solve"))
    if not roles:
        roles = [("Analyst", "analyse"), ("Researcher", "retrieve facts"),
                 ("Critic", "verify")]
    proposals = []
    for name, duty in roles[:4]:
        out, _ = ctx.ask(ctx.llm.render_prompt(
            "baselines/agentverse_role.txt", role=name, duty=duty,
            task=ctx.obs["text"],
            proposals="\n".join(proposals[-4:]) or "(none yet)"))
        ans = out.split("Answer:", 1)[-1].strip() if "Answer:" in out else out
        proposals.append(f"{name}: {ans[:800]}")
    decision = ctx.llm.chat(ctx.llm.render_prompt(
        "baselines/agentverse_vote.txt", task=ctx.obs["text"],
        proposals="\n".join(proposals)))
    final = (decision.split("Decision:", 1)[-1].strip()
             if "Decision:" in decision else decision)
    ok = ctx.submit(final)
    return _result(ctx, ok, 1, ctx.tool_calls(), final)


# --------------------------------------------------------------------------- #
#  6. MetaGPT (SOP assembly line)
# --------------------------------------------------------------------------- #
_METAGPT_LINE = [
    ("Product Manager",
     "Write the requirement/PRD: what must be delivered, constraints, and the "
     "acceptance criteria."),
    ("Architect",
     "Turn the PRD into a concrete plan: the steps and the artefacts needed."),
    ("Engineer",
     "Produce the deliverable itself and end with the final answer."),
    ("QA Engineer",
     "Verify the deliverable against the acceptance criteria; correct it if "
     "needed and restate the final answer."),
]


def _run_metagpt(ctx: _Ctx) -> dict:
    artifact = "(none)"
    final = ""
    for role, duty in _METAGPT_LINE:
        out, _ = ctx.ask(ctx.llm.render_prompt(
            "baselines/metagpt_role.txt", role=role, duty=duty,
            task=ctx.obs["text"], artifact=artifact))
        artifact = (out.split("Artifact:", 1)[-1].strip()
                    if "Artifact:" in out else out.strip())
        final = artifact
    ok = ctx.submit(final)
    return _result(ctx, ok, 1, ctx.tool_calls(), final)


# --------------------------------------------------------------------------- #
#  7. ChatDev (design -> code -> review -> test)
# --------------------------------------------------------------------------- #
def _run_chatdev(ctx: _Ctx) -> dict:
    phases = [("design", "Chief Executive Officer", "Chief Technology Officer"),
              ("code", "Chief Technology Officer", "Programmer"),
              ("review", "Programmer", "Code Reviewer")]
    artifact = "(none)"
    for phase, a, b in phases:
        out, _ = ctx.ask(ctx.llm.render_prompt(
            "baselines/chatdev_phase.txt", phase=phase, role_a=a, role_b=b,
            task=ctx.obs["text"], artifact=artifact))
        artifact = (out.split("Artifact:", 1)[-1].strip()
                    if "Artifact:" in out else out.strip())
    review = artifact
    out, _ = ctx.ask(ctx.llm.render_prompt(
        "baselines/chatdev_test.txt", task=ctx.obs["text"], artifact=artifact,
        review=review))
    final = (out.split("Final Answer:", 1)[-1].strip()
             if "Final Answer:" in out else artifact)
    ok = ctx.submit(final)
    return _result(ctx, ok, 1, ctx.tool_calls(), final)


# --------------------------------------------------------------------------- #
#  8/9. SWE-agent (shell actions) and OpenHands (CodeAct)
# --------------------------------------------------------------------------- #
def _extract_block(text: str) -> str:
    """Pull a fenced code block out of an LLM answer."""
    if "```" not in text:
        return ""
    parts = text.split("```")
    return parts[1].strip() if len(parts) > 1 else ""


def _run_execution_agent(ctx: _Ctx, kind: str) -> dict:
    """Shared driver for the execution-oriented baselines.

    Both emit one action per turn.  A command/code block is executed in the
    environment's shell when it has one; the finish action is submitted to the
    environment and scored.
    """
    tpl = ("baselines/sweagent_step.txt" if kind == "sweagent"
           else "baselines/openhands_step.txt")
    history: list[str] = []
    steps = 0
    final = ""
    for _ in range(ctx.max_steps + ctx.budget):
        out, _ = ctx.ask(ctx.llm.render_prompt(
            tpl, task=ctx.obs["text"],
            history="\n".join(history[-8:]) or "(empty)"))
        if kind == "sweagent" and "Command:" in out:
            cmd = out.split("Command:", 1)[-1].strip().splitlines()[0].strip()
            obs = _try_shell(ctx, cmd)
            history.append(f"Command: {cmd}\nObservation: {obs}")
            continue
        if kind == "openhands" and "Code:" in out:
            code = _extract_block(out.split("Code:", 1)[-1])
            obs = _try_shell(ctx, code)
            history.append(f"Code executed.\nObservation: {obs}")
            continue
        marker = "Submit:" if kind == "sweagent" else "Finish:"
        final = out.split(marker, 1)[-1].strip() if marker in out else out
        steps += 1
        ok = ctx.submit(final)
        history.append(f"{marker} {final[:400]}\nObservation: correct={ok}")
        if ok:
            return _result(ctx, True, steps, ctx.tool_calls(), final)
    return _result(ctx, False, steps, ctx.tool_calls(), final)


def _try_shell(ctx: _Ctx, command: str) -> str:
    try:
        return str(ctx.shell(command))[:2000]
    except NotImplementedError:
        ctx.shell_available = False
        return ("(no execution sandbox is attached to this environment; "
                "the command could not be run. Continue with the web tools or "
                "finish with your answer.)")


def _run_sweagent(ctx: _Ctx) -> dict:
    return _run_execution_agent(ctx, "sweagent")


def _run_openhands(ctx: _Ctx) -> dict:
    return _run_execution_agent(ctx, "openhands")


BASELINE_RUNNERS = {
    "gpt4o_direct": _run_direct,
    "react": _run_react,
    "camel": _run_camel,
    "autogen": _run_autogen,
    "agentverse": _run_agentverse,
    "metagpt": _run_metagpt,
    "chatdev": _run_chatdev,
    "sweagent": _run_sweagent,
    "openhands": _run_openhands,
}


# --------------------------------------------------------------------------- #
#  Evaluation driver
# --------------------------------------------------------------------------- #
def run_one_task(name: str, llm: OpenAICompatibleLLM,
                 env: RealEnvironment, task_index: int,
                 max_steps: int) -> dict:
    """Run one baseline on one task. Returns the task-level record dict."""
    if name not in BASELINE_RUNNERS:
        raise ValueError(f"Unknown baseline: {name}")
    obs = env.reset(task_index)
    ctx = _Ctx(llm, env, obs, max_steps)
    before = llm.usage_summary()
    res = BASELINE_RUNNERS[name](ctx)
    after = llm.usage_summary()
    return {
        "task_id": env.task["task_id"],
        "success": bool(res["success"]),
        "steps": int(res["steps"]),
        "tool_calls": int(res["tool_calls"]),
        "llm_calls": int(after["calls"] - before["calls"]),
        "llm_prompt_tokens": int(after["prompt_tokens"] - before["prompt_tokens"]),
        "llm_completion_tokens": int(after["completion_tokens"]
                                     - before["completion_tokens"]),
        "shell_used": bool(res["shell_used"]),
        "shell_available": res["shell_available"],
        "runtime_ready": res["runtime_ready"],
        # Provenance: keep what the agent actually submitted next to the gold
        # value, so a 0 can be audited instead of taken on trust. 2000 chars is
        # enough to hold a full KG answer list or a browser `stop [answer]`.
        "answer": str(res.get("answer", ""))[:2000],
        "gold": _gold_of(env.task),
    }


def _gold_of(task: dict) -> str:
    """Gold value for the audit trail (see :func:`calix.environment.gold_of`)."""
    return gold_of(task)


def evaluate_baseline(name: str, env: RealEnvironment, num_tasks: int,
                      seed: int, model: str = "gpt-4o",
                      tools: "WebToolkit | None" = None,
                      workers: int = 1,
                      env_factory=None,
                      on_record=None) -> dict:
    """Run one baseline over ``num_tasks`` tasks of a real environment.

    ``tools`` must be the *same* toolkit instance used by CALIX so that tool
    access and budget are identical across methods.
    """
    if name not in BASELINE_RUNNERS:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {sorted(BASELINE_RUNNERS)}")

    root_llm = OpenAICompatibleLLM(model=model, tools=tools)
    parallel = bool(workers and workers > 1 and env_factory is not None)
    # the toolkit is shared with CALIX and its counters are cumulative, so the
    # per-method cost is measured as a delta over this run only
    tool_before = tools.usage() if tools is not None else {}

    _made: list = [root_llm]
    if parallel:
        _local = threading.local()
        _lock = threading.Lock()

        def get_llm():
            obj = getattr(_local, "llm", None)
            if obj is None:
                obj = OpenAICompatibleLLM(
                    model=model,
                    tools=tools.clone() if tools is not None else None)
                _local.llm = obj
                with _lock:
                    _made.append(obj)
            return obj
    else:
        def get_llm():
            return root_llm

    breaker = CircuitBreaker()
    list_ids = _task_ids(env)

    def _run_task(i: int) -> dict:
        if breaker.tripped:
            tid = list_ids[i] if i < len(list_ids) else f"index-{i}"
            # The runtime has been failing for consecutive tasks; do not spend
            # another half hour each rediscovering the same outage.
            return {"task_id": tid, "success": False, "steps": 0,
                    "tool_calls": 0, "llm_calls": 0, "llm_prompt_tokens": 0,
                    "llm_completion_tokens": 0, "shell_used": False,
                    "shell_available": False, "runtime_ready": None,
                    "answer": "", "gold": "", "aborted": True,
                    "error": "circuit breaker open after consecutive "
                             "infrastructure aborts"}
        llm = get_llm()
        e = env_factory() if parallel else env
        try:
            with task_deadline(what=f"{name} task {i}"):
                out = run_one_task(name, llm, e, i, e.max_steps)
        except BaseException as exc:  # noqa: BLE001 - see note below
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            # A lost runtime or an exhausted retry budget used to abort the
            # whole cell, silently discarding every row of a method that had
            # already been measured.  Surface it loudly and mark the single
            # instance aborted -- excluded from rates, never counted as a
            # failure, so an outage cannot masquerade as a wrong answer.
            tid = f"index-{i}"
            try:
                tid = e.task["task_id"]
            except Exception:  # noqa: BLE001
                pass
            print(f"[{name}] task {tid} aborted: "
                  f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
            breaker.record(True)
            return {"task_id": tid, "success": False, "steps": 0,
                    "tool_calls": 0, "llm_calls": 0, "llm_prompt_tokens": 0,
                    "llm_completion_tokens": 0, "shell_used": False,
                    "shell_available": False, "runtime_ready": None,
                    "answer": "", "gold": "", "aborted": True,
                    "error": f"{type(exc).__name__}: {exc}"[:300]}
        breaker.record(False)
        return out

    def run_one(i: int) -> dict:
        rec = _run_task(i)
        # Persist the moment a task is known, so a kill loses at most the one
        # currently in flight.
        if on_record is not None:
            on_record(rec)
        return rec

    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(run_one, range(num_tasks)))
    else:
        records = [run_one(i) for i in range(num_tasks)]

    llm_usage = {"model": root_llm.model, "calls": 0, "prompt_tokens": 0,
                 "completion_tokens": 0, "tool_calls": 0}
    for m in _made:
        u = m.usage_summary()
        for k in ("calls", "prompt_tokens", "completion_tokens", "tool_calls"):
            llm_usage[k] += int(u.get(k, 0))

    graded = [r for r in records if not r.get("aborted")]
    denom = max(len(graded), 1)
    tsr = 100.0 * sum(r["success"] for r in graded) / denom
    out = {"tsr": tsr,
           "avg_steps": sum(r["steps"] for r in graded) / denom,
           "records": records,
           "llm_usage": llm_usage}
    if tools is not None:
        after = tools.usage()
        out["tool_usage"] = {k: int(after[k]) - int(tool_before.get(k, 0))
                             for k in after}
    return out
