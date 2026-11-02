"""Real LLM access for CALIX (no mock / no simulation).

All LLM calls in CALIX go through :class:`OpenAICompatibleLLM`:

- candidate-policy generation   g_LLM(s_t)  -> C_t = {c_t^1, ..., c_t^K}
- policy decomposition          c_t*        -> {u_t^1, ..., u_t^N}
- per-role agent execution      u_t^i       -> y_t^i
- output fusion                 {y_t^i}     -> y_t
- candidate selection (w/o RL)  C_t         -> index k

Prompt templates live in ``prompts/`` so wording is versioned with the
repository.  The client talks to any OpenAI-compatible chat-completions
endpoint; credentials come from the environment:

    CALIX_OPENAI_API_KEY    (falls back to OPENAI_API_KEY)
    CALIX_OPENAI_BASE_URL   (falls back to OPENAI_BASE_URL,
                             default https://api.openai.com/v1)
    CALIX_MODEL             (optional; overrides llm.model in configs/*.yaml)

All three may be placed in ``<repo_root>/.env`` (git-ignored); the file is
loaded automatically by :func:`calix.env.load_dotenv`.  Real environment
variables take precedence over the file.

Every call records token usage when the endpoint returns a ``usage`` field;
:meth:`usage_summary` reports cumulative totals for cost accounting.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .env import load_dotenv
from .policy import CandidatePolicy
from .websearch import TOOL_SPEC, WebToolkit

_NUM_RE = re.compile(r"^\s*(\d+)\s*[\.\)、:：]\s*(.*)$")
_CONF_RE = re.compile(r"confidence\s*[:：]\s*(0?\.\d+|1(?:\.0+)?|0)", re.IGNORECASE)


class LLMError(RuntimeError):
    """Raised when the LLM endpoint fails or returns unusable output."""


class FatalEndpointError(LLMError):
    """A rejection that waiting cannot fix: revoked key, empty balance, ...

    On 2026-10-29 a DeepSeek key ran out of credit mid-run.  Every later task
    came back ``HTTP 402 Insufficient Balance``, and because the runner treats
    "endpoint down" as transient it sat through three 600-second wait rounds
    per method -- 40 minutes of sleep to rediscover the same 402.  A 402, 401
    or 403 will not clear on its own, so it is raised as a distinct type that
    callers can refuse to wait on.

    It subclasses :class:`LLMError`, so existing ``except LLMError`` handlers
    keep working; only the retry/wait logic needs to look at the subtype.
    """

    #: HTTP statuses that mean "this will never succeed without human action".
    FATAL_STATUS = (401, 402, 403)


class OpenAICompatibleLLM:
    """Client for OpenAI-compatible chat-completion endpoints.

    No mock behaviour exists in this class: every method issues a real HTTP
    request and parses the real response.  If the endpoint is unreachable or
    returns malformed output, an :class:`LLMError` is raised instead of
    fabricating content.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2,
                 top_p: float = 0.95, max_tokens: int = 4096,
                 prompt_dir: str | None = None,
                 max_retries: int = 6, request_timeout: int = 120,
                 tools: "WebToolkit | None" = None):
        load_dotenv()
        self.api_key = (os.environ.get("CALIX_OPENAI_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "")
        if not self.api_key:
            raise LLMError(
                "No API key found. Set CALIX_OPENAI_API_KEY (or "
                "OPENAI_API_KEY) before running real experiments."
            )
        self.base_url = (os.environ.get("CALIX_OPENAI_BASE_URL")
                         or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        # CALIX_MODEL (from .env or the shell) overrides the YAML model name,
        # which is handy when a gateway exposes different model identifiers.
        self.model = os.environ.get("CALIX_MODEL") or model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        if prompt_dir is None:
            prompt_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "prompts",
            )
        self.prompt_dir = prompt_dir
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        # Shared web toolkit. CALIX and every baseline are given the same
        # instance so that tool access is identical across methods.
        self.tools = tools
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls = 0

    # ------------------------------------------------------------------ I/O --
    def _post(self, payload: dict) -> dict:
        # Retries are deliberately generous: long benchmark runs must survive
        # transient proxy/TLS flaps (observed: intermittent CERTIFICATE_VERIFY
        # hostname mismatch through a local proxy that recovers seconds later).
        # CALIX_LLM_RETRIES / CALIX_LLM_BACKOFF_CAP override the defaults.
        # ``CALIX_LLM_RETRIES`` overrides the constructor, and the constructor
        # is honoured as-is -- a previous ``max(self.max_retries, 6)`` silently
        # discarded any value below 6, which made the parameter decorative and
        # turned a one-shot health probe into a two-minute wait.
        retries = int(os.environ.get("CALIX_LLM_RETRIES", self.max_retries))
        backoff_cap = float(os.environ.get("CALIX_LLM_BACKOFF_CAP", "60"))
        body = json.dumps(payload).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last_err = e
                # 4xx other than 429 is not retryable (auth, bad request, ...)
                if e.code != 429 and 400 <= e.code < 500:
                    detail = e.read().decode("utf-8", errors="replace")[:400]
                    cls = (FatalEndpointError if e.code
                           in FatalEndpointError.FATAL_STATUS else LLMError)
                    raise cls(
                        f"LLM endpoint rejected the request (HTTP {e.code}): {detail}"
                    ) from e
            except (urllib.error.URLError, TimeoutError, OSError,
                    json.JSONDecodeError) as e:
                # OSError covers ssl.SSLError subclasses (cert verify flaps)
                last_err = e
            time.sleep(min(backoff_cap, 2.0 * (2 ** attempt)))
        raise LLMError(
            f"LLM endpoint unreachable after {retries} attempts: {last_err}"
        )

    def chat(self, prompt: str, *, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        """One real chat-completion call. Returns the assistant content."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        data = self._post(payload)
        self.calls += 1
        usage = data.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Malformed chat-completion response: {data!r:.400}") from e

    def usage_summary(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
        }

    # ------------------------------------------------------------ tool use --
    def chat_with_tools(self, prompt: str, max_tool_calls: int = 3,
                        tool_spec: str = TOOL_SPEC) -> tuple[str, int]:
        """Run ``prompt`` with a tool-use loop.

        The model may emit ``SEARCH[...]`` / ``FETCH[...]``.  Each call is
        executed by the shared toolkit and returned as an ``OBSERVATION``.
        Returns ``(final_text, n_tool_calls)``.
        """
        if self.tools is None or not self.tools.enabled:
            return self.chat(prompt), 0
        transcript = prompt + "\n\n" + tool_spec.format(
            k=self.tools.results_per_search)
        used = 0
        for _ in range(max_tool_calls + 1):
            out = self.chat(transcript)
            result = self.tools.dispatch(out)
            if result is None:
                return out, used
            used += 1
            self.tool_calls += 1
            transcript += (
                f"\n\n{out}\n\nOBSERVATION:\n{result}\n\n"
                "Continue: emit one more tool call, or give your final answer "
                "now without any tool call."
            )
        return out, used

    # ------------------------------------------------------------- prompts --
    def render_prompt(self, template_name: str, **fields) -> str:
        path = os.path.join(self.prompt_dir, template_name)
        with open(path, "r", encoding="utf-8") as f:
            template = f.read()
        # keep only template lines that are not comments
        body = "\n".join(
            ln for ln in template.splitlines() if not ln.startswith("# ")
        ).strip()
        return body.format(**fields)

    # ----------------------------------------------- candidate generation --
    def generate_candidates(self, observation: str, retrieved_memory: str,
                            graph_summary: str, num_candidates: int,
                            retrieval_topk: int = 5) -> list[CandidatePolicy]:
        """g_LLM(s_t): produce exactly ``num_candidates`` candidate policies."""
        prompt = self.render_prompt(
            "candidate_generation.txt",
            K=num_candidates,
            observation=observation,
            retrieval_topk=retrieval_topk,
            retrieved_memory=retrieved_memory or "(no history yet)",
            graph_summary=graph_summary,
        )
        last_raw = ""
        for _ in range(2):
            raw = self.chat(prompt)
            last_raw = raw
            cands = self._parse_numbered_list(raw, num_candidates)
            if len(cands) == num_candidates:
                return [CandidatePolicy(index=k + 1, text=t)
                        for k, t in enumerate(cands)]
            # ask the endpoint to repair its own malformed output once
            prompt = (
                f"Your previous answer did not contain exactly {num_candidates} "
                f"numbered items. Return exactly {num_candidates} distinct "
                f"collaborative policies, one per line, numbered 1..{num_candidates}. "
                f"Observation:\n{observation}\n\nPrevious answer:\n{raw}"
            )
        raise LLMError(
            f"Could not parse {num_candidates} numbered candidate policies from "
            f"the LLM output: {last_raw!r:.600}"
        )

    @staticmethod
    def _parse_numbered_list(text: str, expected: int) -> list[str]:
        """Parse a numbered list; multi-line entries are joined."""
        items: list[str] = []
        current: list[str] | None = None
        for line in text.splitlines():
            m = _NUM_RE.match(line)
            if m:
                if current:
                    items.append(" ".join(current).strip())
                current = [m.group(2)]
            elif current is not None and line.strip():
                current.append(line.strip())
        if current:
            items.append(" ".join(current).strip())
        items = [it for it in items if it]
        return items[:expected]

    # --------------------------------------------------- policy selection --
    def select_candidate(self, observation: str,
                         candidates: list[CandidatePolicy]) -> int:
        """LLM-only selection used by the w/o-RL ablation (no value function)."""
        listing = "\n".join(f"{c.index}. {c.text}" for c in candidates)
        prompt = (
            "You are given an environment observation and a set of candidate "
            "collaborative policies. Choose the single best policy for the "
            "current situation. Answer with only the policy number.\n\n"
            f"Observation:\n{observation}\n\nCandidates:\n{listing}\n\nBest policy number:"
        )
        raw = self.chat(prompt, max_tokens=8)
        m = re.search(r"\d+", raw)
        if not m:
            raise LLMError(f"Could not parse selected candidate index from: {raw!r:.200}")
        idx = int(m.group(0))
        if not 1 <= idx <= len(candidates):
            raise LLMError(f"Selected candidate index {idx} out of range 1..{len(candidates)}")
        return idx - 1

    # ----------------------------------------------------- decomposition --
    def decompose_policy(self, selected_policy: str,
                         agent_roles: list[str]) -> dict[str, str]:
        """c_t* -> {u_t^i}: one real sub-policy per role."""
        prompt = self.render_prompt(
            "policy_decomposition.txt",
            selected_policy=selected_policy,
            agent_roles=", ".join(agent_roles),
        )
        raw = self.chat(prompt)
        return self._parse_role_assignments(raw, agent_roles, fallback=selected_policy)

    @staticmethod
    def _parse_role_assignments(text: str, roles: list[str],
                                fallback: str) -> dict[str, str]:
        out: dict[str, str] = {}
        current_role: str | None = None
        buf: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^\s*([A-Za-z][A-Za-z0-9 _\-]{1,30})\s*[:：]\s*(.*)$", line)
            if m and any(m.group(1).strip().lower() == r.lower() for r in roles):
                if current_role is not None:
                    out[current_role] = " ".join(buf).strip()
                current_role = next(r for r in roles
                                    if r.lower() == m.group(1).strip().lower())
                buf = [m.group(2)]
            elif current_role is not None and line.strip():
                buf.append(line.strip())
        if current_role is not None:
            out[current_role] = " ".join(buf).strip()
        # any role the LLM omitted receives the full policy with a scope note
        for r in roles:
            if not out.get(r):
                out[r] = f"[{r}] Contribute to the shared objective: {fallback}"
        return out

    # ------------------------------------------------------- role agents --
    def run_role(self, role: str, sub_policy: str, observation: str,
                 shared_context: str, max_tool_calls: int = 2
                 ) -> tuple[str, float | None, int]:
        """Execute one agent role via its versioned role prompt.

        Returns ``(text, confidence, n_tool_calls)``.
        """
        template = os.path.join("agents", f"{role.lower()}.txt")
        if not os.path.exists(os.path.join(self.prompt_dir, template)):
            prompt = (
                f"You are the {role} agent in a collaborative multi-agent team.\n"
                f"Sub-policy: {sub_policy}\nObservation: {observation}\n"
                f"Shared context from other agents: {shared_context}\n"
                "Produce your intermediate result, then a line 'Confidence: <0-1>'."
            )
        else:
            prompt = self.render_prompt(
                template, sub_policy=sub_policy, observation=observation,
                shared_context=shared_context or "(none yet)",
            )
        raw, n_tools = self.chat_with_tools(prompt, max_tool_calls=max_tool_calls)
        m = _CONF_RE.search(raw)
        confidence = float(m.group(1)) if m else None
        return raw, confidence, n_tools

    # ------------------------------------------------------------- fusion --
    def fuse_outputs(self, agent_outputs: list[dict],
                     protocol: str = "") -> dict:
        """{y_t^i} -> y_t via the versioned fusion prompt.

        ``protocol`` is the runtime's submission syntax, when it has one.  It is
        restated here because the fusion module writes the text actually handed
        to the environment, and a merged reply that drops the required marker
        is graded zero for phrasing rather than for content.
        """
        rendered = "\n\n".join(
            f"[{o['role']}] (confidence: "
            f"{o['confidence'] if o['confidence'] is not None else 'n/a'})\n{o['text']}"
            for o in agent_outputs
        )
        prompt = self.render_prompt("fusion.txt", agent_outputs=rendered,
                                    protocol=protocol or "none: reply plainly")
        raw = self.chat(prompt)
        decision, confidence, rationale = raw, None, ""
        dm = re.search(r"decision\s*[:：]\s*(.+?)(?:\n\s*confidence\s*[:：]|\n\s*rationale\s*[:：]|\Z)",
                       raw, re.IGNORECASE | re.DOTALL)
        if dm:
            decision = dm.group(1).strip()
        cm = _CONF_RE.search(raw)
        if cm:
            confidence = float(cm.group(1))
        rm = re.search(r"rationale\s*[:：]\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if rm:
            rationale = rm.group(1).strip()
        if not dm:
            # malformed fusion output: fall back to the highest-confidence output
            best = max(agent_outputs,
                       key=lambda o: (-1.0 if o["confidence"] is None else o["confidence"]))
            decision = best["text"]
            rationale = "fallback: highest-confidence agent output"
        return {"decision": decision, "confidence": confidence, "rationale": rationale,
                "raw": raw}
