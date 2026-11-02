"""CALIX core: closed-loop adaptive learning (Algorithm 1) built on
behaviour-aware state construction (Algorithm 2).

All LLM interaction is real: candidate generation, policy decomposition,
per-role execution and fusion are performed by :class:`OpenAICompatibleLLM`
against a live endpoint, and the environment is a real benchmark runtime
(``calix.environment.RealEnvironment``).  No mock LLM, no simulated
environment, and no hidden "correct strategy" exist in this module.

Ablation switches (paper Section 5.3):
- use_rl=False          -> w/o RL (LLM-only candidate selection, no TD update)
- use_graph=False       -> w/o Graph (h_G zeroed in the state)
- use_behavior=False    -> w/o Behavior (no history in the state)
- use_closed_loop=False -> w/o Closed-loop (feedback not propagated into
                           memory / graph / next-state evolution)
"""
from __future__ import annotations

import numpy as np

from .agents import CollaborativeAgentTeam
from .embedding import DeterministicEmbedder
from .environment import RealEnvironment, gold_of
from .graph import DynamicInteractionGraph
from .llm import OpenAICompatibleLLM
from .memory import HistoricalMemory, MemoryItem
from .policy import CandidatePolicy, encode_candidate
from .provenance import ProvenanceRecorder
from .qnetwork import QNetwork, q_induced_softmax
from .replay import ReplayBuffer, Transition
from .reward import compute_reward
from .state import BehaviorAwareStateEncoder
from .watchdog import CircuitBreaker, TaskTimeout, task_deadline
from .websearch import WebToolkit, build_toolkit


def task_ids_of(env: "RealEnvironment | list[dict]") -> list[str]:
    """The real task ids behind ``task_index`` positional indices.

    Rows written for a task that never ran (circuit breaker open, external
    interrupt) used to be tagged ``index-<i>``.  Those ids are invisible to
    ``--resume``, which looks ids up in the eval pool, so the gap they marked
    could never be refilled: after the 2026-10-29 outage 35 sweagent tasks and
    84 calix tasks were permanently unrecoverable from the trace alone.  Any
    bookkeeping row must carry the id of the task it stands for.
    """
    tasks = env if isinstance(env, list) else (getattr(env, "tasks", None)
                                               or [])
    return [str(t.get("task_id", "")) for t in tasks]


class CALIX:
    def __init__(
        self,
        cfg: dict,
        seed: int = 2025,
        use_rl: bool = True,
        use_graph: bool = True,
        use_behavior: bool = True,
        use_closed_loop: bool = True,
        num_agents: int | None = None,
        removed_role: str | None = None,
        num_candidates: int | None = None,
        tools: WebToolkit | None = None,
    ):
        self.cfg = cfg
        self.seed = seed
        self.use_rl = use_rl
        self.use_graph = use_graph
        self.use_behavior = use_behavior
        self.use_closed_loop = use_closed_loop
        self.removed_role = removed_role

        emb_cfg = cfg["embedding"]
        self.embedder = DeterministicEmbedder(
            dim=emb_cfg["dim"], hash_bins=emb_cfg["hash_bins"]
        )

        roles = cfg["agents"]["roles"]
        n = num_agents or cfg["agents"]["num_agents"]
        self.roles = roles[:n] if n <= len(roles) else roles + [
            f"Agent{i}" for i in range(len(roles), n)]

        # Real LLM only.  There is intentionally no alternative backend:
        # running CALIX without a live endpoint is not supported.
        # Shared web toolkit: the same object is handed to every baseline so
        # that all methods operate under identical tool access and budget.
        self.tools = tools if tools is not None else build_toolkit(cfg)
        self.llm = OpenAICompatibleLLM(
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
            top_p=cfg["llm"]["top_p"],
            max_tokens=cfg["llm"]["max_output_tokens"],
            tools=self.tools,
        )
        self.team = CollaborativeAgentTeam(self.roles, self.embedder, self.llm)

        st = cfg["state"]
        self.state_encoder = BehaviorAwareStateEncoder(
            self.embedder, state_dim=st["dim"], hidden_dim=st["hidden_dim"],
            seed=st["encoder_seed"], use_graph=use_graph, use_behavior=use_behavior,
        )

        self.K = num_candidates or cfg["llm"]["num_candidates"]

        q = cfg["qnetwork"]
        device = cfg.get("device", "auto")
        self.q_net = QNetwork(q["input_dim"], q["hidden_dim"], lr=q["lr"],
                              grad_clip=q["grad_clip"], seed=seed + 3,
                              device=device)
        self.target_q_net = QNetwork(q["input_dim"], q["hidden_dim"], lr=q["lr"],
                                     grad_clip=q["grad_clip"], seed=seed + 3,
                                     device=device)
        self.target_q_net.copy_from(self.q_net)
        self.replay = ReplayBuffer(q["replay_capacity"], seed=seed + 4)
        self.gamma = q["gamma"]
        self.batch_size = q["batch_size"]
        self.warmup = q["warmup"]
        self.target_update_interval = q["target_update_interval"]
        self.tau = cfg["llm"]["selection_tau"]
        self._opt_steps = 0
        self._select_rng = np.random.default_rng(seed + 5)
        self.updates_per_step = cfg["qnetwork"].get("updates_per_step", 1)
        self.retrieval_topk = cfg["memory"]["retrieval_topk"]

    # -- helpers ---------------------------------------------------------------
    def _new_episode_structures(self) -> tuple[DynamicInteractionGraph, HistoricalMemory]:
        graph = DynamicInteractionGraph(self.roles, feature_dim=self.embedder.dim)
        role_feats = {r: self.embedder.embed(f"role:{r}") for r in self.roles}
        graph.initialise(role_feats)
        mem = HistoricalMemory(self.cfg["memory"]["capacity"],
                               self.cfg["memory"]["retrieval_topk"])
        return graph, mem

    @staticmethod
    def _format_retrieved(retrieved: list[MemoryItem], max_chars: int = 240) -> str:
        if not retrieved:
            return ""
        lines = []
        for it in retrieved:
            lines.append(
                f"- (reward {it.reward:.2f}, outcome {it.outcome}) "
                f"policy: {it.policy_text[:max_chars]} | feedback: {it.feedback[:120]}"
            )
        return "\n".join(lines)

    def _q_values(self, s: np.ndarray, z_list: list[np.ndarray]) -> np.ndarray:
        X = np.stack([np.concatenate([s, z]) for z in z_list])
        return self.q_net.predict(X)

    def _td_learn(self) -> float | None:
        """Sample a mini-batch from D and run one TD optimisation step."""
        if len(self.replay) < self.warmup:
            return None
        batch = self.replay.sample(self.batch_size)
        X = np.stack([np.concatenate([tr.state, tr.policy_embedding]) for tr in batch])
        # vectorised Bellman targets: one batched target-network forward pass
        # for all non-terminal transitions instead of per-transition calls
        targets = np.asarray([tr.reward for tr in batch], dtype=np.float64)
        nonterm_idx = [
            i for i, tr in enumerate(batch)
            if not tr.done and tr.next_candidate_embeddings is not None
        ]
        if nonterm_idx:
            blocks, counts = [], []
            for i in nonterm_idx:
                tr = batch[i]
                z_next = tr.next_candidate_embeddings
                blocks.append(np.concatenate(
                    [np.repeat(tr.next_state[None, :], len(z_next), axis=0), z_next],
                    axis=1,
                ))
                counts.append(len(z_next))
            q_next = self.target_q_net.predict(np.vstack(blocks))
            max_q = np.maximum.reduceat(q_next, np.cumsum([0] + counts[:-1]))
            targets[nonterm_idx] += self.gamma * max_q
        loss = self.q_net.td_update(X, targets)
        self._opt_steps += 1
        if self._opt_steps % self.target_update_interval == 0:
            self.target_q_net.copy_from(self.q_net)
        return loss

    # -- main episode loop (Algorithm 1) ----------------------------------------
    @staticmethod
    def _prompt_observation(task_text: str, env_text: str, round_index: int,
                            protocol: str = "") -> str:
        """Observation handed to the LLM: always the task, plus any feedback.

        Environments such as GAIA answer an attempt with ``"the submitted
        answer is incorrect, try again"`` and nothing else, so passing ``env_text``
        verbatim drops the question from the prompt. The agent then replies
        "the problem statement is missing" and spends the rest of its budget
        with no task in context. Regression test:
        ``TestTaskContextInPrompt``.

        ``protocol`` is the runtime's own submission syntax (see
        ``RealEnvironment.submission_protocol``).  Baseline templates carry it
        in their prompt; a fused multi-agent answer has to be told the same
        thing or it is scored as zero for phrasing rather than for content.
        """
        base = task_text if (round_index == 0 or env_text == task_text) \
            else f"{task_text}\n\nEnvironment: {env_text}"
        if protocol and protocol not in base:
            base = f"{base}\n\n{protocol}"
        return base

    def run_episode(
        self,
        env: RealEnvironment,
        task_index: int | None = None,
        training: bool = True,
        recorder: ProvenanceRecorder | None = None,
        graph: DynamicInteractionGraph | None = None,
        memory: HistoricalMemory | None = None,
    ) -> dict:
        obs = env.reset(task_index)
        # every episode/task gets the same fresh tool budget
        self.tools.reset_budget()
        tool_before = dict(self.tools.usage())
        if graph is None or memory is None or not self.use_closed_loop:
            graph, memory = self._new_episode_structures()

        cum_reward = 0.0
        steps = 0
        losses: list[float] = []
        env_info: dict = {"success": False}
        # The environment replaces its observation with feedback after the first
        # action ("the submitted answer is incorrect, try again"), so the task
        # text itself disappears from the prompt. On a multi-attempt benchmark
        # like GAIA the agent then answers "the problem statement is missing"
        # and burns the whole budget. Keep the task text and append the latest
        # environment message instead of replacing it.
        task_text = obs["text"]
        last_submission = ""
        protocol = ""
        try:
            protocol = env.submission_protocol()
        except AttributeError:  # older environment objects
            protocol = ""

        for t in range(env.max_steps):
            # --- Stage 1: behaviour-aware state construction (Algorithm 2)
            s_t, aux = self.state_encoder.construct(graph, memory, obs["text"])
            prompt_obs = self._prompt_observation(task_text, obs["text"], t,
                                                  protocol)

            # --- Stage 2: real LLM candidate generation + Q evaluation
            candidates = self.llm.generate_candidates(
                observation=prompt_obs,
                retrieved_memory=self._format_retrieved(aux["retrieved"]),
                graph_summary=graph.summary_text(),
                num_candidates=self.K,
                retrieval_topk=self.retrieval_topk,
            )
            z_list = [encode_candidate(c, self.embedder) for c in candidates]
            q_vals = self._q_values(s_t, z_list)

            if self.use_rl:
                if training:
                    probs = q_induced_softmax(q_vals, self.tau)
                    sel = int(self._select_rng.choice(len(candidates), p=probs))
                else:
                    sel = int(np.argmax(q_vals))
            else:
                # w/o RL: the LLM itself selects among candidates (no values)
                sel = self.llm.select_candidate(obs["text"], candidates)
            c_star = candidates[sel]
            z_star = z_list[sel]

            # --- Stage 3: real multi-agent collaborative execution
            sub_policies = self.team.decompose(c_star.text)
            agent_outputs, exec_info = self.team.execute(
                sub_policies, prompt_obs, removed_role=self.removed_role,
                protocol=protocol,
            )

            # --- environment interaction E(x_t, y_t) with the real runtime
            next_obs, env_info, done = env.step(exec_info["fused_text"])
            last_submission = str(exec_info["fused_text"])
            terminal_success = bool(done and env_info.get("success", False))
            rw = compute_reward(
                success=terminal_success, step=t, max_steps=env.max_steps,
                communications=exec_info["communications"],
                num_agents=exec_info["num_active_agents"],
                alpha=self.cfg["reward"]["alpha"], beta=self.cfg["reward"]["beta"],
                lam=self.cfg["reward"]["lam"],
                success_bonus=self.cfg["reward"]["success_bonus"],
                coord_budget_per_agent=self.cfg["reward"]["coord_budget_per_agent"],
            )
            reward = rw["reward"]
            cum_reward += reward
            steps += 1

            # --- Stage 4: closed-loop update of memory / graph / state
            if self.use_closed_loop:
                memory.update(
                    task_id=obs["task_id"], observation=obs["text"],
                    policy_text=c_star.text, reward=reward,
                    outcome="success" if terminal_success else "in_progress",
                    feedback=next_obs["text"], embedding=aux["observation_embedding"],
                )
                graph.evolve(
                    policy_text=c_star.text, policy_feature=z_star,
                    agent_outputs={o.role: o.embedding for o in agent_outputs},
                    observation_feature=self.embedder.embed(next_obs["text"]),
                )
            s_next, _ = self.state_encoder.construct(graph, memory, next_obs["text"])

            # --- next candidates for the Bellman target
            next_z = None
            if not done:
                next_cands = self.llm.generate_candidates(
                    observation=next_obs["text"],
                    retrieved_memory=self._format_retrieved(
                        memory.retrieve(self.embedder.embed(next_obs["text"]))
                        if self.use_behavior else []),
                    graph_summary=graph.summary_text(),
                    num_candidates=self.K,
                    retrieval_topk=self.retrieval_topk,
                )
                next_z = np.stack([encode_candidate(c, self.embedder) for c in next_cands])

            # --- store transition & TD update
            if self.use_rl and training:
                self.replay.add(Transition(
                    state=s_t, policy_embedding=z_star, reward=reward,
                    next_state=s_next, done=done,
                    next_candidate_embeddings=next_z,
                ))
                for _ in range(self.updates_per_step):
                    loss = self._td_learn()
                    if loss is not None:
                        losses.append(loss)

            # --- decision provenance
            if recorder is not None:
                recorder.record(
                    step=t,
                    state={"l2": float(np.linalg.norm(s_t))},
                    candidates=[c.text for c in candidates],
                    candidate_embeddings="K x 768",
                    q_values=[float(q) for q in q_vals],
                    selected_policy=c_star.text,
                    selected_index=sel,
                    sub_policies=sub_policies,
                    agent_outputs=[o.text for o in agent_outputs],
                    tool_calls=int(exec_info.get("tool_calls", 0)),
                    tool_budget_left=self.tools.budget_left(),
                    fused_output=exec_info["fused_text"],
                    next_observation=next_obs["text"],
                    reward=reward,
                    step_success=bool(env_info.get("step_success", False)),
                    terminal_success=terminal_success,
                    done=bool(done),
                    next_state={"l2": float(np.linalg.norm(s_next))},
                )

            obs = next_obs
            if done:
                break

        tool_after = self.tools.usage()
        return {
            "task_id": obs["task_id"],
            "success": bool(done and env_info.get("success", False)),
            "cum_reward": cum_reward,
            "steps": steps,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "graph": graph,
            "memory": memory,
            "llm_usage": self.llm.usage_summary(),
            "tool_usage": {k: tool_after[k] - tool_before.get(k, 0)
                           for k in tool_after},
            # Provenance, same shape as the baselines: what was submitted and
            # what it was compared against, so a 0 can be audited.
            "answer": last_submission,
            "gold": gold_of(getattr(env, "task", {}) or {}),
            # SWE-bench only: whether the instance container was up, i.e.
            # whether the agent could explore the repository or had to write
            # its patch blind. None on every other benchmark.
            "runtime_ready": getattr(env, "_container_ready", None),
        }

    # -- evaluation ------------------------------------------------------------------
    def _worker(self) -> "CALIX":
        """A parallel-safe view of this model.

        Shares the trained Q-network, the frozen encoder and the embedder
        (all read-only during evaluation) but owns its own LLM client, tool
        budget and RNG, so workers cannot interfere with each other.
        """
        import copy
        w = copy.copy(self)
        w.tools = self.tools.clone()
        w.llm = OpenAICompatibleLLM(
            model=self.cfg["llm"]["model"],
            temperature=self.cfg["llm"]["temperature"],
            top_p=self.cfg["llm"]["top_p"],
            max_tokens=self.cfg["llm"]["max_output_tokens"],
            tools=w.tools,
        )
        w.team = CollaborativeAgentTeam(self.roles, self.embedder, w.llm)
        w._select_rng = np.random.default_rng(self.seed + 5)
        w.replay = None          # never used when training=False
        return w

    def evaluate(self, env: RealEnvironment, num_tasks: int,
                 record_provenance: bool = False,
                 provenance_path: str | None = None,
                 workers: int = 1,
                 env_factory: "Callable[[], RealEnvironment] | None" = None,
                 on_record: "Callable[[dict], None] | None" = None
                 ) -> dict:
        if workers and workers > 1 and env_factory is not None:
            return self._evaluate_parallel(
                num_tasks, workers, env_factory, record_provenance,
                provenance_path, on_record)

        successes, rewards, latencies = [], [], []
        records: list[dict] = []
        recorders: list[ProvenanceRecorder] = []
        breaker = CircuitBreaker()

        def _emit(rec: dict) -> None:
            # A long run must be able to die at any point without losing the
            # work it already finished.  Persist each task the moment it is
            # known, not when the whole cell returns.
            records.append(rec)
            if on_record is not None:
                on_record(rec)

        task_ids = task_ids_of(env)
        for i in range(num_tasks):
            if breaker.tripped:
                tid = task_ids[i] if i < len(task_ids) else f"index-{i}"
                # The runtime has been failing for consecutive tasks; do not
                # spend another half hour each rediscovering the same outage.
                _emit({
                    "task_id": tid, "success": False, "cum_reward": 0.0,
                    "steps": 0, "llm_calls": 0, "llm_prompt_tokens": 0,
                    "llm_completion_tokens": 0, "tool_calls": 0,
                    "answer": "", "gold": "", "runtime_ready": None,
                    "aborted": True,
                    "error": "circuit breaker open after consecutive "
                             "infrastructure aborts",
                })
                continue
            usage_before = dict(self.llm.usage_summary())
            recorder = ProvenanceRecorder() if record_provenance else None
            try:
                with task_deadline(what=f"task {i}"):
                    result = self.run_episode(env, task_index=i,
                                              training=False,
                                              recorder=recorder)
            except Exception as exc:  # noqa: BLE001 - see note below
                # One instance that raises (a pulled-out runtime, an LLM
                # outage) used to abort the whole cell and throw away every
                # row already finished -- an entire method's measurement lost
                # to one bad task.  Record it as aborted instead; those rows
                # are excluded from rates rather than counted as failures.
                tid = f"index-{i}"
                try:
                    tid = env.task["task_id"]
                except Exception:  # noqa: BLE001
                    pass
                print(f"[calix] task {tid} aborted: "
                      f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                breaker.record(True)
                _emit({
                    "task_id": tid, "success": False, "cum_reward": 0.0,
                    "steps": 0, "llm_calls": 0, "llm_prompt_tokens": 0,
                    "llm_completion_tokens": 0, "tool_calls": 0,
                    "answer": "", "gold": "", "runtime_ready": None,
                    "aborted": True,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                })
                continue
            breaker.record(False)
            usage_after = result["llm_usage"]
            successes.append(result["success"])
            rewards.append(result["cum_reward"])
            latencies.append(result["steps"])
            _emit({
                "task_id": result["task_id"],
                "success": result["success"],
                "cum_reward": result["cum_reward"],
                "steps": result["steps"],
                "llm_calls": usage_after["calls"] - usage_before["calls"],
                "llm_prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
                "llm_completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
                "tool_calls": result["tool_usage"]["search_calls"]
                              + result["tool_usage"]["fetch_calls"],
                "answer": str(result.get("answer", ""))[:2000],
                "gold": result.get("gold", ""),
                "runtime_ready": result.get("runtime_ready"),
            })
            if recorder is not None:
                recorders.append(recorder)
                if provenance_path and i == 0:
                    recorder.save(provenance_path)
        return {
            "tsr": 100.0 * float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "avg_steps": float(np.mean(latencies)),
            "records": records,
            "recorders": recorders,
            "llm_usage": self.llm.usage_summary(),
        }

    def _evaluate_parallel(self, num_tasks: int, workers: int,
                           env_factory, record_provenance: bool,
                           provenance_path: str | None,
                           on_record: "Callable[[dict], None] | None" = None
                           ) -> dict:
        """Task-parallel evaluation.

        Tasks are independent, so the result is identical to the sequential
        path; only wall-clock time changes.  Each worker gets its own model
        view (:meth:`_worker`) and its own environment instance.
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading

        # one model view per worker thread (never shared across threads:
        # each owns its LLM client, tool budget and RNG)
        _local = threading.local()
        _made: list[CALIX] = []
        _lock = threading.Lock()

        def get_model() -> "CALIX":
            m = getattr(_local, "model", None)
            if m is None:
                m = self._worker()
                _local.model = m
                with _lock:
                    _made.append(m)
            return m

        breaker = CircuitBreaker()
        parallel_ids = task_ids_of(env_factory())

        def run_one(i: int) -> dict:
            if breaker.tripped:
                tid = (parallel_ids[i] if i < len(parallel_ids)
                       else f"index-{i}")
                rec = {
                    "task_id": tid, "success": False,
                    "cum_reward": 0.0, "steps": 0, "llm_calls": 0,
                    "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
                    "tool_calls": 0, "answer": "", "gold": "",
                    "runtime_ready": None, "aborted": True,
                    "error": "circuit breaker open after consecutive "
                             "infrastructure aborts",
                }
                if on_record is not None:
                    on_record(rec)
                return {"record": rec, "recorder": None}
            model = get_model()
            env = env_factory()
            recorder = ProvenanceRecorder() if record_provenance else None
            before = dict(model.llm.usage_summary())
            try:
                res = model.run_episode(env, task_index=i, training=False,
                                        recorder=recorder)
            except Exception as exc:  # noqa: BLE001 - see note below
                # Same rule as the sequential path: one instance that raises
                # must not take the other workers down with it.  A bare
                # ``pool.map`` propagates the first exception and silently
                # discards every row the batch had already produced, so the
                # isolation has to be inside the worker.
                tid = f"index-{i}"
                try:
                    tid = env.task["task_id"]
                except Exception:  # noqa: BLE001
                    pass
                print(f"[calix] task {tid} aborted: "
                      f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                breaker.record(True)
                rec = {
                    "task_id": tid, "success": False, "cum_reward": 0.0,
                    "steps": 0, "llm_calls": 0, "llm_prompt_tokens": 0,
                    "llm_completion_tokens": 0, "tool_calls": 0,
                    "answer": "", "gold": "", "runtime_ready": None,
                    "aborted": True,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
                if on_record is not None:
                    on_record(rec)
                return {"record": rec, "recorder": recorder}
            after = res["llm_usage"]
            breaker.record(False)
            rec = {
                "task_id": res["task_id"],
                "success": res["success"],
                "cum_reward": res["cum_reward"],
                "steps": res["steps"],
                "llm_calls": after["calls"] - before["calls"],
                "llm_prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                "llm_completion_tokens": after["completion_tokens"] - before["completion_tokens"],
                "tool_calls": res["tool_usage"]["search_calls"]
                              + res["tool_usage"]["fetch_calls"],
                "answer": str(res.get("answer", ""))[:2000],
                "gold": res.get("gold", ""),
                "runtime_ready": res.get("runtime_ready"),
            }
            if on_record is not None:
                on_record(rec)
            return {"record": rec, "recorder": recorder}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(run_one, range(num_tasks)))

        records = [o["record"] for o in out]
        recorders = [o["recorder"] for o in out if o["recorder"] is not None]
        if provenance_path and recorders:
            recorders[0].save(provenance_path)
        total = {"model": self.cfg["llm"]["model"], "calls": 0,
                 "prompt_tokens": 0, "completion_tokens": 0, "tool_calls": 0}
        for m in _made:
            u = m.llm.usage_summary()
            for k in ("calls", "prompt_tokens", "completion_tokens", "tool_calls"):
                total[k] += int(u.get(k, 0))
        return {
            "tsr": 100.0 * float(np.mean([r["success"] for r in records])),
            "avg_reward": float(np.mean([r["cum_reward"] for r in records])),
            "avg_steps": float(np.mean([r["steps"] for r in records])),
            "records": records,
            "recorders": recorders,
            "llm_usage": total,
        }
