"""CALIX core: closed-loop adaptive learning (Algorithm 1) built on
behaviour-aware state construction (Algorithm 2).

Ablation switches (paper Section 5.3):
- use_rl=False          -> w/o RL (heuristic candidate selection, no TD update)
- use_graph=False       -> w/o Graph (h_G zeroed in the state)
- use_behavior=False    -> w/o Behavior (no history in the state)
- use_closed_loop=False -> w/o Closed-loop (feedback not propagated into
                           memory / graph / next-state evolution)
"""
from __future__ import annotations

import numpy as np

from .agents import CollaborativeAgentTeam
from .embedding import DeterministicEmbedder
from .environment import BenchmarkEnv
from .graph import DynamicInteractionGraph
from .llm import MockLLM
from .memory import HistoricalMemory
from .policy import encode_candidate
from .provenance import ProvenanceRecorder
from .qnetwork import QNetwork, q_induced_softmax
from .replay import ReplayBuffer, Transition
from .reward import compute_reward
from .state import BehaviorAwareStateEncoder
from .llm import MockLLM, OpenAICompatibleLLM


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
        self.roles = roles[:n] if n <= len(roles) else roles + [f"Agent{i}" for i in range(len(roles), n)]
        self.team = CollaborativeAgentTeam(self.roles, self.embedder, seed=seed + 1)

        st = cfg["state"]
        self.state_encoder = BehaviorAwareStateEncoder(
            self.embedder, state_dim=st["dim"], hidden_dim=st["hidden_dim"],
            seed=st["encoder_seed"], use_graph=use_graph, use_behavior=use_behavior,
        )

        #sim = cfg["simulation"]
        sim = cfg["openai"]
        self.K = num_candidates or cfg["llm"]["num_candidates"]
        backend = cfg["llm"].get ("backend", "openai")
        if backend == "openai":
            self.llm = OpenAICompatibleLLM (
                model=cfg["llm"]["model"],
                temperature=cfg["llm"]["temperature"],
                top_p=cfg["llm"]["top_p"],
                max_tokens=cfg["llm"]["max_output_tokens"],
            )
        else:
            self.llm = MockLLM (
                num_candidates=self.K,
                base_recall=sim["llm_candidate_recall"],
                heuristic_select=sim["llm_heuristic_select"],
                seed=seed + 2,
            )

        q = cfg["qnetwork"]
        self.q_net = QNetwork(q["input_dim"], q["hidden_dim"], lr=q["lr"],
                              grad_clip=q["grad_clip"], seed=seed + 3)
        self.target_q_net = QNetwork(q["input_dim"], q["hidden_dim"], lr=q["lr"],
                                     grad_clip=q["grad_clip"], seed=seed + 3)
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

    # -- helpers ---------------------------------------------------------------
    def _new_episode_structures(self) -> tuple[DynamicInteractionGraph, HistoricalMemory]:
        graph = DynamicInteractionGraph(self.roles, feature_dim=self.embedder.dim)
        role_feats = {r: self.embedder.embed(f"role:{r}") for r in self.roles}
        graph.initialise(role_feats)
        mem = HistoricalMemory(self.cfg["memory"]["capacity"],
                               self.cfg["memory"]["retrieval_topk"])
        return graph, mem

    def _context_quality(self, mem: HistoricalMemory, graph: DynamicInteractionGraph) -> float:
        mem_q = min(1.0, len(mem) / 20.0)
        graph_q = min(1.0, len(graph.nodes) / (len(self.roles) + 1))
        if not self.use_behavior:
            mem_q = 0.0
        if not self.use_graph:
            graph_q = 0.0
        return 0.6 * mem_q + 0.4 * graph_q

    def _q_values(self, s: np.ndarray, z_list: list[np.ndarray]) -> np.ndarray:
        X = np.stack([np.concatenate([s, z]) for z in z_list])
        net = self.q_net
        return net.predict(X)

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
    def run_episode(
        self,
        env: BenchmarkEnv,
        task_index: int | None = None,
        training: bool = True,
        recorder: ProvenanceRecorder | None = None,
        graph: DynamicInteractionGraph | None = None,
        memory: HistoricalMemory | None = None,
    ) -> dict:
        obs = env.reset(task_index)
        if graph is None or memory is None or not self.use_closed_loop:
            graph, memory = self._new_episode_structures()

        cum_reward = 0.0
        steps = 0
        losses: list[float] = []

        for t in range(env.max_steps):
            # --- Stage 1: behaviour-aware state construction (Algorithm 2)
            s_t, aux = self.state_encoder.construct(graph, memory, obs["text"])

            # --- Stage 2: LLM-guided candidate generation + Q evaluation
            ctx_q = self._context_quality(memory, graph)
            candidates = self.llm.generate_candidates(
                correct_strategy=env.required_strategy,
                context_quality=ctx_q,
                num_candidates=self.K,
                hint_text=obs["text"],
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
                chosen = self.llm.heuristic_choose(candidates, env.required_strategy)
                sel = chosen.index
            c_star = candidates[sel]
            z_star = z_list[sel]

            # --- Stage 3: multi-agent collaborative execution
            sub_policies = self.team.decompose(c_star.text)
            strategy_match = c_star.strategy == env.required_strategy
            agent_outputs, exec_info = self.team.execute(
                sub_policies, strategy_match, removed_role=self.removed_role
            )

            # --- environment interaction E(x_t, y_t)
            next_obs, env_info, done, _ = env.step(c_star.strategy, exec_info)
            rw = compute_reward(
                success=env_info["success"], step=t, max_steps=env.max_steps,
                action_success=strategy_match,
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
                    outcome="success" if env_info["success"] else "in_progress",
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
                    correct_strategy=env.required_strategy,
                    context_quality=self._context_quality(memory, graph),
                    num_candidates=self.K, hint_text=next_obs["text"],
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
                    fused_output=exec_info["fused_text"],
                    next_observation=next_obs["text"],
                    reward=reward,
                    next_state={"l2": float(np.linalg.norm(s_next))},
                )

            obs = next_obs
            if done:
                break

        return {
            "task_id": env.task["task_id"],
            "success": bool(env_info["success"]),
            "cum_reward": cum_reward,
            "steps": steps,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "failure_events": list(env.failure_events),
            "graph": graph,
            "memory": memory,
        }

    # -- training over episodes ---------------------------------------------------
    def train(self, env: BenchmarkEnv, episodes: int,
              log_every: int = 10) -> dict:
        rewards = []
        for e in range(episodes):
            result = self.run_episode(env, training=True)
            rewards.append(result["cum_reward"])
        return {"episode_rewards": rewards}

    # -- evaluation ------------------------------------------------------------------
    def evaluate(self, env: BenchmarkEnv, num_tasks: int,
                 record_provenance: bool = False,
                 provenance_path: str | None = None) -> dict:
        successes, rewards, latencies = [], [], []
        recorders: list[ProvenanceRecorder] = []
        all_failures: list[dict] = []
        for i in range(num_tasks):
            recorder = ProvenanceRecorder() if record_provenance else None
            result = self.run_episode(env, task_index=i, training=False,
                                      recorder=recorder)
            successes.append(result["success"])
            rewards.append(result["cum_reward"])
            latencies.append(result["steps"])
            all_failures.extend(result["failure_events"])
            if recorder is not None:
                recorders.append(recorder)
                if provenance_path and i == 0:
                    recorder.save(provenance_path)
        return {
            "tsr": 100.0 * float(np.mean(successes)),
            "avg_reward": float(np.mean(rewards)),
            "avg_steps": float(np.mean(latencies)),
            "failures": all_failures,
            "recorders": recorders,
        }
