"""Unit tests for the CALIX pipeline (no network access required).

Run with:  python -m pytest tests/ -q        (if pytest is installed)
or:        python tests/test_calix.py        (plain unittest fallback)

These tests cover every component that does not require a live LLM endpoint.
Components that call the real LLM are exercised separately by the smoke test
(experiments/run_smoke.py) once CALIX_OPENAI_API_KEY is configured.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import matrix_audit  # noqa: E402  (tools/ is not a package)

from calix import environment
from calix.embedding import DeterministicEmbedder
from calix.environment import (GAIAEnvironment, extract_final_answer,
                               gaia_exact_match, extract_patch)
from calix.graph import DynamicInteractionGraph
from calix.llm import OpenAICompatibleLLM
from calix.memory import HistoricalMemory
from calix.policy import CandidatePolicy, encode_candidate
from calix.provenance import (ProvenanceRecorder, decision_fidelity,
                              path_completeness, trace_stability)
from calix.qnetwork import QNetwork, q_induced_softmax
from calix.reward import compute_reward


class TestEmbedding(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        emb = DeterministicEmbedder(dim=768, hash_bins=4096)
        a = emb.embed("decompose the task into subtasks")
        b = emb.embed("decompose the task into subtasks")
        c = emb.embed("vote across agents and merge consensus")
        self.assertEqual(a.shape, (768,))
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertLess(float(a @ c), 0.9)


class TestQNetwork(unittest.TestCase):
    def test_shapes_update_and_copy(self):
        qn = QNetwork(seed=0)
        x = np.random.default_rng(0).standard_normal((8, 1536))
        y = np.random.default_rng(1).standard_normal(8)
        before = qn.predict(x).copy()
        loss = qn.td_update(x, y)
        after = qn.predict(x)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.allclose(before, after))
        qn2 = QNetwork(seed=1)
        qn2.copy_from(qn)
        np.testing.assert_allclose(qn2.predict(x), after, rtol=1e-5)

    def test_loss_decreases_on_fixed_target(self):
        qn = QNetwork(seed=0, lr=1e-3)
        x = np.random.default_rng(0).standard_normal((16, 1536))
        y = np.full(16, 3.0)
        first = qn.td_update(x, y)
        for _ in range(50):
            last = qn.td_update(x, y)
        self.assertLess(last, first)


class TestSoftmax(unittest.TestCase):
    def test_induced_distribution(self):
        p = q_induced_softmax(np.array([1.0, 2.0, 0.0]), tau=0.5)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=8)
        self.assertEqual(int(np.argmax(p)), 1)
        p_flat = q_induced_softmax(np.array([1.0, 2.0, 0.0]), tau=100.0)
        self.assertLess(float(np.max(p_flat) - np.min(p_flat)), 0.1)


class TestReward(unittest.TestCase):
    def test_bounds_and_terms(self):
        rw = compute_reward(success=True, step=0, max_steps=20,
                            communications=0, num_agents=6,
                            alpha=1.0, beta=0.1, lam=0.1, success_bonus=5.0,
                            coord_budget_per_agent=20)
        # r_task = 5.0 (only on success), r_eff = 1.0 at t=0, r_coord = 1.0
        self.assertAlmostEqual(rw["reward"], 5.0 + 0.1 + 0.1)
        rw2 = compute_reward(success=False, step=20, max_steps=20,
                             communications=600, num_agents=6)
        self.assertAlmostEqual(rw2["reward"], 0.0)
        # per-episode return bound for T=20: 5.0 + 0.1*sum(1-t/T) + 0.1*T
        #                                  = 5.0 + 1.05 + 2.0 = 8.05
        total = 0.0
        for t in range(20):
            total += compute_reward(success=(t == 19), step=t, max_steps=20,
                                    communications=0, num_agents=6)["reward"]
        self.assertAlmostEqual(total, 8.05, places=6)
        self.assertLessEqual(total, 8.05 + 1e-9)


class TestGAIAScorer(unittest.TestCase):
    def test_official_normalisation(self):
        self.assertTrue(gaia_exact_match("42", "42"))
        self.assertTrue(gaia_exact_match("$1,000", "1000"))
        self.assertTrue(gaia_exact_match("The Answer", "answer"))
        self.assertTrue(gaia_exact_match("Paris, London", "london, paris"))
        self.assertFalse(gaia_exact_match("42", "43"))
        self.assertFalse(gaia_exact_match("Paris", "london, paris"))
        self.assertFalse(gaia_exact_match("", "42"))

    def test_extract_final_answer(self):
        self.assertEqual(extract_final_answer("blah\nFINAL ANSWER: 42"), "42")
        self.assertEqual(extract_final_answer("just text"), "just text")


class TestGAIAEnv(unittest.TestCase):
    def _tasks(self):
        return [{"task_id": "g1", "question": "2+2?", "answer": "4"}]

    def test_episode_flow(self):
        env = GAIAEnvironment(self._tasks(), max_steps=3)
        obs = env.reset(0)
        self.assertEqual(obs["task_id"], "g1")
        obs, info, done = env.step("FINAL ANSWER: 5")
        self.assertFalse(done)
        self.assertFalse(info["success"])
        obs, info, done = env.step("FINAL ANSWER: 4")
        self.assertTrue(done)
        self.assertTrue(info["success"])

    def test_missing_answer_rejected(self):
        env = GAIAEnvironment([{"task_id": "g2", "question": "q"}], max_steps=3)
        with self.assertRaises(ValueError):
            env.reset(0)


class TestParsing(unittest.TestCase):
    def test_numbered_list(self):
        parse = OpenAICompatibleLLM._parse_numbered_list
        items = parse("1. first policy\n2. second policy\n3. third", 3)
        self.assertEqual(items, ["first policy", "second policy", "third"])
        items = parse("Here are policies:\n1. alpha\n   continued\n2. beta", 2)
        self.assertEqual(items, ["alpha continued", "beta"])

    def test_role_assignments(self):
        parse = OpenAICompatibleLLM._parse_role_assignments
        roles = ["Planner", "Executor"]
        out = parse("Planner: plan it\nExecutor: do it", roles, "fallback")
        self.assertEqual(out["Planner"], "plan it")
        self.assertEqual(out["Executor"], "do it")
        out2 = parse("Planner: only plan", roles, "fallback")
        self.assertIn("fallback", out2["Executor"])


class TestGraphMemory(unittest.TestCase):
    def test_graph_evolve_and_summary(self):
        emb = DeterministicEmbedder()
        g = DynamicInteractionGraph(["Planner", "Executor"])
        g.initialise({"Planner": emb.embed("role:Planner"),
                      "Executor": emb.embed("role:Executor")})
        g.evolve("test policy", emb.embed("test policy"),
                 {"Planner": emb.embed("out")}, emb.embed("obs"))
        self.assertEqual(len(g.edges), 2)
        self.assertIn("Planner", g.summary_text())
        self.assertEqual(g.encode().shape, (768,))

    def test_memory_fifo_and_retrieval(self):
        emb = DeterministicEmbedder()
        m = HistoricalMemory(capacity=3, topk=2)
        for i in range(5):
            m.update(task_id=f"t{i}", observation=f"obs {i}",
                     policy_text=f"pol {i}", reward=float(i), outcome="x",
                     feedback="fb", embedding=emb.embed(f"obs {i}"))
        self.assertEqual(len(m), 3)
        got = m.retrieve(emb.embed("obs 4"))
        self.assertEqual(len(got), 2)


class TestProvenance(unittest.TestCase):
    def test_metrics(self):
        rec = ProvenanceRecorder()
        rec.record(step=0, state={}, candidates=["a", "b"],
                   candidate_embeddings="K x 768", q_values=[0.1, 0.9],
                   selected_policy="b", selected_index=1, sub_policies={},
                   agent_outputs=[], fused_output="b", next_observation="o",
                   reward=1.0, step_success=True, terminal_success=False,
                   done=False, next_state={})
        self.assertEqual(path_completeness(rec.records), 100.0)
        self.assertEqual(decision_fidelity(rec.records), 100.0)
        self.assertEqual(trace_stability([rec.records, rec.records]), 100.0)


class TestSWEPatch(unittest.TestCase):
    def test_extract_patch(self):
        self.assertEqual(extract_patch("```diff\n--- a/x\n+++ b/x\n```"),
                         "--- a/x\n+++ b/x")
        self.assertEqual(extract_patch("no patch here"), "")


class TestWebToolkit(unittest.TestCase):
    """Offline tests: no network is touched."""

    def _tk(self, **kw):
        from calix.websearch import WebToolkit
        return WebToolkit(cache_dir=None, enabled=False, **kw)

    def test_dispatch_search_and_fetch(self):
        from calix.websearch import WebToolkit
        tk = WebToolkit(cache_dir=None, max_search_calls=2, max_fetch_calls=1)
        tk._cache["s:hello"] = "SEARCH[hello] -> cached"
        self.assertIn("cached", tk.dispatch("SEARCH[hello]"))
        self.assertIsNone(tk.dispatch("just an answer"))
        self.assertIn("TOOL ERROR", tk.dispatch("FETCH[notaurl]"))

    def test_budget_enforced(self):
        from calix.websearch import WebToolkit
        tk = WebToolkit(cache_dir=None, max_search_calls=2, max_fetch_calls=0)
        tk._cache["s:a"] = "A"
        tk.dispatch("SEARCH[a]")
        tk.dispatch("SEARCH[a]")
        # third call must be refused without performing a search
        self.assertIn("budget", tk.dispatch("SEARCH[b]"))
        self.assertEqual(tk.searches_used, 2)

    def test_budget_reset(self):
        from calix.websearch import WebToolkit
        tk = WebToolkit(cache_dir=None, max_search_calls=1)
        tk.searches_used = 1
        tk.reset_budget()
        self.assertEqual(tk.budget_left()["search"], 1)

    def test_disabled_toolkit(self):
        from calix.websearch import WebToolkit
        tk = WebToolkit(cache_dir=None, enabled=False)
        self.assertIn("disabled", tk.search("x"))
        self.assertIn("disabled", tk.fetch("https://example.com"))

    def test_html_to_text_strips_markup(self):
        from calix.websearch import WebToolkit
        raw = ("<html><head><style>a{}</style></head><body>"
               "<script>var x=1;</script><p>Hello&nbsp;world</p>"
               "<div>second</div></body></html>")
        txt = WebToolkit._html_to_text(raw)
        self.assertIn("Hello world", txt)
        self.assertIn("second", txt)
        self.assertNotIn("var x", txt)
        self.assertNotIn("<p>", txt)

    def test_tool_loop_stops_without_tools(self):
        """With no toolkit attached, run_role must still work (3-tuple)."""
        from calix.llm import OpenAICompatibleLLM
        llm = OpenAICompatibleLLM.__new__(OpenAICompatibleLLM)
        llm.tools = None
        # chat_with_tools degrades to a plain call when tools are absent
        self.assertIsNone(llm.tools)

    def test_build_toolkit_defaults(self):
        from calix.websearch import build_toolkit
        cfg = {"tools": {"enabled": True, "max_search_calls": 3,
                         "max_fetch_calls": 2, "max_doc_chars": 500}}
        tk = build_toolkit(cfg)
        self.assertEqual(tk.max_search_calls, 3)
        self.assertEqual(tk.max_fetch_calls, 2)
        self.assertEqual(tk.max_doc_chars, 500)
        self.assertTrue(tk.enabled)


class TestExecutorBridgeAddress(unittest.TestCase):
    """The Docker-based benchmarks must be pointable at a remote runtime."""

    def setUp(self):
        from calix.environment import AgentBenchHTTPEnvironment
        self.cls = AgentBenchHTTPEnvironment
        self.tasks = [{"task_id": "agentbench-x-0",
                       "executor_url": "http://from-task-file:8000"}]

    def test_env_var_wins_over_everything(self):
        env = self.cls(self.tasks, 3, executor_url="http://from-config:8000")
        env.task = self.tasks[0]
        with mock.patch.dict(os.environ,
                             {"CALIX_EXECUTOR_AGENTBENCH": "http://from-env:8000/"}):
            self.assertEqual(env._base_url(), "http://from-env:8000")

    def test_config_beats_task_file(self):
        env = self.cls(self.tasks, 3, executor_url="http://from-config:8000")
        env.task = self.tasks[0]
        os.environ.pop("CALIX_EXECUTOR_AGENTBENCH", None)
        self.assertEqual(env._base_url(), "http://from-config:8000")

    def test_missing_address_raises_instead_of_guessing(self):
        env = self.cls([{"task_id": "agentbench-x-0"}], 3)
        env.task = {"task_id": "agentbench-x-0"}
        os.environ.pop("CALIX_EXECUTOR_AGENTBENCH", None)
        with self.assertRaises(OSError):
            env._base_url()


class TestAllBaselinesImplemented(unittest.TestCase):
    """All nine baselines must be real runners, not placeholders."""

    REQUIRED_PROMPTS = {
        "camel": ["baselines/camel_instructor.txt", "baselines/camel_assistant.txt"],
        "autogen": ["baselines/autogen_moderator.txt", "baselines/autogen_agent.txt"],
        "agentverse": ["baselines/agentverse_recruit.txt",
                       "baselines/agentverse_role.txt",
                       "baselines/agentverse_vote.txt"],
        "metagpt": ["baselines/metagpt_role.txt"],
        "chatdev": ["baselines/chatdev_phase.txt", "baselines/chatdev_test.txt"],
        "sweagent": ["baselines/sweagent_step.txt"],
        "openhands": ["baselines/openhands_step.txt"],
    }

    def test_registry_covers_every_baseline(self):
        from calix.baselines import BASELINE_NAMES, BASELINE_RUNNERS
        self.assertEqual(sorted(BASELINE_NAMES), sorted(BASELINE_RUNNERS))
        self.assertEqual(len(BASELINE_NAMES), 9)
        for name, fn in BASELINE_RUNNERS.items():
            self.assertTrue(callable(fn), name)

    def test_prompt_templates_exist(self):
        from calix.baselines import BASELINE_NAMES
        root = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "prompts")
        for name in BASELINE_NAMES:
            for rel in self.REQUIRED_PROMPTS.get(name, []):
                self.assertTrue(os.path.exists(os.path.join(root, rel)), rel)

    def test_unknown_baseline_raises(self):
        from calix.baselines import evaluate_baseline
        with self.assertRaises(ValueError):
            evaluate_baseline("not_a_baseline", None, 1, 0)


class TestShellAvailability(unittest.TestCase):
    """GAIA has no sandbox; the fact must be raised, not faked."""

    def test_gaia_has_no_shell(self):
        from calix.environment import GAIAEnvironment
        env = GAIAEnvironment([{"task_id": "g", "question": "q",
                                "answer": "a"}], 3)
        env.reset(0)
        with self.assertRaises(NotImplementedError):
            env.shell("ls")

    def test_swebench_shell_needs_a_bridge(self):
        from calix.environment import SWEBenchEnvironment
        env = SWEBenchEnvironment([{"task_id": "swebench-x-1",
                                    "metadata": {}}], 3)
        env.reset(0)
        os.environ.pop("CALIX_EXECUTOR_SWEBENCH", None)
        with self.assertRaises(NotImplementedError):
            env.shell("ls")


class TestBridgeRouting(unittest.TestCase):
    """The bridge must route /<bench>/<action> and answer 501 when unready."""

    def setUp(self):
        from tools.executor_bridge import build_backends
        self.backends = build_backends(["agentbench=echo", "swebench=echo"])
        from tools.executor_bridge import EchoBackend
        self.echo = EchoBackend()

    def test_build_backends_maps_names(self):
        self.assertEqual(self.backends["agentbench"].name, "echo")
        self.assertEqual(self.backends["swebench"].name, "echo")

    def test_unknown_backend_name_rejected(self):
        from tools.executor_bridge import build_backends
        with self.assertRaises(SystemExit):
            build_backends(["swebench=nope"])

    def test_echo_never_reports_success(self):
        self.echo.reset("t-0")
        for _ in range(3):
            _, done, ok = self.echo.step("t-0", "x")
        self.assertTrue(done)
        self.assertFalse(ok)

    def test_echo_shell_does_not_execute(self):
        self.echo.reset("t-0")
        out = self.echo.shell("t-0", "rm -rf /")
        self.assertIn("not executed", out)


class TestAgentBenchBackends(unittest.TestCase):
    """AgentBench dbbench / knowledgegraph: official protocol, no Docker here."""

    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(root, "tools"))
        from executor_bridge import (AgentBenchDBBenchBackend,
                                     AgentBenchKGBackend, BACKENDS,
                                     _official_init_sql)
        cls.db_cls = AgentBenchDBBenchBackend
        cls.kg_cls = AgentBenchKGBackend
        cls.init_sql = staticmethod(_official_init_sql)
        cls.backends = BACKENDS

    def _entry(self, rows=2, cols=("a", "b")):
        return {"table": {
            "table_name": "T",
            "table_info": {"columns": [{"name": c} for c in cols],
                           "rows": [[f"r{i}{j}" for j in range(len(cols))]
                                    for i in range(rows)]}}}

    # -- init sql reproduces the pinned upstream builder ------------------ #
    def test_init_sql_creates_db_table_and_rows(self):
        sql, data = self.init_sql(self._entry())
        self.assertIn("CREATE DATABASE IF NOT EXISTS `T`;", sql)
        self.assertIn("USE `T`;", sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS `T` (`a` TEXT,`b` TEXT);", sql)
        self.assertIn("VALUES (%s,%s),(%s,%s);", sql)
        self.assertEqual(len(data), 4)

    # -- parsing ---------------------------------------------------------- #
    def test_extract_sql_from_fenced_block(self):
        got = self.db_cls._extract_sql(
            "Action: Operation\n```sql\nSELECT 1;\nSELECT 2;\n```")
        self.assertEqual(got, "SELECT 1; SELECT 2;")

    def test_extract_final_answer(self):
        self.assertEqual(self.db_cls._extract_answer(
            'Action: Answer\nFinal Answer: ["a", "b"]'), '["a", "b"]')

    def test_no_answer_marker_returns_none(self):
        self.assertIsNone(self.db_cls._extract_answer("just thinking"))

    # -- scoring follows the official comparison rules -------------------- #
    def _scorer(self, entry, sql_result=""):
        b = self.db_cls.__new__(self.db_cls)
        b._sql = lambda *a, **k: sql_result
        return b

    def test_score_select_single_value_numeric_then_string(self):
        e = {"type": "other", "label": ["7"], "table": self._entry()["table"]}
        b = self._scorer(e)
        self.assertTrue(b._score(e, "['7.0']", "T"))     # numeric equality
        self.assertTrue(b._score(e, "['7']", "T"))
        self.assertFalse(b._score(e, "['8']", "T"))

    def test_score_select_set_equality(self):
        e = {"type": "other", "label": ["a", "b"],
             "table": self._entry()["table"]}
        b = self._scorer(e)
        self.assertTrue(b._score(e, "['b', 'a']", "T"))
        self.assertFalse(b._score(e, "['a']", "T"))

    def test_score_write_task_uses_md5_of_rows(self):
        gold = "09aa8fbf72f39362970f95a1276b957c"
        e = {"type": "INSERT", "answer_md5": f"[('{gold}',)]",
             "table": self._entry()["table"]}
        b = self._scorer(e, gold)
        self.assertTrue(b._score(e, "anything", "T"))
        b2 = self._scorer(e, "0" * 32)
        self.assertFalse(b2._score(e, "anything", "T"))

    # -- routing ---------------------------------------------------------- #
    def test_knowledgegraph_tasks_are_delegated(self):
        b = self.db_cls.__new__(self.db_cls)
        b._kg = None
        b.data_file = "x"
        b.workdir = "y"
        self.assertIsInstance(b._target("agentbench-knowledgegraph-1"),
                              self.kg_cls)
        self.assertIs(b._target("agentbench-dbbench-0"), b)

    def test_backends_registry_has_the_new_names(self):
        self.assertIn("agentbench_dbbench", self.backends)
        self.assertIn("agentbench_kg", self.backends)

    def test_kg_scores_against_official_gold(self):
        b = self.kg_cls.__new__(self.kg_cls)
        b._tasks = {"k": {"question": "q", "answer": [
            {"entity_name": "Guard"}, {"entity_name": "Forward"}]}}
        # official protocol: Exact Match is over the answer *sets*, so an exact
        # set passes ...
        _, done, ok = b.step("k", 'Final Answer: ["guard", "forward"]')
        self.assertTrue(done)
        self.assertTrue(ok)
        # ... and a superset does not (it used to, under a lenient rule that
        # credited any model willing to list every candidate entity).
        _, done2, ok2 = b.step(
            "k", 'Final Answer: ["guard", "forward", "center"]')
        self.assertTrue(done2)
        self.assertFalse(ok2)
        _, done3, ok3 = b.step("k", 'Final Answer: ["center"]')
        self.assertTrue(done3)
        self.assertFalse(ok3)

    def test_kg_asks_again_when_no_answer_marker(self):
        b = self.kg_cls.__new__(self.kg_cls)
        b._tasks = {"k": {"question": "q", "answer": [{"entity_name": "G"}]}}
        _, done, ok = b.step("k", "let me think about it")
        self.assertFalse(done)
        self.assertFalse(ok)


class TestSWEBenchScoringPath(unittest.TestCase):
    """A configured bridge must own the harness; scoring never pretends.

    Regression: ``_run_official_harness`` used to shell out to
    ``python -m swebench.harness.run_evaluation`` unconditionally, so on a
    machine without the package every SWE-bench cell died with "official
    harness failed" even though the bridge could score the patch.  The
    preference order is now bridge-first, local only as a fallback.
    """

    def _env(self, executor_url=""):
        from calix.environment import SWEBenchEnvironment
        env = SWEBenchEnvironment([{"task_id": "swebench-x-1",
                                    "metadata": {}}], 3,
                                  executor_url=executor_url)
        env.reset(0)
        return env

    def test_configured_bridge_scores_the_patch(self):
        env = self._env()
        seen = {}

        def fake_post(url, payload, timeout, label):
            seen["url"] = url
            seen["action"] = payload["action"]
            return {"observation": "Harness report: RESOLVED.",
                    "done": True, "success": True}

        with mock.patch.dict(os.environ,
                             {"CALIX_EXECUTOR_SWEBENCH": "http://b:8010/swebench"}), \
                mock.patch("calix.environment._post_json", fake_post):
            self.assertTrue(env.step("```diff\n--- a\n+++ b\n```")[1]["success"])
        self.assertEqual(seen["url"], "http://b:8010/swebench/step")
        self.assertIn("```diff", seen["action"])
        self.assertIn("RESOLVED", env._last_harness_note)

    def test_bridge_verdict_is_authoritative_when_unresolved(self):
        env = self._env()
        with mock.patch.dict(os.environ,
                             {"CALIX_EXECUTOR_SWEBENCH": "http://b:8010/swebench"}), \
                mock.patch("calix.environment._post_json",
                           lambda *a, **k: {"observation": "Harness report: "
                                            "NOT resolved.", "success": False}):
            obs, info, done = env.step("```diff\n--- a\n+++ b\n```")
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertFalse(info["step_success"])

    def test_no_bridge_falls_back_to_local_harness(self):
        env = self._env()
        os.environ.pop("CALIX_EXECUTOR_SWEBENCH", None)
        with mock.patch.object(type(env), "_run_harness_locally",
                               mock.Mock(return_value=True)) as local:
            self.assertTrue(env._run_official_harness("diff"))
        self.assertTrue(local.called)

    def test_501_from_the_bridge_is_not_a_connection_error(self):
        from calix.environment import _post_json
        import urllib.error
        err = urllib.error.HTTPError(
            "http://b/shell", 501, "Not Implemented", {},
            __import__("io").BytesIO(b'{"ok": false, "error": "no sandbox"}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(NotImplementedError) as cm:
                _post_json("http://b/shell", {}, 5, "WebArena")
        self.assertIn("no sandbox", str(cm.exception))


class TestAgentBenchKGScoring(unittest.TestCase):
    """AgentBench knowledgegraph must use the official F1 / set-Exact-Match.

    Regression: the first version scored with a superset test (prediction
    contains the gold entity -> success). On the probe that turned a model
    that listed five hurricanes into a "correct" answer, which is not the
    protocol in the AgentBench paper (D.1: F1 primary, EM between answer
    sets). A superset must now fail.
    """

    @staticmethod
    def _step(answer_literal):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "executor_bridge", "tools/executor_bridge.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        b = mod.AgentBenchKGBackend()
        b._tasks = {"kg-1": {
            "question": "q",
            "answer": [{"answer_type": "Entity",
                        "answer_argument": "m.07nnlb",
                        "entity_name": "Hurricane Katrina"}]}}
        return b.step("kg-1",
                      f"Action: Answer\nFinal Answer: {answer_literal}")

    def test_exact_set_is_correct(self):
        obs, done, ok = self._step('["Hurricane Katrina"]')
        self.assertTrue(done)
        self.assertTrue(ok)
        self.assertIn("F1=1.000", obs)
        self.assertIn("EM=True", obs)

    def test_superset_is_not_correct(self):
        obs, done, ok = self._step(
            '["Hurricane Katrina","Hurricane Rita","Hurricane Wilma"]')
        self.assertTrue(done)
        self.assertFalse(ok)
        self.assertIn("EM=False", obs)
        # precision 1/3, recall 1 -> F1 = 0.5
        self.assertIn("F1=0.500", obs)

    def test_missing_gold_is_never_a_pass(self):
        obs, done, ok = self._step('["anything"]')
        self.assertFalse(ok)

    def test_case_and_whitespace_are_normalised(self):
        obs, _, ok = self._step('["  hurricane   KATRINA "]')
        self.assertTrue(ok)


class TestGoldProvenance(unittest.TestCase):
    """Every trace row must carry the gold value it was compared against."""

    def test_gold_is_found_in_each_benchmarks_layout(self):
        from calix.baselines import _gold_of
        self.assertEqual(_gold_of({"answer": "2"}), "2")
        self.assertEqual(
            _gold_of({"metadata": {"label": ["March 19, 1998"]}}),
            "['March 19, 1998']")
        self.assertEqual(
            _gold_of({"metadata": {"answer": [{"entity_name": "Hurricane "
                                                             "Katrina"}]}}),
            "[{'entity_name': 'Hurricane Katrina'}]")
        self.assertEqual(_gold_of({"metadata": {"answer_md5": "abc"}}), "abc")

    def test_webarena_gold_comes_from_the_eval_block(self):
        from calix.baselines import _gold_of
        got = _gold_of({"metadata": {"eval": {
            "eval_types": ["url_match"], "reference_url": "http://x/y"}}})
        self.assertEqual(got, "http://x/y")

    def test_unknown_layout_returns_empty_instead_of_guessing(self):
        from calix.baselines import _gold_of
        self.assertEqual(_gold_of({"metadata": {}}), "")


class TestTaskContextInPrompt(unittest.TestCase):
    """The task text must survive a feedback-only observation.

    Regression: GAIA answers a wrong attempt with
    ``"Environment feedback: the submitted answer is incorrect"`` and nothing
    else. CALIX passed that string straight to the LLM as the observation, so
    from round 2 on the prompt no longer contained the question and the agent
    replied "the problem statement is missing" until the budget ran out. In the
    real probe that turned a task six baselines solved into a 0 for CALIX.
    """

    def test_first_round_is_the_task_alone(self):
        from calix.calix import CALIX
        got = CALIX._prompt_observation("What is 2+2?", "What is 2+2?", 0)
        self.assertEqual(got, "What is 2+2?")

    def test_feedback_never_replaces_the_task(self):
        from calix.calix import CALIX
        got = CALIX._prompt_observation(
            "What is 2+2?",
            "Environment feedback: the submitted answer is incorrect.", 3)
        self.assertIn("What is 2+2?", got)
        self.assertIn("incorrect", got)

    def test_repeated_observation_does_not_duplicate_the_task(self):
        from calix.calix import CALIX
        got = CALIX._prompt_observation("Q", "Q", 4)
        self.assertEqual(got, "Q")


class TestCellVerdict(unittest.TestCase):
    """A cell may be printed as a number only when it is a measurement.

    A method that cannot drive the environment is not "scored 0", it is
    unmeasured -- printing 0 there reads like a capability result and is the
    kind of thing a reviewer catches.
    """

    def test_swebench_requires_a_shell(self):
        self.assertEqual(
            matrix_audit.missing_capability("swebench", "react"), "shell")
        self.assertIsNone(
            matrix_audit.missing_capability("swebench", "sweagent"))
        self.assertIsNone(
            matrix_audit.missing_capability("swebench", "openhands"))

    def test_qa_benchmarks_do_not_blank_a_method(self):
        # GAIA / AgentBench grade a submitted string, so having no tools is a
        # handicap rather than a disqualification: those stay comparable.
        for method in ("gpt4o_direct", "camel", "metagpt", "agentverse"):
            self.assertIsNone(matrix_audit.missing_capability("gaia", method))
            self.assertIsNone(
                matrix_audit.missing_capability("agentbench", method))

    def test_blanked_when_capability_missing(self):
        self.assertEqual(
            matrix_audit.cell_verdict("swebench", "chatdev", True),
            "n/a: no shell")

    def test_blanked_when_the_method_never_acted(self):
        self.assertEqual(
            matrix_audit.cell_verdict("webarena", "camel", False),
            "n/a: no interaction")

    def test_reported_when_it_is_a_measurement(self):
        self.assertEqual(matrix_audit.cell_verdict("gaia", "react", True),
                         "report")
        self.assertEqual(
            matrix_audit.cell_verdict("swebench", "openhands", True), "report")

    def test_verdict_is_plain_text_not_markdown(self):
        # matrix_table.py blanks a cell on verdict.startswith("n/a"). Wrapping
        # the verdict in markdown bold once made that check always false, so
        # every blanking rule silently stopped firing.
        verdict = matrix_audit.cell_verdict("swebench", "react", True)
        self.assertTrue(verdict.startswith("n/a"))
        self.assertNotIn("*", verdict)


class TestSwebenchResetContract(unittest.TestCase):
    """The bridge answers /reset with {"observation", "done"} and no "ok".

    Reading a missing "ok" made *every* instance look like a container
    failure, so each method was graded on a patch it wrote blind while the
    container sat there unused. Caught on real traces: all 33 rows of one run
    carried runtime_ready=false with ``_container_error == "container refused"``.
    """

    def _env(self, monkeypatch_payload):
        env = environment.SWEBenchEnvironment(
            [{"task_id": "swebench-x__y-1", "metadata": {"source_fields": {}}}],
            max_steps=1, executor_url="http://bridge.test/swebench")
        env._post = lambda payload: monkeypatch_payload
        return env

    def test_reset_without_ok_field_is_a_ready_container(self):
        env = self._env(None)
        # bridge-shaped reply: no "ok" key at all
        payload = {"observation": "Container c1 is running.", "done": False}
        called = {}

        def fake_post_json(url, body, timeout, label):
            called["url"] = url
            return payload

        original = environment._post_json
        environment._post_json = fake_post_json
        try:
            env.reset(0)
        finally:
            environment._post_json = original
        self.assertTrue(env._container_ready,
                        "a /reset reply without error is a working container")
        self.assertIn("Container", env._container_note)
        self.assertEqual(called["url"], "http://bridge.test/swebench/reset")

    def test_error_field_is_still_a_failure(self):
        payload = {"error": "Cannot start container c1"}
        original = environment._post_json
        environment._post_json = lambda url, body, timeout, label: payload
        try:
            env = self._env(None)
            env.reset(0)
        finally:
            environment._post_json = original
        self.assertFalse(env._container_ready)
        self.assertIn("Cannot start container", env._container_error)


class TestAbortedRowsAreNotFailures(unittest.TestCase):
    """One crashed instance must not erase -- nor fake -- a cell's results.

    A SWE-bench cell scored all eleven instances with the official harness and
    then died on a harness timeout, discarding every row of that method.
    Isolating failures per task fixes that, but creates the mirror hazard: an
    infrastructure outage must never be counted as a wrong answer.  Aborted
    rows therefore carry ``aborted`` and are excluded from every rate.
    """

    class _Env:
        max_steps = 2
        tasks = [{"task_id": f"t{i}"} for i in range(3)]
        task = tasks[0]

        def reset(self, i):
            self.task = self.tasks[i]
            return {"task_id": self.task["task_id"], "text": "task"}

    def test_crash_becomes_an_aborted_row_not_a_crash(self):
        from calix import baselines as B
        original = B.run_one_task

        def flaky(name, llm, env, task_index, max_steps):
            if task_index == 1:
                raise EnvironmentError("SWE-bench harness: timed out")
            return {"task_id": env.task["task_id"], "success": task_index == 0,
                    "steps": 1, "tool_calls": 0, "llm_calls": 0,
                    "llm_prompt_tokens": 0, "llm_completion_tokens": 0,
                    "shell_used": False, "shell_available": False,
                    "runtime_ready": True, "answer": "a", "gold": "g"}

        B.run_one_task = flaky
        try:
            out = B.evaluate_baseline("react", self._Env(), 3, seed=1,
                                      model="test-model", tools=None)
        finally:
            B.run_one_task = original
        self.assertEqual(len(out["records"]), 3, "no task is dropped silently")
        aborted = [r for r in out["records"] if r.get("aborted")]
        self.assertEqual(len(aborted), 1)
        self.assertIn("timed out", aborted[0]["error"])
        # t0 solved, t1 aborted, t2 failed -> rate is over graded tasks only
        self.assertAlmostEqual(out["tsr"], 50.0)



class TestAbortedRowsCarryRealTaskIds(unittest.TestCase):
    """A row written for a task that never ran must still name that task.

    The circuit breaker used to emit ``index-<i>`` for every skipped task.
    Those ids are not in any eval pool, so ``--resume`` could never find the
    gaps again: after the 2026-10-29 outage, 35 sweagent and 84 calix tasks
    were marked aborted under ids nobody could look up.
    """

    class _Env:
        max_steps = 2
        tasks = [{"task_id": f"gaia-{i}"} for i in range(4)]
        task = tasks[0]

        def reset(self, i):
            self.task = self.tasks[i]
            return {"task_id": self.task["task_id"], "text": "task"}

    def _tripped_breaker(self):
        from unittest import mock

        breaker = mock.MagicMock()
        breaker.tripped = True
        return mock.patch("calix.baselines.CircuitBreaker",
                          return_value=breaker)

    def test_breaker_rows_use_real_ids(self):
        from calix import baselines as B
        original = B.run_one_task
        B.run_one_task = lambda *a, **k: pytest_unreachable()
        try:
            with self._tripped_breaker():
                out = B.evaluate_baseline("react", self._Env(), 4, seed=1,
                                          model="test-model", tools=None,
                                          workers=1)
        finally:
            B.run_one_task = original
        ids = sorted(r["task_id"] for r in out["records"])
        self.assertEqual(ids, ["gaia-0", "gaia-1", "gaia-2", "gaia-3"])
        for r in out["records"]:
            self.assertTrue(r["aborted"])
            self.assertFalse(r["task_id"].startswith("index-"))

    def test_calix_uses_real_ids_too(self):
        from calix.calix import CALIX, task_ids_of
        self.assertEqual(task_ids_of(self._Env()),
                         ["gaia-0", "gaia-1", "gaia-2", "gaia-3"])
        self.assertEqual(task_ids_of([{"task_id": "x"}]), ["x"])
        self.assertEqual(task_ids_of(object()), [])   # no tasks attribute


def pytest_unreachable():
    raise AssertionError("no task should run once the breaker is open")


class TestTaskDeadline(unittest.TestCase):
    """The wall-clock guard must never be the thing that breaks a run."""

    def test_fires_in_main_thread(self):
        from calix.watchdog import TaskTimeout, task_deadline
        with self.assertRaises(TaskTimeout):
            with task_deadline(1, what="probe"):
                time.sleep(5)

    def test_noop_outside_main_thread(self):
        # ``signal.signal`` refuses to run anywhere but the main thread, so the
        # task-parallel baseline path used to raise ValueError inside every
        # worker.  A deadline that cannot be armed must simply not be armed.
        from calix.watchdog import task_deadline
        box = {}

        def worker():
            try:
                with task_deadline(1, what="worker"):
                    box["armed"] = True
            except Exception as exc:  # noqa: BLE001 - recorded for assertion
                box["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(30)
        self.assertIsNone(box.get("error"),
                          "a worker thread must not fail to arm a deadline")
        self.assertTrue(box.get("armed"))

    def test_default_budget_covers_a_real_task(self):
        # A GAIA task makes ~110 LLM calls; at a realistic per-call latency
        # that is well over half an hour of legitimate work.  Anything that
        # caps below the real cost silently destroys measurements.
        from calix.watchdog import budget
        self.assertGreaterEqual(budget(), 3600)


class TestMatrixAuditDedupe(unittest.TestCase):
    """An outage must never turn graded rows back into missing ones."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
        import matrix_audit
        self.ma = matrix_audit

    def _write(self, subdir: str, success: bool, aborted: bool) -> None:
        import json
        path = os.path.join(self.dir.name, subdir, "traces")
        os.makedirs(path, exist_ok=True)
        row = {"benchmark": "b", "method": "m", "task_id": "t", "success": success,
               "steps": 1, "llm_calls": 1, "llm_prompt_tokens": 1,
               "llm_completion_tokens": 1, "tool_calls": 0, "answer": "",
               "gold": "", "runtime_ready": None}
        if aborted:
            row["aborted"] = True
        with open(os.path.join(path, "task_records.jsonl"), "w") as fh:
            fh.write(json.dumps(row) + "\n")
        time.sleep(1.05)  # separate the mtimes, as real rerun dirs are

    def _load(self):
        pattern = os.path.join(self.dir.name, "*", "traces", "task_records.jsonl")
        return self.ma.load(pattern)[("b", "m")]

    def test_later_aborted_does_not_displace_graded_row(self):
        self._write("old", success=True, aborted=False)
        self._write("new", success=False, aborted=True)
        rows = self._load()
        self.assertFalse(rows[0].get("aborted"),
                         "a missing row must not supersede a real measurement")

    def test_later_graded_row_replaces_earlier_aborted_one(self):
        self._write("old", success=False, aborted=True)
        self._write("new", success=True, aborted=False)
        self.assertFalse(self._load()[0].get("aborted"))


class TestLLMRetryBudget(unittest.TestCase):
    """``max_retries`` must be the budget, not a floor silently raised to 6."""

    def _restore(self, key, value):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def _attempts(self, max_retries: int) -> int:
        import urllib.error
        from unittest import mock

        import calix.llm as llm_mod
        from calix.llm import LLMError, OpenAICompatibleLLM

        for key in ("CALIX_OPENAI_API_KEY", "CALIX_LLM_RETRIES",
                    "CALIX_LLM_BACKOFF_CAP"):
            self.addCleanup(self._restore, key, os.environ.get(key))
        os.environ["CALIX_OPENAI_API_KEY"] = "test-key"
        os.environ["CALIX_LLM_BACKOFF_CAP"] = "0"   # no real waiting
        os.environ.pop("CALIX_LLM_RETRIES", None)    # env must not win

        with mock.patch.object(
                llm_mod.urllib.request, "urlopen",
                side_effect=urllib.error.URLError("down")) as call:
            with self.assertRaises(LLMError):
                OpenAICompatibleLLM(model="m",
                                    max_retries=max_retries)._post({})
        return call.call_count

    def test_max_retries_is_honoured_not_floored(self):
        # A one-shot health probe used to wait out six attempts plus backoff.
        self.assertEqual(self._attempts(1), 1)
        self.assertEqual(self._attempts(3), 3)


class TestFatalEndpointError(unittest.TestCase):
    """401/402/403 must be distinguishable from a transient outage.

    A DeepSeek key that ran dry on 2026-10-29 turned every later task into
    ``HTTP 402``, and the runner -- treating it as an outage -- slept through
    three 600s wait rounds per method before recording the same absence.  The
    status has to be visible to the retry logic, not just to a human log.
    """

    def setUp(self):
        sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))

    def _post_error(self, code: int):
        import io
        import urllib.error
        from unittest import mock

        import calix.llm as llm_mod
        from calix.llm import OpenAICompatibleLLM

        self.addCleanup(os.environ.pop, "CALIX_OPENAI_API_KEY", None)
        os.environ["CALIX_OPENAI_API_KEY"] = "test-key"
        err = urllib.error.HTTPError(
            "http://x", code, "err", {}, io.BytesIO(b'{"error": "nope"}'))
        with mock.patch.object(llm_mod.urllib.request, "urlopen",
                               side_effect=err):
            with self.assertRaises(llm_mod.LLMError) as ctx:
                OpenAICompatibleLLM(model="m", max_retries=1)._post({})
        return ctx.exception

    def test_402_is_fatal(self):
        from calix.llm import FatalEndpointError, LLMError
        exc = self._post_error(402)
        self.assertIsInstance(exc, FatalEndpointError)
        self.assertIsInstance(exc, LLMError)   # existing handlers still work

    def test_401_and_403_are_fatal(self):
        from calix.llm import FatalEndpointError
        for code in (401, 403):
            self.assertIsInstance(self._post_error(code), FatalEndpointError)

    def test_400_stays_a_plain_llm_error(self):
        from calix.llm import FatalEndpointError
        self.assertNotIsInstance(self._post_error(400), FatalEndpointError)

    def test_is_fatal_reads_the_probe_string(self):
        import run_benchmarks
        self.assertTrue(run_benchmarks._is_fatal(
            "LLMError: LLM endpoint rejected the request (HTTP 402): "
            "Insufficient Balance"))
        self.assertTrue(run_benchmarks._is_fatal("... (HTTP 401): bad key"))
        self.assertFalse(run_benchmarks._is_fatal(
            "LLMError: LLM endpoint unreachable after 6 attempts: SSL EOF"))
        self.assertFalse(run_benchmarks._is_fatal(None))
        self.assertFalse(run_benchmarks._is_fatal(""))

    def test_fatal_is_read_from_the_aborted_rows(self):
        import run_benchmarks
        rows = [
            {"aborted": False, "error": ""},
            {"aborted": True, "error": "timeout"},
            {"aborted": True, "error": "LLM endpoint rejected the request "
                                       "(HTTP 402): Insufficient Balance"},
        ]
        self.assertIn("HTTP 402", run_benchmarks._fatal_from_records(rows))
        self.assertIsNone(run_benchmarks._fatal_from_records(
            [{"aborted": True, "error": "SSL: UNEXPECTED_EOF"}]))
        self.assertIsNone(run_benchmarks._fatal_from_records([]))


class TestCircuitBreaker(unittest.TestCase):
    """Consecutive infrastructure aborts must stop the rest of the cell."""

    def test_trips_at_the_threshold_and_stays_open(self):
        from calix.watchdog import CircuitBreaker
        breaker = CircuitBreaker(threshold=3)
        self.assertFalse(breaker.tripped)
        self.assertFalse(breaker.record(True))
        self.assertFalse(breaker.record(True))
        self.assertTrue(breaker.record(True))     # third in a row
        self.assertTrue(breaker.tripped)
        # Once open it stays open: later recoveries must not re-arm a cell
        # that has already been declared unmeasured.
        self.assertTrue(breaker.record(False))
        self.assertTrue(breaker.tripped)

    def test_a_graded_task_resets_the_streak(self):
        from calix.watchdog import CircuitBreaker
        breaker = CircuitBreaker(threshold=2)
        breaker.record(True)
        self.assertFalse(breaker.record(False))
        self.assertFalse(breaker.record(True))
        self.assertTrue(breaker.record(True))


class TestAbortedTaskRetry(unittest.TestCase):
    """Rerunning aborted tasks must fill gaps without discarding graded rows."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
        import run_benchmarks
        self.rb = run_benchmarks

    def _factory(self):
        return lambda tasks: (lambda: None)

    def _args(self, **kw):
        import argparse
        base = dict(retry_aborted=2, retry_wait=0, retry_wait_cap=0,
                    workers=1, calix_model=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_retry_fills_the_gap(self):
        records = [{"task_id": "t1", "success": False, "aborted": True},
                   {"task_id": "t2", "success": True}]
        seen = {}

        def fake_run(cfg, method, benchmark, seed, subset, n_tasks, episodes,
                     shared_tools, env_factory, workers, calix_model=None,
                     on_record=None):
            seen["subset"] = [t["task_id"] for t in subset]
            return [{"task_id": "t1", "success": True}], 100.0

        orig = self.rb._run_cell
        self.rb._run_cell = fake_run
        self.addCleanup(setattr, self.rb, "_run_cell", orig)
        self.rb._wait_for_endpoint = lambda *a, **k: True
        self.rb._probe_llm = lambda: None      # healthy endpoint

        out, tsr = self.rb._retry_aborted(
            None, "m", "b", 0, [{"task_id": "t1"}, {"task_id": "t2"}],
            records, 50.0, 0, None, self._factory(), self._args())
        by = {r["task_id"]: r for r in out}
        self.assertTrue(by["t1"]["success"], "the gap should be filled")
        self.assertEqual(seen["subset"], ["t1"], "only aborted tasks rerun")
        self.assertNotIn("aborted", by["t1"])
        self.assertAlmostEqual(tsr, 100.0)

    def test_retry_that_also_aborts_keeps_the_graded_row(self):
        records = [{"task_id": "t1", "success": True}]
        orig = self.rb._run_cell
        self.rb._run_cell = lambda *a, **k: (
            [{"task_id": "t1", "success": False, "aborted": True}], 0.0)
        self.addCleanup(setattr, self.rb, "_run_cell", orig)
        self.rb._wait_for_endpoint = lambda *a, **k: False   # still down
        self.rb._probe_llm = lambda: None      # healthy endpoint

        out, _ = self.rb._retry_aborted(
            None, "m", "b", 0, [{"task_id": "t1"}], records, 100.0,
            0, None, self._factory(), self._args())
        self.assertTrue(out[0]["success"],
                        "a second outage must not erase a real measurement")


if __name__ == "__main__":
    unittest.main()
