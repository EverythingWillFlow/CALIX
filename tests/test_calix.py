"""Lightweight unit tests for the CALIX reproducibility harness.

Run with:  python -m pytest tests/ -q        (if pytest is installed)
or:        python tests/test_calix.py        (plain unittest fallback)
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calix.config import load_config
from calix.embedding import DeterministicEmbedder
from calix.environment import BenchmarkEnv
from calix.calix import CALIX
from calix.qnetwork import QNetwork
from calix.reward import compute_reward

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "benchmarks", "agentbench.json")


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
    def test_shapes_and_update(self):
        qn = QNetwork(seed=0)
        x = np.random.default_rng(0).standard_normal((8, 1536))
        y = np.random.default_rng(1).standard_normal(8)
        before = qn.predict(x).copy()
        loss = qn.td_update(x, y)
        after = qn.predict(x)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.allclose(before, after))
        qn2 = QNetwork(seed=0)
        qn2.copy_from(qn)
        np.testing.assert_allclose(qn2.predict(x), after)


class TestReward(unittest.TestCase):
    def test_components(self):
        r = compute_reward(success=True, step=5, max_steps=20,
                           communications=12, num_agents=6, action_success=True)
        self.assertAlmostEqual(r["r_task"], 1.5)
        self.assertAlmostEqual(r["r_eff"], 0.75)
        self.assertAlmostEqual(r["r_coord"], 0.9)
        self.assertAlmostEqual(r["reward"], 1.5 + 0.075 + 0.09 + 5.0)


class TestEnvironment(unittest.TestCase):
    def test_match_beats_mismatch(self):
        tasks = BenchmarkEnv.load_tasks(BENCH)
        env = BenchmarkEnv(tasks, "agentbench", max_steps=20, seed=3)
        env.reset(0)
        for _ in range(20):
            _, info, done, _ = env.step(env.required_strategy,
                                        {"num_active_agents": 6})
            if done:
                break
        self.assertTrue(info["success"])
        env.reset(0)
        wrong = (env.required_strategy + 1) % 8
        for _ in range(20):
            _, info2, done2, _ = env.step(wrong, {"num_active_agents": 6})
            if done2:
                break
        self.assertFalse(info2["success"])
        self.assertEqual(len(env.failure_events), 20)


class TestClosedLoop(unittest.TestCase):
    def test_short_training_runs(self):
        cfg = load_config()
        tasks = BenchmarkEnv.load_tasks(BENCH)
        train = [t for t in tasks if t["split"] == "train"]
        env = BenchmarkEnv(train, "agentbench", max_steps=20, seed=2025)
        cx = CALIX(cfg, seed=2025)
        hist = cx.train(env, episodes=3)
        self.assertEqual(len(hist["episode_rewards"]), 3)
        res = cx.evaluate(BenchmarkEnv(train, "agentbench",
                                                max_steps=20, seed=9), num_tasks=3)
        self.assertIn("tsr", res)
        self.assertIn("avg_reward", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
