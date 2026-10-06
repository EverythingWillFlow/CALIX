"""
Benchmark environment interface for AgentBench v0.2, GAIA, WebArena v0.2.0,
and SWE-bench v4.1.0. Each environment talks to the official evaluation
harness of the corresponding benchmark.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BenchmarkEnvironment(ABC):
    @abstractmethod
    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        ...

    @abstractmethod
    def evaluate(self, trajectory: Any) -> Dict[str, Any]:
        ...


class AgentBenchEnvironment(BenchmarkEnvironment):
    """AgentBench v0.2 official task worker."""

    def __init__(self, base_url: str = "http://localhost:5000"):
        import requests
        self.requests = requests
        self.base_url = base_url
        self.current_task = None

    def reset(self, task):
        self.current_task = task
        r = self.requests.post(
            f"{self.base_url}/task/reset",
            json={"task_id": task["task_id"]},
        )
        r.raise_for_status()
        return r.json()["observation"]

    def step(self, action):
        r = self.requests.post(
            f"{self.base_url}/task/step",
            json={"action": action},
        )
        r.raise_for_status()
        data = r.json()
        return data["observation"], data["reward"], data["done"], data

    def evaluate(self, trajectory):
        r = self.requests.post(
            f"{self.base_url}/task/evaluate",
            json={"task_id": self.current_task["task_id"],
                  "trajectory": trajectory},
        )
        r.raise_for_status()
        return r.json()


class GAIAEnvironment(BenchmarkEnvironment):
    """GAIA official validator (pinned revision)."""

    def __init__(self):
        self.current_task = None

    def reset(self, task):
        self.current_task = task
        return {"question": task["Question"], "tools": task.get("tools", [])}

    def step(self, action):
        # GAIA is single-turn; the agent's final answer terminates the episode.
        return {}, 0.0, True, {}

    def evaluate(self, trajectory):
        from gaia_official_benchmark import GAIAAnswerValidator
        validator = GAIAAnswerValidator()
        answer = trajectory.get("final_answer", "")
        gold = self.current_task.get("Final answer", "")
        return {"success": validator.validate(answer, gold)}


class WebArenaEnvironment(BenchmarkEnvironment):
    """WebArena v0.2.0 official browser + Docker websites."""

    def __init__(self, config_path: str = "configs/webarena_config.json"):
        import json
        with open(config_path) as f:
            self.config = json.load(f)
        self.env = None
        self.current_task = None

    def reset(self, task):
        from browser_env import ScriptBrowserEnv
        self.current_task = task
        self.env = ScriptBrowserEnv(
            headless=True,
            observation_type="accessibility_tree",
            current_viewport_only=True,
        )
        obs, info = self.env.reset(options={"config_file": task["config_file"]})
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def evaluate(self, trajectory):
        from evaluation_harness import evaluate
        return evaluate(trajectory, self.current_task)


class SWEBenchEnvironment(BenchmarkEnvironment):
    """SWE-bench v4.1.0 official Docker evaluation harness."""

    def __init__(self, dataset: str = "princeton-nlp/SWE-bench"):
        self.dataset = dataset
        self.current_task = None

    def reset(self, task):
        self.current_task = task
        return {
            "repo": task["repo"],
            "base_commit": task["base_commit"],
            "problem_statement": task["problem_statement"],
        }

    def step(self, action):
        return {}, 0.0, False, {}

    def evaluate(self, trajectory):
        import subprocess
        patch_file = trajectory["patch_file"]
        instance_id = self.current_task["instance_id"]
        result = subprocess.run(
            [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset", self.dataset,
                "--predictions_path", patch_file,
                "--instance_ids", instance_id,
                "--max_workers", "1",
            ],
            capture_output=True, text=True,
        )
        return {"success": "resolved" in result.stdout, "raw": result.stdout}


def build_environment(benchmark: str) -> BenchmarkEnvironment:
    if benchmark == "agentbench":
        return AgentBenchEnvironment()
    if benchmark == "gaia":
        return GAIAEnvironment()
    if benchmark == "webarena":
        return WebArenaEnvironment()
    if benchmark == "swebench":
        return SWEBenchEnvironment()
    raise ValueError(f"Unknown benchmark: {benchmark}")