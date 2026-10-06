"""
Baseline executors. Each baseline wraps the official repository of the
corresponding framework. The wrapper is responsible for feeding the
benchmark observation to the framework and returning a trajectory that
the official evaluator can consume.
"""

from abc import ABC, abstractmethod


class BaseBaseline(ABC):
    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def run(self, observation, environment, task) -> dict:
        ...


class GPT4oDirect(BaseBaseline):
    def __init__(self, model: str = "gpt-4o-2024-05-13", **kw):
        super().__init__(**kw)
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def run(self, observation, environment, task):
        prompt = str(observation)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
        return {"final_answer": answer, "trajectory": [answer]}


class ReActBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from react_agent import ReActAgent  # from ysymyth/ReAct
        agent = ReActAgent()
        trajectory = []
        obs = observation
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = environment.step(action)
            trajectory.append({"action": action, "obs": obs, "reward": reward})
        return {"trajectory": trajectory}


class CAMELBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from camel.agents import RolePlaying  # from camel-ai/camel
        role_play = RolePlaying(assistant_role_name="Assistant",
                                user_role_name="User",
                                task_prompt=str(observation))
        session = role_play.init_chat()
        trajectory = []
        for _, msg in session:
            trajectory.append(msg)
        return {"trajectory": trajectory}


class AutoGenBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from autogen import AssistantAgent, UserProxyAgent
        assistant = AssistantAgent("assistant", llm_config={"config_list": []})
        user = UserProxyAgent("user", code_execution_config=False)
        user.initiate_chat(assistant, message=str(observation))
        return {"trajectory": assistant.chat_messages}


class AgentVerseBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from agentverse import AgentVerse  # from OpenBMB/AgentVerse
        av = AgentVerse()
        result = av.run(str(observation))
        return {"trajectory": result}


class MetaGPTBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from metagpt.software_company import generate_repo  # from geekan/MetaGPT
        repo = generate_repo(str(observation))
        return {"trajectory": repo}


class ChatDevBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from chatdev.chat_chain import ChatChain  # from OpenBMB/ChatDev
        chain = ChatChain(config_path="configs/chatdev.yaml",
                          config_phase_path="configs/phase.yaml",
                          config_role_path="configs/role.yaml",
                          task_prompt=str(observation),
                          project_name="calix_task",
                          org_name="CALIX")
        chain.pre_processing()
        chain.make_recruitment()
        chain.execute_chain()
        chain.post_processing()
        return {"trajectory": chain}


class SWEAgentBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from sweagent.agent.agents import DefaultAgent  # from princeton-nlp/SWE-agent
        agent = DefaultAgent()
        result = agent.run(task["problem_statement"])
        return {"trajectory": result, "patch_file": result.get("patch_file")}


class OpenHandsBaseline(BaseBaseline):
    def run(self, observation, environment, task):
        from openhands.evaluation.benchmarks.swe_bench import run_infer  # official
        result = run_infer(
            agent="CodeActAgent",
            config_path="configs/openhands_config.toml",
            task=task,
        )
        return {"trajectory": result}


BASELINES = {
    "gpt4o_direct": GPT4oDirect,
    "react": ReActBaseline,
    "camel": CAMELBaseline,
    "autogen": AutoGenBaseline,
    "agentverse": AgentVerseBaseline,
    "metagpt": MetaGPTBaseline,
    "chatdev": ChatDevBaseline,
    "sweagent": SWEAgentBaseline,
    "openhands": OpenHandsBaseline,
}


def build_baseline(name: str, seed: int = 0) -> BaseBaseline:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    return BASELINES[name](seed=seed)