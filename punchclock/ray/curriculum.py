"""Curriculum learning module."""
# Ray example:
# https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/curriculum_capable_env.py#L9
# https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
# https://github.com/ray-project/ray/blob/master/rllib/evaluation/episode_v2.py#L26

# Standard Library Imports
import random
from copy import deepcopy
from dataclasses import dataclass
from operator import itemgetter
from pprint import pprint

# Third Party Imports
from numpy import array, where
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# %% Imports
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override

# Punch Clock Imports
from punchclock.common.utilities import chainedGet, updateRecursive
from punchclock.ray.build_env import buildEnv


# %% ConfigurableCurriculumFnV2
def configurableCurriculumFnV2(
    train_results: dict,
    task_settable_env: TaskSettableEnv,
    env_ctx: EnvContext,
) -> int:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        int: The task to set the env to. This may be the same as the
            current one.
    """
    pprint(train_results)
    curriculum_config, curriculum_map = task_settable_env.getCurriculum()
    cur_task = task_settable_env.get_task()
    metric_val = getMetricValue(
        train_results=train_results, metric=curriculum_config["results_metric"]
    )

    task = updateTask(cur_task, metric_val, curriculum_map=curriculum_map)
    task_config = curriculum_map[task][2]
    metric_threshold = curriculum_map[task][1]

    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nMetric value = {metric_val}"
        f"\nMetric threshold = {metric_threshold}"
        f"\nPrior task {cur_task}"
        f"\nSetting env to task {task}"
        f"\nTask config = {task_config}"
    )
    return task


# %% Custom callback
class CustomCallbacks(DefaultCallbacks):
    """Functions to be called throughout training."""

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",  # noqa
        base_env: BaseEnv,
        policies: dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
    ):
        """Record last values of some custom metrics in every episode.

        Metrics:
            custody_sum (int):
            custody_percent (float):
        """
        pprint(f"episode vars = {vars(episode)}")
        last_info = episode._last_infos
        pprint(f"\n {last_info=}")
        # last_info is a 2-item dict, where the 0th item is empty and the 1st item
        # is info returned from env.
        last_info1 = list(last_info.values())[1]
        # pprint(f"\n {last_info1=}")
        episode.custom_metrics["last_custody_sum"] = last_info1.get("custody_sum", None)
        episode.custom_metrics["last_custody_percent"] = last_info1.get(
            "custody_percent", None
        )
        episode.custom_metrics["curriculum_task"] = last_info1.get("cur_task", None)


# %% ConfigurableCirriculumEnvV2
class ConfigurableCurriculumEnvV2(TaskSettableEnv):
    """Curriculum learning wrapper around SSAScheduler.

    Stores curriculum.

    Task is a dict that is operated on by env_config.update(task).
    """

    def __init__(self, config: EnvContext):
        """Wrap taskable environment.

        Args:
            config (EnvContext): Env config plus some extra items (see Ray documentation
                for EnvContext). Must include
                    "curriculum_config": (CurriculumConfig | dict)
                See CurriculumConfig for details.
        """
        assert "curriculum_config" in config
        cur_config = config.get("curriculum_config")
        assert isinstance(cur_config, (CurriculumConfig, dict))

        # Create curriculum map: list of tuples (index, metric level, task)
        # Convert cur_config to dict if not already
        if isinstance(cur_config, CurriculumConfig):
            self.curriculum_map = cur_config.genCurriculumMap()
            cur_config = cur_config.__dict__
        else:
            self.curriculum_map = CurriculumConfig(**cur_config).genCurriculumMap()

        self.curriculum_config = cur_config

        # Start at task 0 if not provided in config
        self.cur_task = config.get("start_task", self.curriculum_map[0][0])
        # self.cur_task = self.curriculum_map[0][0]
        self.cur_metric_level = self.curriculum_map[0][1]
        self.cur_task_config = self.curriculum_map[0][2]

        self.backup_config = deepcopy(config)
        self.env = None
        self._makeEnv(config)  # create self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):
        """Reset env with possibly new task."""
        if self.switch_env:
            self.switch_env = False
            self._makeEnv(self.backup_config)
        self._timesteps = 0
        obs, info = self.env.reset(seed=seed, options=options)
        info = self._appendTaskToInfo(info)
        return obs, info

    def step(self, action):
        """Step env."""
        self._timesteps += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
        info = self._appendTaskToInfo(info)
        rew = self._transformReward(rew)
        return obs, rew, terminated, truncated, info

    def _makeEnv(self, config: dict):
        """Make environment from backup config with updates from cur_task."""
        new_config = deepcopy(config)
        task = self.cur_task
        task_config = self.curriculum_map[task][2]
        print(f"{task=}")
        print(f"{task_config=}")

        # Use special update function that can handle nested dict tasks
        new_config = updateRecursive(new_config, task_config)
        # prevents error in buildEnv caused by unrecognized arg
        new_config.pop("start_task", None)
        new_config.pop("curriculum_config", None)

        self.env = buildEnv(new_config)

    def _transformReward(self, rew: float) -> float:
        """Optionally transform reward based on curriculum task level."""
        if self.curriculum_config["transform_reward"] is True:
            # cur_level = self.curriculum_config["task_map"].index(self.cur_task)
            new_reward = rew * (self.cur_task + 1)
        else:
            new_reward = rew
        return new_reward

    def _appendTaskToInfo(self, info: dict) -> dict:
        """Append task data to info."""
        task_dict = {
            "cur_task": self.cur_task,
            "cur_task_config": self.cur_task_config,
        }
        new_info = deepcopy(info)
        new_info.update(task_dict)

        return new_info

    def getCurriculum(self) -> tuple[dict, list[tuple]]:
        """Return curriculum config."""
        return self.curriculum_config, self.curriculum_map

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(1, 10) for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self) -> int:
        """Implement this to get the current task (curriculum level)."""
        return self.cur_task

    @override(TaskSettableEnv)
    def set_task(self, task: int):
        """Implement this to set the task (curriculum level) for this env."""
        assert isinstance(task, int)
        self.cur_task = task
        self.cur_task_config = self.curriculum_map[task][2]
        # self.cur_task = task
        self.switch_env = True


@dataclass
class CurriculumConfig:
    """Class for standardizing curriculum inputs that are dicts.

    Args:
        results_metric (list[str]): The path of dict keys to get to the desired
            value in a training results dict. If accessing top level of dict, can
            input a str.
        metric_levels (list[int | float]): The levels of the curriculum
            that the value of the results metric is measured against. Must
            be same length as task_map. Each value is the value that must be met
            or exceeded to move from current task.
        task_map (list[dict]): The map between levels and tasks. Each entry
            must have at least 1 item that is used in the env config. Must
            be same length as metric_levels.
        transform_reward (bool, optional): If True, base reward from environment
            will be transformed. Defaults to False.
    """

    results_metric: list[str] | str
    metric_levels: list[int | float]
    task_map: list[dict]
    transform_reward: bool = False

    def __post_init__(self):
        """Check args."""
        assert isinstance(self.results_metric, (list, str))
        assert len(self.metric_levels) == len(self.task_map)
        if isinstance(self.results_metric, str):
            self.results_metric = [self.results_metric]

    def genCurriculumMap(self) -> list[tuple[int, float, dict]]:
        """Generate curriculum map.

        Curriculum map is list of tuples defining the sequence of tasks associated
        with their metric threshold and a unique index.
        """
        curriculum_map = [
            (t, metric, task_config)
            for t, (metric, task_config) in enumerate(
                zip(self.metric_levels, self.task_map)
            )
        ]

        return curriculum_map


def getMetricValue(train_results: dict, metric: list[str]) -> float | None:
    """Get value of metric from possibly multi-level dict.

    Args:
        train_results (dict): The train results returned by Algorithm.train().
        metric (list[str]): The path of dict keys to get to the desired value in
            a training results dict. If accessing top level of dict, can input a
            str.

    Returns:
        float | None: If key path into multi-level dict is faulty, returns
            None. Otherwise, should return a single numerical value.
    """
    # This seems like a lot of code for a 1-liner function, but the 1 functional
    # line is not self-explanatory.
    return chainedGet(train_results, *metric, default=None)


def updateTask(cur_task: int, metric_val: float, curriculum_map: list[tuple]) -> int:
    """Update task based on metric value and threshold.

    Increments task if metric_val >= threshold. Otherwise, new task is same as
    current. If current task is final in sequence, task is repeated.

    Args:
        cur_task (int): Index of task in curriculum map.
        metric_val (float): Value of evaluation metric. See CurriculumConfig.
        curriculum_map (list[tuple]): Map of task indices, metric thresholds, and
            task configs. See CurriculumConfig.

    Returns:
        int: Index of new task.
    """
    if cur_task == curriculum_map[-1][0]:
        new_task = cur_task
    else:
        # metric_threshold: value metric must meet/exceed to exit current task
        metric_threshold = curriculum_map[cur_task][1]

        if metric_val >= metric_threshold:
            # increment new_task up
            new_task = cur_task + 1
        else:
            # repeat new_task if metric threshold not met
            new_task = cur_task

    return new_task
