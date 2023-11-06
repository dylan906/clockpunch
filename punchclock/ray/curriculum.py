"""Curriculum learning module."""
# Ray example:
# https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/curriculum_capable_env.py#L9
# https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
# https://github.com/ray-project/ray/blob/master/rllib/evaluation/episode_v2.py#L26

# Standard Library Imports
import random
from copy import deepcopy
from pprint import pprint

# Third Party Imports
from numpy import max, min
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# %% Imports
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override

# Punch Clock Imports
from punchclock.common.utilities import chainedGet, findNearest
from punchclock.ray.build_env import buildEnv


# %% Curriculum class
class ConfigurableCurriculumFn:
    """A configurable curriculum tasking function to be used in a Ray Tuner.

    To be used with ConfigurableCurriculumEnv.

    Usage:
        1. Configure the function on instantiation
        2. When setting Tuner parameters, set the instance as the env_task_fn arg.

    Example:
        curriculumFunc = ConfigurableCurriculum(results_metric, metric_levels, task_map)
        algo_config = (
            get_trainable_cls("PPO")
            .get_default_config()
            .environment(
                ConfigurableCurriculumEnv,
                env_config=env_config,
                env_task_fn=curriculumFunc,
            )
        )
        tuner = tune.Tuner(param_space = algo_config.to_dict())
    """

    def __init__(
        self,
        results_metric: list[str] | str,
        metric_levels: list[int | float],
        task_map: list[dict],
    ):
        """Initialize curriculum to later call via __call__().

        Args:
            results_metric (list[str] | str): The path of keys to get to the desired
                value in a training results dict. If accessing top level of dict,
                can input a str.
            metric_levels (list[int  |  float]): The levels of the curriculum
                that the value of the results metric is measured against. Must
                be same length as task_map.
            task_map (list[dict]): The map between levels and tasks. Each entry
                must have at least 1 item that is used in the env config. Must
                be same length as metric_levels.
        """
        assert len(metric_levels) == len(task_map)
        if isinstance(results_metric, str):
            results_metric = [results_metric]

        self.metric = results_metric
        self.metric_levels = metric_levels
        self.task_map = task_map

    def __call__(
        self,
        train_results: dict,
        task_settable_env: TaskSettableEnv,
        env_ctx: EnvContext,
    ) -> dict:
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
            dict: The task to set the env to. This may be the same as the
                current one.
        """
        pprint(f"{train_results}")
        cur_task = task_settable_env.get_task()
        print(f"current task (via curriculum_fn) = {cur_task}")

        metric_val = self.getMetricValue(train_results)
        closest_level, idx = findNearest(
            x=self.metric_levels,
            val=metric_val,
            round="down",
            return_index=True,
        )
        task = self.task_map[idx]
        print(
            f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
            f"\nR={train_results['episode_reward_mean']}"
            f"\nSetting env to task={task}"
            f"\nMetric value = {metric_val}"
            f"\nNearest metric level = {closest_level}"
        )
        return task

    def getMetricValue(self, train_results: dict) -> float | None:
        """Get value of metric from possibly multi-level dict.

        Args:
            train_results (dict): The train results returned by Algorithm.train().

        Returns:
            float | None: If key path into multi-level dict is faulty, returns
                None. Otherwise, should return a single numerical value.
        """
        # This seems like a lot of code for a 1-liner function, but the 1 functional
        # line is not self-explanatory.
        return chainedGet(train_results, *self.metric, default=None)


# %% Curriculum function
def curriculumFnCustody(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Increases level of difficulty based on mean percent targets in custody at end
    of episode.

    Args:
        train_results: The train results returned by Algorithm.train().
        task_settable_env: A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx: The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    pprint(f"{train_results}")
    cur_level = task_settable_env.get_task()
    print(f"current level (curriculum_fn) = {cur_level}")

    num_targets = task_settable_env.env.num_targets
    percent_custody = (
        train_results["custom_metrics"]["last_custody_sum_mean"] / num_targets
    )
    percent_custody_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    closest_percent, idx = findNearest(
        x=percent_custody_levels,
        val=percent_custody,
        round="down",
        return_index=True,
    )
    new_task = idx

    # Clamp between valid values, just in case:
    new_task = max([min([new_task, len(percent_custody_levels)]), 1])

    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
        f"\nPercent targets in custody = {percent_custody}"
        f"\nNearest custody level = {closest_percent}"
    )
    return new_task


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
        episode.custom_metrics["last_custody_sum"] = last_info1["custody_sum"]
        episode.custom_metrics["last_custody_percent"] = last_info1[
            "custody_sum"
        ] / len(last_info1["custody"])


# %% Curriculum Wrapper
class CurriculumCustodyEnv(TaskSettableEnv):
    """Curriculum learning wrapper around SSAScheduler.

    Wrap a SSAScheduler env and increase horizon with difficult level. Task (difficulty
    levels) can range from 1 to 6.
    """

    # Increase horizon for each level of difficulty
    MAP = [50, 100, 150, 200, 250, 288]

    def __init__(self, config: EnvContext):
        self.cur_level = config.get("start_level", 1)
        self.horizon = config.get("horizon", self.MAP[0])
        self.backup_config = deepcopy(config)
        self.env = None
        self._makeEnv(config)  # create self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):
        if self.switch_env:
            self.switch_env = False
            self._makeEnv(self.backup_config)
        self._timesteps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self._timesteps += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated, truncated, info

    def _makeEnv(self, config: dict):
        new_config = deepcopy(config)
        new_config["horizon"] = self.MAP[self.cur_level - 1]
        print(f"{self.cur_level=}")
        print(f"horizon = {new_config['horizon']}")

        # prevents error in buildEnv caused by unrecognized arg
        new_config.pop("start_level", None)

        self.env = buildEnv(new_config)

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(1, 10) for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        self.cur_level = task
        self.switch_env = True


# %% ConfigurableCirriculumEnv
class ConfigurableCurriculumEnv(TaskSettableEnv):
    """Curriculum learning wrapper around SSAScheduler.

    Task is a dict that is operated on by env_config.update(task).
    """

    def __init__(self, config: EnvContext):
        self.cur_task = config.get("start_task", {})
        assert isinstance(self.cur_task, dict)

        self.backup_config = deepcopy(config)
        self.env = None
        self._makeEnv(config)  # create self.env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):
        if self.switch_env:
            self.switch_env = False
            self._makeEnv(self.backup_config)
        self._timesteps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self._timesteps += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated, truncated, info

    def _makeEnv(self, config: dict):
        new_config = deepcopy(config)
        task = self.cur_task
        print(f"{self.cur_task=}")

        new_config.update(task)
        # prevents error in buildEnv caused by unrecognized arg
        new_config.pop("start_task", None)

        self.env = buildEnv(new_config)

    @override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(1, 10) for _ in range(n_tasks)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_task

    @override(TaskSettableEnv)
    def set_task(self, task):
        """Implement this to set the task (curriculum level) for this env."""
        assert isinstance(task, dict)
        self.cur_task = task
        self.switch_env = True
