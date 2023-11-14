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
        pprint(train_results)
        cur_task = task_settable_env.get_task()
        print(f"current task (via curriculum_fn) = {cur_task}")
        metric_val = self.getMetricValue(train_results)
        if cur_task not in self.task_map:
            # Current task can be outside of task map if env was just initialized
            task = self.task_map[0]
        else:
            task = self.incrementTask(cur_task, metric_val)

        print(
            f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
            f"\nR={train_results['episode_reward_mean']}"
            f"\nSetting env to task {task}"
            f"\nMetric value = {metric_val}"
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

    def incrementTask(self, cur_task: dict, metric_val: float) -> dict:
        """Increment a task forward in the task map if metric meets threshold.

        If metric_val doesn't meet threshold, then repeat current task.

        If cur_task is at the end of the curriculum, repeat the current task.
        """
        cur_task_idx = self.task_map.index(cur_task)
        if cur_task_idx == len(self.task_map) - 1:
            # repeat current task if at end of curriculum
            print("End of curriculum reached")
            task = cur_task
        else:
            next_metric_val = self.metric_levels[cur_task_idx + 1]
            if metric_val > next_metric_val:
                # increment task up
                task = self.task_map[cur_task_idx + 1]
            else:
                # repeat task if metric threshold not met
                task = self.task_map[cur_task_idx]

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
        episode.custom_metrics["last_custody_sum"] = last_info1["custody_sum"]
        episode.custom_metrics["last_custody_percent"] = last_info1["custody_percent"]


# %% ConfigurableCirriculumEnv
class ConfigurableCurriculumEnv(TaskSettableEnv):
    """Curriculum learning wrapper around SSAScheduler.

    Task is a dict that is operated on by env_config.update(task).
    """

    def __init__(self, config: EnvContext):
        """Wrap taskable environment.

        Args:
            config (EnvContext): Env config plus some extra items (see Ray documentation
                for EnvContext).
        """
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
        """Reset env with possibly new task."""
        if self.switch_env:
            self.switch_env = False
            self._makeEnv(self.backup_config)
        self._timesteps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Step env."""
        self._timesteps += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated, truncated, info

    def _makeEnv(self, config: dict):
        """Make environment from backup config with updates from cur_task."""
        new_config = deepcopy(config)
        task = self.cur_task
        print(f"{self.cur_task=}")

        # Use special update function that can handle nested dict tasks
        new_config = updateRecursive(new_config, task)
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
