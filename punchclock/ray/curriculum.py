"""Curriculum learning module."""
# Ray example:
# https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/curriculum_capable_env.py#L9

# Standard Library Imports
import random
from copy import deepcopy

# Third Party Imports
import numpy as np

# %% Imports
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

# Punch Clock Imports
from punchclock.environment.env import SSAScheduler
from punchclock.ray.build_env import buildEnv


# %% Curriculum function
def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
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
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    # Our env supports tasks 1 (default) to 6.
    # With each task, rewards get scaled up by a factor of 10, such that:
    # Level 1: Expect rewards between 0.0 and 1.0.
    # Level 2: Expect rewards between 1.0 and 10.0, etc..
    # We will thus raise the level/task each time we hit a new power of 10.0
    # new_task = int(np.log10(train_results["episode_reward_mean"]) + 2.1)
    cur_level = task_settable_env.get_task()
    print(f"current level (curriculum_fn) = {cur_level}")
    if train_results["episode_reward_mean"] > 1:
        new_task = cur_level + 1
    else:
        new_task = cur_level

    # Clamp between valid values, just in case:
    new_task = max(min(new_task, 6), 1)
    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task


# %% Curriculum Wrapper
class CurriculumCapableEnv(TaskSettableEnv):
    """Curriculum learning wrapper around SSAScheduler.

    This simply wraps a SSAScheduler env and makes it harder with each
    task. Task (difficulty levels) can range from 1 to 6."""

    # Increase horizon for each level of difficulty
    MAPS = [50, 100, 150, 200, 250, 288]

    def __init__(self, config: EnvContext):
        self.cur_level = config.get("start_level", 1)
        self.horizon = config.get("horizon", 10)
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
        # Make rewards scale with the level exponentially:
        # Level 1: x1
        # Level 2: x10
        # Level 3: x100, etc..
        rew *= 10 ** (self.cur_level - 1)
        if self._timesteps >= self.horizon:
            terminated = True
        return obs, rew, terminated, truncated, info

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

    def _makeEnv(self, config: dict):
        new_config = deepcopy(config)
        new_config["horizon"] = self.MAPS[self.cur_level - 1]
        print(f"{self.cur_level=}")
        print(f"horizon = {new_config['horizon']}")
        del new_config["start_level"]  # prevents error in buildEnv caused by
        # unrecognized arg
        self.env = buildEnv(new_config)
