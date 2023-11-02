"""Custom curriculum learning example."""
# %% Imports
# Standard Library Imports
import random
from pprint import pprint

# Third Party Imports
import ray
from gymnasium.spaces import Box, Dict
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# %% Imports
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.policy import Policy
from ray.rllib.utils import check_env
from ray.rllib.utils.annotations import override
from ray.tune.registry import get_trainable_cls

# Punch Clock Imports
from punchclock.environment.misc_wrappers import RandomInfo


# %% Test curriculum_fn
def test_curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Increases env level +1 every call, up to 6.
    """
    cur_level = task_settable_env.get_task()
    print(f"current level (curriculum_fn) = {cur_level}")
    new_task = cur_level + 1

    # Clamp between valid values, just in case:
    new_task = max(min(new_task, 6), 1)
    print(
        f"Worker #{env_ctx.worker_index} vec-idx={env_ctx.vector_index}"
        f"\nR={train_results['episode_reward_mean']}"
        f"\nSetting env to task={new_task}"
    )
    return new_task


# %% Test Env
class TestCurriculumEnv(TaskSettableEnv):
    """Env gives reward equal to current level set on instantiation."""

    def __init__(self, config: EnvContext):
        """Default level is 1, but is set in config."""
        self.cur_level = config.get("start_level", 1)
        self.horizon = config.get("horizon", 10)
        self.env = None
        self._makeEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):  # noqa
        if self.switch_env:
            self.switch_env = False
            self._makeEnv()
        self._timesteps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):  # noqa
        """Set info_b differently if final step of env."""
        self._timesteps += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["info_b"] = 1
        else:
            info["info_b"] = 0
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

    def _makeEnv(self):
        """Reward is equal to current task (task is a number).

        Info is two random numbers.
        """
        reward = self.get_task()
        print(f"from _makeEnv, {reward=}")
        self.env = RandomInfo(
            RandomEnv(
                {
                    "reward_space": Box(low=reward, high=reward, shape=()),
                }
            ),
            info_space=Dict(
                {
                    "info_a": Box(low=0, high=1),
                    "info_b": Box(low=0, high=1, dtype=int),
                }
            ),
        )


# %% Custom Callbacks to use for custom metrics
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
        """Record last info from episode in custom_metrics."""
        pprint(f"episode vars = {vars(episode)}")
        last_info = episode._last_infos
        pprint(f"\n {last_info=}")
        # last_info is a 2-item dict, where the 0th item is empty and the 1st item
        # is info returned from env.
        last_info1 = list(last_info.values())[1]
        pprint(f"\n {last_info1=}")
        episode.custom_metrics["last_info_a"] = last_info1["info_a"]
        episode.custom_metrics["last_info_b"] = last_info1["info_b"]


# %% Scripts
if __name__ == "__main__":
    env = TestCurriculumEnv(config={})
    check_env(env)

    ray.init(num_cpus=3, num_gpus=0)

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment(
            TestCurriculumEnv,
            env_config={},
            env_task_fn=test_curriculum_fn,
        )
        .callbacks(CustomCallbacks)  # comment out this to skip custom Callbacks
        .framework("torch")
    )

    stop = {
        "training_iteration": 3,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=3),
    )
    results = tuner.fit()

    # %% done
    print("done")
