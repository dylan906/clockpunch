"""Custom curriculum learning example."""
# %% Imports
# Standard Library Imports
import random

# Third Party Imports
import ray
from gymnasium.spaces import Box, Dict
from ray import air, tune

# %% Imports
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils import check_env
from ray.rllib.utils.annotations import override
from ray.tune.registry import get_trainable_cls


# %% Test curriculum_fn
def test_curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to."""
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
    def __init__(self, config: EnvContext):
        self.cur_level = config.get("start_level", 1)
        self.horizon = config.get("horizon", 10)
        self.env = None
        # self.backup_config = deepcopy(config)
        self._makeEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.switch_env = False
        self._timesteps = 0

    def reset(self, *, seed=None, options=None):
        if self.switch_env:
            self.switch_env = False
            self._makeEnv()
        self._timesteps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        self._timesteps += 1
        obs, rew, terminated, truncated, info = self.env.step(action)
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
        reward = self.get_task()
        print(f"from _makeEnv, {reward=}")
        self.env = RandomEnv(
            {
                "observation_space": Dict({"a": Box(low=0, high=5, dtype=int)}),
                "reward_space": Box(low=reward, high=reward, shape=()),
            }
        )


if __name__ == "__main__":
    env = TestCurriculumEnv(config={})
    check_env(env)

    ray.init(num_cpus=3, num_gpus=0)

    config = (
        get_trainable_cls("PPO")
        .get_default_config()  # or "curriculum_env" if registered above
        .environment(
            TestCurriculumEnv,
            env_config={},
            env_task_fn=test_curriculum_fn,
        )
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
