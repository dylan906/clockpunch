"""Test curriculum.py."""
# %% Imports
# Standard Library Imports

# Third Party Imports
import ray
from gymnasium.spaces import Box, Dict, MultiBinary
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils import check_env
from ray.tune.registry import get_trainable_cls

# Punch Clock Imports
from punchclock.ray.build_env import buildEnv
from punchclock.ray.curriculum import (
    ConfigurableCurriculumEnv,
    ConfigurableCurriculumFn,
    CustomCallbacks,
)

# %% Test Env
print("\nTest base env...")
# Build a test env with the right info space. Check that it can build correctly,
# then test with CurriculumCustodyEnv.
env_config = {
    "horizon": 10,
    "agent_params": {"num_targets": 4},
    "constructor_params": {
        "wrappers": [
            {
                "wrapper": "RandomInfo",
                "wrapper_config": {
                    "info_space": Dict(
                        {
                            "custody_sum": Box(0, 5, dtype=int, shape=()),
                            "custody": MultiBinary(4),
                        }
                    )
                },
            },
            {
                "wrapper": "FlatDict",
            },
        ]
    },
}
env = buildEnv(env_config)
env.reset()
env.step(env.action_space.sample())
check_env(env)

# %% Test ConfigurableCurriculumEnv
env = ConfigurableCurriculumEnv(config=env_config)
obs, info = env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, rew, _, _, info = env.step(env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

env.set_task({"horizon": 42})
env.reset()
task = env.get_task()
print(f"{task=}")

# %% Test ConfigurableCurriculumFn
task_map = [{"horizon": 10}, {"horizon": 20}]
c = ConfigurableCurriculumFn(
    results_metric=["custom_metrics", "last_custody_sum_mean"],
    metric_levels=[0.2, 0.5],
    task_map=task_map,
)
results = {
    "custom_metrics": {
        "last_custody_sum_mean": 1.2,
    },
    "episode_reward_mean": 1,
}
env_ctx = EnvContext(env_config=env_config, worker_index=0)
task = c(train_results=results, task_settable_env=env, env_ctx=env_ctx)
print(f"{task=}")

# %% Test Fit
ray.init(num_cpus=3, num_gpus=0)

algo_config = (
    get_trainable_cls("PPO")
    .get_default_config()  # or "curriculum_env" if registered above
    .environment(
        ConfigurableCurriculumEnv,
        env_config=env_config,
        env_task_fn=c,
    )
    .callbacks(CustomCallbacks)
    .framework("torch")
)

stop = {
    "training_iteration": 1,
    # "timesteps_total": 10,
    # "episode_reward_mean": 1,
}

tuner = tune.Tuner(
    "PPO",
    param_space=algo_config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=3),
)
results = tuner.fit()

# %% Done
print("done")
