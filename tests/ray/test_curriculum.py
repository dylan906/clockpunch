"""Test curriculum.py."""
# %% Imports
# Standard Library Imports
from pathlib import Path

# Third Party Imports
import ray
from gymnasium.spaces import Box, Dict
from ray import air, tune
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils import check_env
from ray.tune.registry import get_trainable_cls, register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.environment.env_parameters import SSASchedulerParams
from punchclock.environment.misc_wrappers import RandomInfo
from punchclock.ray.build_env import buildEnv
from punchclock.ray.curriculum import (
    CurriculumCustodyEnv,
    CustomCallbacks,
    curriculumFnCustody,
)

# %% Test Env
# Need to generate a random SSAScheduler env here, just he base env.
config = SSASchedulerParams(horizon=10, agent_params={}, filter_params={})

env_base = RandomInfo(
    RandomEnv(),
    info_space=Dict({"sum_custody": Box(low=0, high=10, shape=(), dtype=int)}),
)
# %% Tests
# env = CurriculumCustodyEnv(env_config)

# check_env(env)

ray.init(num_cpus=3, num_gpus=0)

config = (
    get_trainable_cls("PPO")
    .get_default_config()  # or "curriculum_env" if registered above
    .environment(
        CurriculumCustodyEnv,
        env_config=env_config,
        env_task_fn=curriculumFnCustody,
    )
    .framework("torch")
    # .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

# os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(1)
stop = {
    "training_iteration": 3,
    # "timesteps_total": 10,
    # "episode_reward_mean": 1,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=3),
)
results = tuner.fit()

# %%
