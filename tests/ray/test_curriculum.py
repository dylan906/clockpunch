"""Test curriculum.py."""
# %% Imports
# Standard Library Imports
import os
from pathlib import Path

# Third Party Imports
import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import check_env
from ray.tune.registry import get_trainable_cls, register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.ray.build_env import buildEnv
from punchclock.ray.curriculum import CurriculumCapableEnv, curriculum_fn

# %% Tests
# register_env("ssa_env", buildEnv)

config_path = Path(__file__).parent.joinpath("config_test").with_suffix(".json")
loaded_config = loadJSONFile(config_path)
env_config = loaded_config["param_space"]["env_config"]
env_config.update({"start_level": 1})

env = CurriculumCapableEnv(env_config)

check_env(env)

ray.init(num_cpus=20, num_gpus=0)

config = (
    get_trainable_cls("PPO")
    .get_default_config()  # or "curriculum_env" if registered above
    .environment(
        CurriculumCapableEnv,
        # env_config={"start_level": 1},
        env_config=env_config,
        env_task_fn=curriculum_fn,
    )
    .framework("torch")
    # .rollouts(num_rollout_workers=1, num_envs_per_worker=1)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

# os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(1)
stop = {
    "training_iteration": 3,
    "timesteps_total": 10,
    "episode_reward_mean": 1,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=3),
)
results = tuner.fit()
