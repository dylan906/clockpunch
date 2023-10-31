"""Test curriculum.py."""
# %% Imports
# Standard Library Imports
from pathlib import Path

# Third Party Imports
import ray
from ray import air, tune
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls, register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.ray.build_env import buildEnv
from punchclock.ray.curriculum import CurriculumCapableEnv, curriculum_fn

# %% Tests
# register_env("ssa_env", buildEnv)

config_path = Path(__file__).parent.joinpath("config_test").with_suffix(".json")
config = loadJSONFile(config_path)
# env_config = config["env_config"]

config.update({"start_level": 1})

env = CurriculumCapableEnv(config["param_space"]["env_config"])

ray.init(num_cpus=2)
config = (
    get_trainable_cls("PPO")
    .get_default_config()  # or "curriculum_env" if registered above
    .environment(
        CurriculumCapableEnv,
        # env_config={"start_level": 1},
        env_config=config,
        env_task_fn=curriculum_fn,
    )
    .framework("torch")
    # .rollouts(num_rollout_workers=2, num_envs_per_worker=5)
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)

stop = {
    "training_iteration": 1,
    # "timesteps_total": args.stop_timesteps,
    # "episode_reward_mean": args.stop_reward,
}

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(stop=stop, verbose=2),
)
results = tuner.fit()
