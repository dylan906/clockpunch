"""Issue 7 reproduction script."""
# %% Imports
# Third Party Imports
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_tuner import buildEnv

# %% Load config
config = loadJSONFile("issues/iss7/iss7_config.json")

# %% Register env and model
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)


algo_config = (
    ppo.PPOConfig()
    .training(model={**config["param_space"]["model"]})
    .environment(
        env="my_env",
        env_config=config["param_space"]["env_config"],
    )
    .framework("torch")
)

algo = algo_config.build()

env = buildEnv(config["param_space"]["env_config"])
obs = env.observation_space.sample()
action = algo.compute_single_action(obs)
results = algo.training_step()

# %% done
print("done")
