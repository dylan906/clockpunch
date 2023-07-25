"""Issue 7 reproduction script."""
# %% Imports

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiDiscrete
from ray.air import RunConfig
from ray.rllib.algorithms import ppo
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.tune import Tuner
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

# %% Modify config (custom env)
run_config = RunConfig(**config["run_config"])
# Disable preprocessor
# config["param_space"]["model"]["_disable_preprocessor_api"] = True

# %% Random Env
env_random = RandomEnv(
    {
        "observation_space": Dict(
            {
                "observations": Box(0, 1, shape=(32,), dtype=float),
                "action_mask": Box(0, 1, shape=(10,), dtype=int),
            }
        ),
        "action_space": MultiDiscrete([10]),
    }
)
env_random.step(env_random.action_space.sample())

algo_config_rand = (
    ppo.PPOConfig()
    .training(
        model={"custom_model": "action_mask_model", "fcnet_hiddens": [20, 30]},
    )
    .environment(
        env=RandomEnv,
        env_config={
            "observation_space": env_random.observation_space,
            "action_space": env_random.action_space,
        },
    )
    .framework("torch")
)

algo_random = algo_config_rand.build()
results = algo_random.training_step()
print(f"random env results : \n{results}")

# %% Custom Env
# check environment functionality
env = buildEnv(config["param_space"]["env_config"])
env.reset()
env.step(env.action_space.sample())

# build algo w/ custom env
algo_config_customenv = (
    ppo.PPOConfig()
    .training(
        model={**config["param_space"]["model"]},
    )
    .environment(
        env="my_env",
        env_config=config["param_space"]["env_config"],
    )
    .framework("torch")
)
algo_customenv = algo_config_customenv.build()

try:
    results = algo_customenv.training_step()
    print(f"custom env results : \n{results}")
except Exception as e:
    print(e)

obs = env.observation_space.sample()
action = algo_customenv.compute_single_action(obs)

tuner = Tuner(
    trainable="PPO",
    param_space=config["param_space"],
    run_config=run_config,
    tune_config=config["tune_config"],
)
tuner.fit()
# %% done
print("done")
