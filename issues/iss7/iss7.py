"""Issue 7 reproduction script."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiDiscrete
from ray.air import CheckpointConfig, RunConfig
from ray.air.config import FailureConfig
from ray.rllib.algorithms import ppo
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.tune import Tuner
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.utilities import loadJSONFile
from punchclock.environment.wrappers import ActionMask
from punchclock.nets.action_mask_model import MyActionMaskModel
from punchclock.ray.build_tuner import buildEnv

# %% Load config
config = loadJSONFile("issues/iss7/iss7_config.json")

# %% Register env and model
register_env("my_env", buildEnv)
ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)

# %% Modify config
# Disable preprocessor
# config["param_space"]["model"]["_disable_preprocessor_api"] = True
run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_at_end=True,
    ),
    failure_config=FailureConfig(max_failures=2),
    # stop=stopper,
    **config["run_config"],
)

# Custom Env
algo_config = (
    ppo.PPOConfig()
    .training(model={**config["param_space"]["model"]})
    .environment(
        env="my_env",
        env_config=config["param_space"]["env_config"],
    )
    .framework("torch")
)
env = buildEnv(config["param_space"]["env_config"])

# Random Env
# env = RandomEnv(
#     {
#         "observation_space": Dict(
#             {
#                 "observations": Box(0, 1, shape=(36,), dtype=float),
#                 "action_mask": Box(0, 1, shape=(10,), dtype=int),
#             }
#         ),
#         "action_space": MultiDiscrete([10]),
#     }
# )
# algo_config = (
#     ppo.PPOConfig()
#     .training(
#         model={**config["param_space"]["model"]},
#     )
#     .environment(
#         env=RandomEnv,
#         env_config={
#             "observation_space": env.observation_space,
#             "action_space": env.action_space,
#         },
#     )
#     .framework("torch")
# )

algo = algo_config.build()

# test algo
obs = env.observation_space.sample()
action = algo.compute_single_action(obs)
# results = algo.training_step()

tuner = Tuner(
    trainable="PPO",
    param_space=config["param_space"],
    run_config=run_config,
    tune_config=config["tune_config"],
)
tuner.fit()
# %% done
print("done")
