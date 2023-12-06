"""Demo MaskedGTrXL on small env."""
# %% Imports
# Standard Library Imports
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
print(os.environ["RAY_DEDUP_LOGS"])

# Third Party Imports
import ray
import ray.rllib.algorithms.ppo as ppo
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from ray import air, tune
from ray.rllib.examples.env.action_mask_env import ActionMaskEnv
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe
from punchclock.nets.gtrxl import MaskedGTrXL

# %% Script
# Set Ray logger to not deduplicate all logs

ray.init(num_cpus=3)
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(2)

ModelCatalog.register_custom_model("MaskedGTrXL", MaskedGTrXL)
register_env("MaskRepeatAfterMe", MaskRepeatAfterMe)

env_config_maskrepeat = {"mask_config": "off"}
env_config_rand = {
    "observation_space": Dict(
        {
            "observations": Box(0, 1, shape=[5]),
            "action_mask": Box(0, 1, shape=[2], dtype=int),
        }
    ),
    "action_space": MultiDiscrete([2]),
}

config = (
    ppo.PPOConfig()
    .environment(RandomEnv, env_config=env_config_rand)
    # .environment(MaskRepeatAfterMe, env_config=env_config_maskrepeat)
    # .environment(
    #     ActionMaskEnv,
    #     env_config={
    #         "action_space": Discrete(100),
    #         "observation_space": Box(-1, 1, (5,)),
    #     },
    # )
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedGTrXL",
            # Extra kwargs to be passed to your model's c'tor.
            # "custom_model_config": {"max_seq_len": 10},
        }
    )
)
stop = {"training_iteration": 5}

algo = config.build()
algo.train()
# algo.stop()
# tuner = tune.Tuner(
#     "PPO",
#     param_space=config.to_dict(),
#     run_config=air.RunConfig(stop=stop, verbose=3),
# )
# tuner.fit()
# ray.shutdown()
# %% done
print("done")
