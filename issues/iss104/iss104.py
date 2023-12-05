"""Demo MaskedGTrXL on small env."""
# %% Imports
# Standard Library Imports
import os

# Third Party Imports
import ray
import ray.rllib.algorithms.ppo as ppo
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe
from punchclock.nets.gtrxl import MaskedGTrXL

# %% Script
ray.init(num_cpus=3)
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = str(2)

ModelCatalog.register_custom_model("MaskedGTrXL", MaskedGTrXL)
register_env("MaskRepeatAfterMe", MaskRepeatAfterMe)

env_config = {"mask_config": "off"}

config = (
    ppo.PPOConfig()
    .environment(MaskRepeatAfterMe, env_config=env_config)
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedGTrXL",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {"max_seq_len": 10},
        }
    )
)
stop = {"training_iteration": 5}

algo = config.build()
algo.train()
algo.stop()
# tuner = tune.Tuner(
#     "PPO",
#     param_space=config.to_dict(),
#     run_config=air.RunConfig(stop=stop, verbose=3),
# )
# tuner.fit()
# ray.shutdown()
# %% done
print("done")
