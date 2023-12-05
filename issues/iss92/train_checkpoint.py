"""Create a checkpoint."""
# Standard Library Imports
import random
import string
from pathlib import Path

# Third Party Imports
from ray import air, tune
from ray.rllib.models import ModelCatalog

# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe
from punchclock.nets.lstm_mask import MaskedLSTM

# %% Script
dir_path = Path(__file__).parent
storage_path = dir_path.joinpath("data")

ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

param_space = {
    "framework": "torch",
    "env": "CartPole-v1",
    # "env": MaskRepeatAfterMe,
    "model": {
        # "custom_model": "MaskedLSTM",
        # "custom_model_config": {
        #     "lstm_state_size": 4,
        #     "fcnet_hiddens": [10],
        #     "fcnet_activation": "relu",
        # },
    },
}

rand_str = "".join(random.choices(string.ascii_uppercase, k=3))
exp_name = "training_run_" + rand_str

# Train with normal method
tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=air.RunConfig(
        stop={
            "training_iteration": 2,
        },
        name=exp_name,
        storage_path=storage_path,
    ),
)

results = tuner.fit()
