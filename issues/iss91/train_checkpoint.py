"""Create a checkpoint."""
# Standard Library Imports
import random
import string

# Third Party Imports
from ray import air, tune
from ray.rllib.models import ModelCatalog

# Punch Clock Imports
from issues.iss88.mask_repeat_after_me import MaskRepeatAfterMe
from punchclock.nets.lstm_mask import MaskedLSTM
from punchclock.ray.build_tuner import buildTuner

# %% Script
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)


param_space = {
    "framework": "torch",
    "env": MaskRepeatAfterMe,
    "model": {
        "custom_model": "MaskedLSTM",
        "custom_model_config": {
            "fcnet_hiddens": [6, 6],
            "fcnet_activation": "relu",
            "lstm_state_size": tune.grid_search([2, 4, 6, 8]),
        },
    },
}

rand_str = "".join(random.choices(string.ascii_uppercase, k=3))
exp_name = "training_run_" + rand_str
# tuner = tune.Tuner(
#     trainable="PPO",
#     param_space=param_space,
#     run_config=air.RunConfig(
#         stop={
#             "training_iteration": 10,
#         },
#         name=exp_name,
#     ),
# )

# use buildTuner() instead of manually making Tuner
tuner = buildTuner(
    {
        "trainable": "PPO",
        "param_space": param_space,
        "run_config": {
            "stop": {"training_iteration": 10},
            "name": exp_name,
        },
    },
    override_date=True,
)
results = tuner.fit()
