"""Demo MaskedLSTM on simple env."""
# https://github.com/ray-project/ray/blob/1845f1c9f2533955574a6b946e932511c1e02c48/rllib/examples/env/repeat_after_me_env.py#L6
# %% Imports
# Third Party Imports
import ray.rllib.algorithms.ppo as ppo
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.spaces.utils import flatten, flatten_space
from numpy import array, ones
from ray import air, tune
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.models import ModelCatalog

# Punch Clock Imports
from issues.iss88.mask_repeat_after_me import MaskRepeatAfterMe
from punchclock.nets.lstm_mask import MaskedLSTM

env = MaskRepeatAfterMe()
env.reset()
obs, r, _, _, _ = env.step(env.action_space.sample())
print(f"obs = {obs}")
assert env.observation_space.contains(env.observation_space.sample())
assert env.action_space.contains(env.action_space.sample())
# %% Build Tuner and run fit
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

param_space = {
    "framework": "torch",
    "env": MaskRepeatAfterMe,
    "model": {
        "custom_model": "MaskedLSTM",
        "custom_model_config": {
            "fcnet_hiddens": [6, 6],
            "fcnet_activation": "relu",
            "lstm_state_size": 10,
        },
    },
}
tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=air.RunConfig(
        stop={"training_iteration": 10},
    ),
)
results = tuner.fit()

# %% Done
print("done")
