"""Tests for lstm_mask.py."""
# %% Imports

# Third Party Imports
import gymnasium as gym
import ray.rllib.algorithms.ppo as ppo
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
from torch import tensor

# Punch Clock Imports
from punchclock.nets.lstm_mask import MaskedLSTM

torch, nn = try_import_torch()

# %% Make test Env
env_config = {
    "observation_space": gym.spaces.Dict(
        {
            "observations": gym.spaces.Box(0, 1, shape=[4]),
            "action_mask": gym.spaces.Box(0, 1, shape=[2], dtype=int),
        }
    ),
    "action_space": gym.spaces.MultiDiscrete([2]),
}

env = RandomEnv(env_config)
print(f"observation_space = {env.observation_space}")
print(f"action_space = {env.action_space}")


# %% Preprocess function for obs
def preprocessObs(obs: dict) -> dict:
    """Convert components of observation to Tensors."""
    # Obs into .forward() are expected in a slightly different format than the
    # raw env's observation space at instantiation.
    for k, v in obs.items():
        obs[k] = tensor(v)
    obs = {"obs": obs}
    return obs


# %% Build model
model = MaskedLSTM(
    obs_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=2,
    # begin custom config kwargs
    fcnet_hiddens=[6, 6],
    fcnet_activation="relu",
    lstm_state_size=10,
)

# %% Test model (basic)
obs = env.observation_space.sample()
# override action mask to make sure we don't have all same values
obs["action_mask"][0] = 0
obs["action_mask"][1] = 1
print(f"obs (raw) = {obs}")
obs = preprocessObs(obs)
print(f"obs (preprocessed) = {obs}")

seq_lens = tensor(array([0]))
init_state = model.get_initial_state()
[logits, state] = model.forward(
    input_dict=obs, state=init_state, seq_lens=seq_lens
)
print(f"logits = {logits}")
print(f"state = {state}")

# %% Test training
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

config = (
    ppo.PPOConfig()
    .environment(RandomEnv, env_config=env_config)
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedLSTM",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                "fcnet_hiddens": [6, 6],
                "fcnet_activation": "relu",
                "lstm_state_size": 10,
            },
        }
    )
)
algo = config.build()
algo.train()
algo.stop()

# %% Done
print("done")
