"""Test gtrxl.py."""
# %% Imports
# Third Party Imports
import gymnasium as gym
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models.torch.attention_net import GTrXLNet
from torch import tensor

# Punch Clock Imports
from punchclock.nets.gtrxl import MaskedGTrXL
from punchclock.nets.utils import preprocessDictObs

# %% Make test Env
print("\nBuild test env...")
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
env.reset()

# # %% Ray example model
# env2 = RandomEnv(
#     {
#         "observation_space": gym.spaces.Box(0, 1, shape=[4]),
#         "action_space": gym.spaces.MultiDiscrete([2]),
#     }
# )
# env2.reset()
# model = GTrXLNet(
#     observation_space=env2.observation_space,
#     action_space=env2.action_space,
#     num_outputs=2,
#     model_config={"max_seq_len": 10},
#     name="derp",
# )

# obs = preprocessDictObs(obs=env2.observation_space.sample())
# obs = env2.observation_space.sample()
# obs = {"obs": tensor(obs)}
# [logits, state] = model.forward(
#     input_dict=obs,
#     state=model.get_initial_state(),
#     seq_lens=tensor(array([1])),
# )

# %% Model
model = MaskedGTrXL(
    observation_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=2,
    model_config={"max_seq_len": 10},
    name="derp",
)

print(f"{model=}")

# %% Test model (basic)
print("\nTest model (basic)...")
obs = env.observation_space.sample()
# override action mask to make sure we don't have all same values
obs["action_mask"][0] = 0
obs["action_mask"][1] = 1
print(f"obs (raw) = {obs}")
obs = preprocessDictObs(obs)
print(f"obs (preprocessed) = {obs}")

seq_lens = tensor(array([1]))
init_state = model.get_initial_state()
[logits, state] = model.forward(input_dict=obs, state=init_state, seq_lens=seq_lens)
print(f"logits = {logits}")
print(f"state = {state}")
# %% done
print("done")
