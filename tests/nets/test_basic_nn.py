"""Test for basic_nn.py"""
# %% Imports
# Third Party Imports
import gymnasium as gym
import torch
from torch.nn import MaxPool1d

# Punch Clock Imports
from scheduler_testbed.nets.basic_nn import MultiDiscreteFeedForward

# %% Setup

# make observation and action space
obs_space = gym.spaces.Box(-1, 1, shape=[5])

# number of values actions can take
num_vals = 3
# number of actions that can be taken
num_actions = 10
der = [num_vals for i in range(num_actions)]
act_space = gym.spaces.MultiDiscrete(der)

print(f"obs_space.sample() = {obs_space.sample()}")
print(f"act_space.sample() = {act_space.sample()}")
print(f"shape of act_space = {act_space.shape}")

# %% Instantiate Model
print("\nInstantiate model...")
my_model = MultiDiscreteFeedForward(
    obs_space=obs_space,
    action_space=act_space,
    model_config={},
    name="der",
)
print(f"observation space size= {my_model.obs_dim}")
print(f"action space size= {my_model.act_dim}")

# %% Test MaxPool1d
print("\nTest custom MaxPool1d()...")

# set kernel to number of values actions can take
# NOTE: kernel = stride
kernel = my_model.num_action_vals
print(f"kernel = {kernel}")
m = MaxPool1d(kernel, return_indices=True)
# 2D input tensor
in_test = torch.randn(
    1, num_vals * num_actions
)  # must be divisible by kernel (stride)
print(f"input = {in_test}")
print(f"input size = {in_test.size()}")
[max_val, indices_manual] = m(in_test)
print(f"max_val = {max_val}")
print(f"manual indices = {indices_manual}")
modulo_indices = indices_manual % num_vals
print(f"manual indices modulo = {modulo_indices}")

indices = my_model.maxPoolIndex1d(in_test)
print(f"indices from method = {indices}")
# %% Test Forward
print("\nTest model.forward()...")

obs = {
    "obs": torch.tensor(obs_space.sample()),
}
print(f"obs = {obs}")
features = my_model.forward(obs)
# Why is features not the size of outputs? It's the size of inputs.
print(f"features = {features}")


# %%
print("done")
