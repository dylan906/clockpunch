"""Test for action_mask_model.py."""

# Based on Ray examples:
# https://github.com/ray-project/ray/blob/3e1cdeb117c945ba42df93d629f9a70189e38db9/rllib/examples/models/action_mask_model.py
# https://github.com/ray-project/ray/blob/3e1cdeb117c945ba42df93d629f9a70189e38db9/rllib/examples/action_masking.py
# https://github.com/ray-project/ray/blob/3e1cdeb117c945ba42df93d629f9a70189e38db9/rllib/examples/env/action_mask_env.py
# %% Imports
# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiDiscrete
from gymnasium.spaces.utils import flatdim, flatten, flatten_space
from numpy import array, int64
from ray.rllib.algorithms import ppo
from ray.rllib.examples.env.random_env import RandomEnv
from torch import tensor

# Punch Clock Imports
from punchclock.environment.obs_wrappers import FlatDict
from punchclock.nets.action_mask_model import MyActionMaskModel

# %% Test params
print("\nTest params...")
N = 4
M = 2

obs_space = gym.spaces.Dict(
    {
        # observations must be a primitive gym space (not a Dict)
        "observations": flatten_space(
            gym.spaces.Box(
                low=0,
                high=1,
                shape=(N, M),
                dtype=int64,
            )
        ),
        "action_mask": flatten_space(
            gym.spaces.Box(
                low=0,
                high=1,
                shape=(N + 1, M),
                dtype=int64,
            )
        ),
    }
)
obs_sample = obs_space.sample()
print(
    f"shape of obs_space:\n\
    observations = {obs_sample['observations'].shape} \n\
    action_mask = {obs_sample['action_mask'].shape}"
)

action_space = gym.spaces.MultiDiscrete(array([N + 1] * M), dtype=int64)

for _ in range(5):
    print(f"action_space.sample() = {action_space.sample()}")

num_outputs = flatdim(action_space)
print(f"num_outputs = {num_outputs}")

# model_config must contain "custom_model_config": {} at a minimum. But
# "custom_model_config" can just be an empty dict, but it CANNOT be None. Ray
# defaults will be filled in for non-specified entries of model_config. First
# layer must be size of flattened observations.
obs_space_size = obs_space["observations"].shape[0]
name = "test_model"
# %% Initialize model
print("\nInitializing model...")

model = MyActionMaskModel(
    obs_space=obs_space,
    action_space=action_space,
    num_outputs=num_outputs,
    model_config=None,
    name=name,
    fcnet_hiddens=[15, 20],
    fcnet_activation="relu",
    no_masking=False,
)

# %% Test forward
print("\nTesting forward()...")
# Single observation case
test_obs = {"obs": obs_space.sample()}
test_obs["obs"]["action_mask"] = tensor(test_obs["obs"]["action_mask"])
test_obs["obs"]["observations"] = tensor(test_obs["obs"]["observations"])
print(f"test observation = {test_obs}")

[logits, state] = model.forward(input_dict=test_obs, state=None, seq_lens=None)
print(f"logits = \n{logits}")
print(f"state = {state}")

# %% Test value_function
print("\nTesting value_function()...")

y = model.value_function()
print(f"value = {y}")

# Multiple observations
print("\nTesting multiple observations...")

test_obs = {"obs": [model.obs_space.sample() for i in range(3)]}
# convert ndarray entries to tensors
for item in test_obs["obs"]:
    item["action_mask"] = tensor(item["action_mask"])
    item["observations"] = tensor(item["observations"])

print(f"test observations = {[ob for ob in test_obs['obs']]}")

[logits, state] = model.forward(input_dict=test_obs, state=None, seq_lens=None)
print(f"logits = \n{logits}")
print(f"state = {state}")
# %% Test with masking disabled
print("\nTest masking disabled...")

# model_config = {
# "custom_model_config": {
#     "no_masking": True,
# },
# "no_masking": True,
# "fcnet_hiddens": [15, 20],
# "fcnet_activation": "relu",
# }
model = MyActionMaskModel(
    obs_space=obs_space,
    action_space=action_space,
    num_outputs=num_outputs,
    model_config=None,
    name=name,
    fcnet_hiddens=[15, 20],
    fcnet_activation="relu",
    no_masking=False,
)

print(f"model.no_masking = {model.no_masking}")
# Single observation case
test_obs = {"obs": obs_space.sample()}
test_obs["obs"]["action_mask"] = tensor(test_obs["obs"]["action_mask"])
test_obs["obs"]["observations"] = tensor(test_obs["obs"]["observations"])
print(f"test observation = {test_obs}")

[logits, state] = model.forward(input_dict=test_obs, state=None, seq_lens=None)
print(f"logits = \n{logits}")
print(f"state = {state}")

# %% Test with Full environment
print("\nTest with random environment...")
act_space = MultiDiscrete([2, 2])
obs_space = Dict(
    {
        "observations": Box(-1.0, 1.0, shape=(2,), dtype=int64),
        "action_mask": flatten_space(act_space),
    }
)

rand_env = RandomEnv(
    {
        "observation_space": obs_space,
        "action_space": act_space,
    }
)

rand_model = MyActionMaskModel(
    obs_space=rand_env.observation_space,
    action_space=rand_env.action_space,
    name="my_model",
    num_outputs=obs_space["action_mask"].shape[0],
    fcnet_hiddens=[10, 5],
    fcnet_activation="relu",
)

obs_sample = rand_env.observation_space.sample()
obs_sample["observations"] = tensor(obs_sample["observations"])
obs_sample["action_mask"] = tensor(obs_sample["action_mask"])
obs_sample = {"obs": obs_sample}
print(f"action_mask = {obs_sample['obs']['action_mask']}")

[logits, _] = rand_model.forward(obs_sample, None, None)
print(f"rand model logits = {logits}")
val = rand_model.value_function()
print(f"val = {val}")

# %% Test build algo
# See docs for Policy API:
# https://docs.ray.io/en/latest/rllib/package_ref/policy/policy.html
print("\nTest build algo and compute actions...")

# Make algorithm config, then build algo from config. Ust rand_env as env.
config = (
    ppo.PPOConfig()
    .environment(
        RandomEnv,
        env_config={
            "observation_space": rand_env.observation_space,
            "action_space": rand_env.action_space,
        },
    )
    .training(
        # The ActionMaskModel retrieves the invalid actions and avoids them.
        # When using .training() input, custom_model_kwargs are input via
        # `custom_model_config` dict.
        model={
            "custom_model": MyActionMaskModel,
            "custom_model_config": {
                "fcnet_hiddens": [10, 5],
                "fcnet_activation": "relu",
                "no_masking": False,
            },
        },
    )
    .framework("torch")
)

algo = config.build()
print(f"algo = {algo}")
print("Algorithm built successfully")
print("Training algo...")
algo.training_step()
# for i in range(2):
#     print(f" step {i}")
#     algo.train()
print("...multiple training steps complete.")

# Get policy and calculate actions for a single observation. Note that
# compute_single_action() expects a flattened observation, whereas model.forward()
# expects a raw (unflattened) obs.
pol = algo.get_policy()
obs = rand_env.observation_space.sample()
obs["observations"] = tensor(obs["observations"])
obs["action_mask"] = tensor(obs["action_mask"])
flat_obs = flatten(rand_env.observation_space, obs)
[actions, _, extra_features] = pol.compute_single_action(flat_obs)
print(f"actions = {actions}")
print(f"actions in action_space? {rand_env.action_space.contains(actions)}")

# %% Test that policy isn't choosing invalid actions
# Build env w/ MultiDiscrete action mask.
act_space = MultiDiscrete([2, 2])
obs_space = Dict(
    {
        "observations": Box(-1.0, 1.0, shape=(2,), dtype=int64),
        "action_mask": act_space,
    }
)

md_env = FlatDict(
    RandomEnv(
        {
            "observation_space": obs_space,
            "action_space": act_space,
        }
    )
)


# Emulate a wrapper to ensure that the action mask is 1-hot encoded per action
# space dimension. If we don't do this, then obs["action_mask"] from env.step()
# are not guaranteed to look like a flattened MultiDiscrete.
def wrapOb(ob):
    """Flatten 'action_mask' part of observation."""
    flat_mask = flatten(md_env.observation_space["action_mask"], ob["action_mask"])
    new_ob = deepcopy(ob)
    new_ob["action_mask"] = flat_mask
    return new_ob


def checkForNonVisTaskings(flat_action, action_mask):
    """Return True if non-visible targets tasked."""
    return not (flat_action <= action_mask).all()


# All 1s in flat_action should correspond to 1s in action_mask.
action = md_env.action_space.sample()
for _ in range(5):
    # step MultiDiscrete environment, then manually flatten mask
    [ob, _, _, _, _] = md_env.step(action)
    ob_flat = wrapOb(ob)
    # algo is based on flattened obs["action_mask"], so expects a flat ob
    action = algo.compute_single_action(ob_flat, explore=False)
    flat_action = flatten(md_env.action_space, action)
    print(f"action_mask = {ob_flat['action_mask']}")
    print(f"flat_action = {flat_action}")
    print(
        f"Mask violated? {checkForNonVisTaskings(flat_action, ob_flat['action_mask'],)}"
    )

# %% Done
print("done")
