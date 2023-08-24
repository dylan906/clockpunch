"""Test for policy_base_class_v2.py."""
# %% Imports
# Third Party Imports
import gymnasium as gym

# Punch Clock Imports
from punchclock.environment.obs_wrappers import MakeDict
from punchclock.policies.policy_base_class_v2 import CustomPolicy


# %% Function for transforming obs wrapper
def doubleObs(obs):
    """Doubles the value of obs."""
    return obs * 2


# %% Test instantiation
print("\nTest __init__()...")
# First test with improper observation space.
# Build an environment that DOESN'T have a dict observation space. Instantiating
# the policy should error.
env = gym.make("CartPole-v1")
env.reset()
try:
    pol = CustomPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
except Exception as err:
    print(err)

# Test with correct observation space
env = MakeDict(env)
env.reset()
pol = CustomPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
)
print(f"obs space = {pol.observation_space}")
print(f"action space = {pol.action_space}")

# %% Test ComputeSingleAction

# %% done
print("done")
