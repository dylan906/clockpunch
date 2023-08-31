"""Test for policy_base_class_v2.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete

# Punch Clock Imports
from punchclock.policies.policy_base_class_v2 import CustomPolicy

# %% Test instantiation
print("\nTest __init__()...")
# First test with improper observation space.
try:
    pol = CustomPolicy(
        observation_space=MultiDiscrete(1),
        action_space=MultiDiscrete([1]),
    )
except Exception as err:
    print(err)


# Sub-class CustomPolicy correctly
class TestPolicy(CustomPolicy):
    """A test policy."""

    def computeAction(self, obs):
        """Compute action."""
        return self.action_space.sample()


test_policy = TestPolicy(
    observation_space=Dict(
        {
            "observations": Dict({"a": Box(0, 1)}),
            "action_mask": MultiBinary((1, 1)),
        }
    ),
    action_space=MultiDiscrete([1]),
)
print(f"instantiated policy: {test_policy}")

# %% done
print("done")
