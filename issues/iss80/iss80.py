"""Test nonzero wrapper."""
# %% Imports

# Third Party Imports
from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete

# from gymnasium.utils.env_checker import check_env
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.info_wrappers import TransformInfoWithNumpy
from punchclock.environment.misc_wrappers import (
    CopyObsInfoItem,
    OperatorWrapper,
    RandomInfo,
)

# %% TransformInfoWith Numpy
print("\nTest TransformInfoWithNumpy...")
rand_env = RandomInfo(
    RandomEnv(
        {"observation_space": Dict({}), "action_space": MultiDiscrete([1])}
    ),
    info_space=Dict({"a": MultiBinary((2, 2))}),
)

test_env = CopyObsInfoItem(
    env=rand_env,
    copy_from="info",
    copy_to="info",
    from_key="a",
    to_key="indices",
)
test_env = TransformInfoWithNumpy(
    env=test_env, numpy_func_str="nonzero", key="indices"
)

test_env = OperatorWrapper(
    env=test_env,
    obs_or_info="info",
    func_str="getitem",
    key="a",
    copy_key="non_zeros",
    b_key="indices",
)


(_, _, _, _, info) = test_env.step(test_env.action_space.sample())
print(f"info (via step) = {info}")
