"""Tests for misc_wrappers.py."""
# %% Imports
# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FilterObservation
from numpy import ones
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.environment.misc_wrappers import (
    CopyObsInfoItem,
    CustodyWrapper,
    IdentityWrapper,
    MaskViolationChecker,
    ModifyObsOrInfo,
    OperatorWrapper,
    RandomInfo,
    getIdentityWrapperEnv,
)
from punchclock.policies.policy_builder import buildSpace

# %% RandomInfo
print("\nTest RandomInfo...")
rand_env = RandomEnv()
_, info_unwrapped = rand_env.reset()
randinfo_env = RandomInfo(rand_env)
_, _, _, _, info_wrapped = randinfo_env.step(randinfo_env.action_space.sample())
print(f"unwrapped info = {info_unwrapped}")
print(f"wrapped info = {info_wrapped}")

randinfo_env = RandomInfo(rand_env, info_space=Dict({"a": Box(0, 1)}))
_, _, _, _, info_wrapped = randinfo_env.step(randinfo_env.action_space.sample())
print(f"wrapped info = {info_wrapped}")

try:
    check_env(randinfo_env)
except Exception as ex:
    print(ex)

# %% Test IdentityWrapper
print("\nTest IdentityWrapper...")
rand_env = RandomEnv({"observation_space": Box(0, 1)})
identity_env = IdentityWrapper(rand_env)

print(f"identity env = {identity_env}")
unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = identity_env.observation(unwrapped_obs)
print(f"unwrapped obs = {unwrapped_obs}")
print(f"wrapped obs = {wrapped_obs}")

obs, reward, term, trunc, info = identity_env.step(
    identity_env.action_space.sample()
)

identity_env = IdentityWrapper(rand_env, id="foo")
print(f"identity env = {identity_env}")
print(f"env.id = {identity_env.id}")

try:
    check_env(identity_env)
except Exception as ex:
    print(ex)


# %% ModifyObsOrInfo
class TestMOI(ModifyObsOrInfo):
    """Test class."""

    def __init__(self, env, obs_info: str):  # noqa
        super().__init__(env=env, obs_info=obs_info)

    def modifyOI(self, obs, info):  # noqa
        info.update({"der": 0})

        return obs, info


rand_env = RandomInfo(
    RandomEnv({"observation_space": Dict({"a": Box(0, 1)})}),
    info_space=Dict({"b": Box(low=0, high=1)}),
)

moi_env = TestMOI(rand_env, obs_info="info")
obs, info = moi_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")
obs, _, _, _, info = moi_env.step(moi_env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")
unwrapped_obs = moi_env.unwrapped.observation_space.sample()
obs = moi_env.observation(unwrapped_obs)
print(f"obs unwrapped= {unwrapped_obs}")
print(f"obs wrapped= {obs}")
# %% CopyObsInfoItem
print("\nTest CopyObsInfoItem...")
info_space_config = {"space": "Box", "low": 0, "high": 3}
rand_env = RandomInfo(
    RandomEnv({"observation_space": Dict({"a": Box(0, 1)})}),
    info_space=Dict({"b": buildSpace(info_space_config)}),
)

print("\n  info -> obs")
appinfo_env = CopyObsInfoItem(
    rand_env,
    copy_from="info",
    copy_to="obs",
    from_key="b",
    info_space_config=info_space_config,
)
obs, info = appinfo_env.reset()
print(f"unwrapped obs space = {rand_env.observation_space}")
print(f"wrapped obs space = {appinfo_env.observation_space}")
print(f"reset obs = {obs}")
print(f"reset info = {info}")
assert appinfo_env.observation_space.contains(obs)

obs, _, _, _, info = appinfo_env.step(appinfo_env.action_space.sample())
print(f"step obs = {obs}")
print(f"step info = {info}")
assert appinfo_env.observation_space.contains(obs)

try:
    check_env(appinfo_env)
except Exception as ex:
    print(ex)

print("\n  obs -> info")
appinfo_env = CopyObsInfoItem(
    rand_env,
    copy_from="obs",
    copy_to="info",
    from_key="a",
)
obs, info = appinfo_env.reset()
print(f"unwrapped obs space = {rand_env.observation_space}")
print(f"wrapped obs space = {appinfo_env.observation_space}")
print(f"reset obs = {obs}")
print(f"reset info = {info}")

obs, _, _, _, info = appinfo_env.step(appinfo_env.action_space.sample())
print(f"step obs = {obs}")
print(f"step info = {info}")

print("\n  obs -> obs")
appinfo_env = CopyObsInfoItem(
    rand_env,
    copy_from="obs",
    copy_to="obs",
    from_key="a",
    to_key="a_copy",
)
obs, info = appinfo_env.reset()
print(f"unwrapped obs space = {rand_env.observation_space}")
print(f"wrapped obs space = {appinfo_env.observation_space}")
print(f"reset obs = {obs}")

obs, _, _, _, info = appinfo_env.step(appinfo_env.action_space.sample())
print(f"step obs = {obs}")

print("\n  info -> info")
appinfo_env = CopyObsInfoItem(
    rand_env,
    copy_from="info",
    copy_to="info",
    from_key="b",
    to_key="b_copy",
)
obs, info = appinfo_env.reset()
print(f"reset info = {info}")

obs, _, _, _, info = appinfo_env.step(appinfo_env.action_space.sample())
print(f"reset info = {info}")

# %% OperatorWrapper
print("\nTest OperatorWrapper...")
rand_env = RandomInfo(
    RandomEnv({"observation_space": Dict({"foo": Box(low=0, high=1)})}),
    info_space=Dict({"a": Box(low=0, high=2, shape=[3])}),
)
op_env = OperatorWrapper(
    rand_env,
    obs_or_info="info",
    func_str="getitem",
    key="a",
    copy_key="b",
    b=1,
)
obs, info = op_env.reset()
print(f"obs = {obs}")
print(f"info = {info}")

# %% MaskViolationChecker
print("\nTest MaskViolationChecker...")
rand_env = RandomInfo(
    RandomEnv(
        {
            "action_space": MultiDiscrete([3, 3, 3]),
        }
    ),
    info_space=Dict({"a": MultiBinary((3, 3))}),
)
mvc_env = MaskViolationChecker(rand_env, mask_key="a")

# Test that action mask violation gets caught
for _ in range(2):
    (obs, reward, termination, truncation, info) = mvc_env.step(
        action=mvc_env.action_space.sample()
    )


# Test with mask = ones (guaranteed pass)
mvc_env.previous_mask = ones((3, 3))
(obs, reward, termination, truncation, info) = mvc_env.step(
    action=mvc_env.action_space.sample()
)
print("Test passed")

# %% Test getIdentityWrapperEnv
print("\nTest getIdentityWrapperEnv...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1, shape=[2, 3]),
            }
        )
    }
)
wrapped_env = FilterObservation(
    IdentityWrapper(FilterObservation(rand_env, ["a"])),
    ["a"],
)

ienv = getIdentityWrapperEnv(wrapped_env)
print(ienv)

try:
    getIdentityWrapperEnv(rand_env)
except Exception as e:
    print(e)

# %% Test CustodyWrapper
print("\nTest CustodyWrapper...")
env_rand = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict({"a": Box(0, 1, shape=(3, 6, 6))}),
            "action_space": MultiDiscrete([4, 4]),
        }
    ),
    info_space=Dict({}),
)
cw = CustodyWrapper(
    env_rand,
    obs_info="obs",
    key="a",
    config={
        "func": "tr_cov",
        "threshold": 1,
    },
)
obs, info = cw.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

print(f"unwrapped obs space = \n{cw.unwrapped.observation_space.keys()}")
print(f"wrapped obs space = \n{cw.observation_space.keys()}")
obs = cw.unwrapped.observation_space.sample()
# Make covariance all positive to be ensure diagonals are properly conditioned.
# obs["est_cov"] = abs(obs["est_cov"])
print(f"wrapped obs = {cw.observation(obs)}")


# %% Done
print("done")
