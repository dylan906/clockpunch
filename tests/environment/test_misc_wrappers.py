"""Tests for misc_wrappers.py."""
# %% Imports
# Standard Library Imports
from collections import OrderedDict

# Third Party Imports
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import FilterObservation
from numpy import array, array_equal, inf, nan, ones
from ray.rllib.examples.env.random_env import RandomEnv

# Punch Clock Imports
from punchclock.common.custody_tracker import DebugCustody
from punchclock.environment.misc_wrappers import (
    CheckNanInf,
    ConvertCustody2ActionMask,
    CopyObsInfoItem,
    CustodyWrapper,
    IdentityWrapper,
    MaskViolationChecker,
    ModifyNestedDict,
    ModifyObsOrInfo,
    OperatorWrapper,
    RandomInfo,
    TruncateIfNoCustody,
    VisMap2ActionMask,
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
obs, info = identity_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = identity_env.observation(unwrapped_obs)
print(f"unwrapped obs = {unwrapped_obs}")
print(f"wrapped obs = {wrapped_obs}")

obs, reward, term, trunc, info = identity_env.step(identity_env.action_space.sample())

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

# test with item from info as second arg
rand_env = RandomInfo(
    RandomEnv({"observation_space": Dict({"foo": Box(low=0, high=1)})}),
    info_space=Dict(
        {
            "a": Box(low=0, high=2, shape=[3]),
            "b": MultiBinary([1]),
        }
    ),
)
op_env = OperatorWrapper(
    rand_env,
    obs_or_info="info",
    func_str="getitem",
    key="a",
    copy_key="copy_key",
    b_key="b",
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
mvc_env = MaskViolationChecker(rand_env, mask_key="a", log_violations=True)

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

# Test with mask with 0 column
mvc_env.previous_mask[:, 0] = 0
(obs, reward, termination, truncation, info) = mvc_env.step(
    action=mvc_env.action_space.sample()
)

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

# %% TruncateIfNoCustody
print("\nTest TruncateIfNoCustody...")
env_rand = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict({"a": MultiBinary((1))}),
        }
    ),
    info_space=Dict({"custody_info": Dict({"b": MultiBinary((1))})}),
)
tinc_env = TruncateIfNoCustody(env=env_rand, obs_info="obs", key="a")
obs, info = tinc_env.reset()

obs, _, _, truncate, info = tinc_env.step(tinc_env.action_space.sample())
print(f"obs = {obs}")
print(f"truncate = {truncate}")

# %% Test ConvertCustody2ActionMask
print("\nTest ConvertCustody2ActionMask...")
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict({"custody": MultiBinary(3)}),
            "action_space": MultiDiscrete([4, 4]),
        },
    ),
    info_space=Dict({"custody_alt": MultiBinary(3)}),
)

env_custody2am = ConvertCustody2ActionMask(
    rand_env,
    obs_info="obs",
    key="custody",
    new_key="mask",
)
obs, info = env_custody2am.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = env_custody2am.step(env_custody2am.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

obs_nomask = rand_env.observation_space.sample()
obs_mask = env_custody2am.observation(obs_nomask)
print(f"unwrapped obs  = \n{obs_nomask}")
print(f"wrapped obs = \n{obs_mask}")
assert env_custody2am.observation_space.contains(obs_mask)
assert env_custody2am.mask2d_space.contains(obs_mask["mask"])

# Test with info
env_custody2am = ConvertCustody2ActionMask(
    rand_env,
    obs_info="info",
    key="custody_alt",
    new_key="mask",
)
obs, info = env_custody2am.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = env_custody2am.step(env_custody2am.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

# %% Test VisMap2ActionMask
print("\nTest VisMap2ActionMask...")
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict(
                {"a": MultiBinary((2, 2)), "b": Box(0, 1, shape=(1, 1))}
            ),
            "action_space": MultiDiscrete([3, 3]),
        }
    ),
    info_space=Dict({"a": MultiBinary((2, 2))}),
)
am_env = VisMap2ActionMask(rand_env, obs_info="obs", vis_map_key="a", new_key="a_mask")
obs, info = am_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = am_env.step(am_env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

unmasked_obs = rand_env.observation_space.sample()
obs_mask = am_env.observation(unmasked_obs)
print(f"pre-mask obs space = {rand_env.observation_space}")
print(f"post-mask obs space = {am_env.observation_space}")
assert am_env.observation_space.contains(obs_mask)

# Test with info
print("  Test with info")
am_env = VisMap2ActionMask(rand_env, obs_info="info", vis_map_key="a", new_key="a_mask")
obs, info = am_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = am_env.step(am_env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

unmasked_obs = rand_env.observation_space.sample()
obs_mask = am_env.observation(unmasked_obs)
print(f"pre-mask obs space = {rand_env.observation_space}")
print(f"post-mask obs space = {am_env.observation_space}")
assert am_env.observation_space.contains(obs_mask)

# %% Test combo VisMap2ActionMask, ConvertCustody2ActionMask
# This test checks that the action masks generated by ConvertCustody2ActionMask
# and VisMap2ActionMask are consistent with each other. Meaning that the sensor-action
# pair from one class corresponds to the same pair from the other class.
print("\nTest combo VisMap2ActionMask and ConvertCustody2ActionMask...")

# Test env: 4 targets, 2 sensors
num_targets = 2
num_sensors = 2
rand_env = RandomInfo(
    RandomEnv(
        {
            "observation_space": Dict(
                {
                    "debug_custody": MultiBinary((num_targets)),
                    "debug_vismap": MultiBinary((num_targets, num_sensors)),
                }
            ),
            "action_space": MultiDiscrete([num_targets + 1] * num_sensors),
        },
    )
)

# Wrap with CustodyWrapper, ConvertCustody2ActionMask, VisMap2ActionMask
env_1wrap = CustodyWrapper(
    rand_env,
    obs_info="obs",
    key="debug_custody",
    config={"func": DebugCustody(num_targets=num_targets).update},
)
env_2wrap = ConvertCustody2ActionMask(env_1wrap, obs_info="obs", key="custody")
env_3wrap = VisMap2ActionMask(
    env_2wrap,
    obs_info="obs",
    vis_map_key="debug_vismap",
    new_key="vis_action_mask",
)

for env in [env_1wrap, env_2wrap, env_3wrap]:
    env.reset()

# unwrapped_obs = rand_env.observation_space.sample()
unwrapped_obs = OrderedDict(
    {"debug_custody": array([0, 0]), "debug_vismap": array([[0, 0], [0, 0]])}
)

obs_1wrap = env_1wrap.observation(unwrapped_obs)
obs_2wrap = env_2wrap.observation(obs_1wrap)
obs_3wrap = env_3wrap.observation(obs_2wrap)
print(f"unwrapped obs  = \n{unwrapped_obs}")
print(f"custody obs = \n{obs_1wrap['custody']}")
print(f"custody action mask obs = \n{obs_2wrap['custody']}")
print(f"visibility action mask obs = \n{obs_3wrap['vis_action_mask']}")
assert env_3wrap.observation_space.contains(obs_3wrap)
assert array_equal(obs_2wrap["custody"], obs_3wrap["vis_action_mask"])

# %% Test CheckNanInf
print("\nTest CheckNanInf...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(-inf, inf, shape=(3,)),
                "b": Box(-inf, inf, shape=(3,)),
            }
        ),
    }
)

cni_env = CheckNanInf(rand_env)
obs, info = cni_env.reset()
unwrapped_obs = rand_env.observation_space.sample()
try:
    obs = cni_env.observation(unwrapped_obs)
    print("Test passed")
except Exception as e:
    print("Test failed -- false positive")
    print(e)

unwrapped_obs["a"][0] = nan
try:
    obs = cni_env.observation(unwrapped_obs)
    print("Test failed to identify nan/inf")
except Exception as e:
    print(e)
    print("Test passed")

unwrapped_obs["a"][1] = inf
unwrapped_obs["b"][2] = -inf

try:
    obs = cni_env.observation(unwrapped_obs)
    print("Test failed to identify nan/inf")
except Exception as e:
    print(e)
    print("Test passed")

# %% Test ModifyNestedDict
print("\nTest ModifyNestedDict...")
dict_space = Dict({"a": Dict({"b": Box(-inf, inf, shape=(3,))})})
rand_env = RandomInfo(
    RandomEnv({"observation_space": dict_space}),
    info_space=dict_space,
)
print(f"{rand_env.observation_space=}")
print(f"{rand_env.info_space=}")

# Test info/delete
print("\n   Test info/delete")
mnd_env = ModifyNestedDict(
    env=rand_env, obs_info="info", keys_path=["a", "b"], append_delete="delete"
)
obs, info = mnd_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = mnd_env.step(mnd_env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

# Test info/append
print("\n   Test info/append")
mnd_env = ModifyNestedDict(
    env=rand_env,
    obs_info="info",
    keys_path=["a", "c"],
    append_delete="append",
    value_path=["a", "b"],
)
obs, info = mnd_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

# Test obs/delete
print("\n   Test obs/delete")
mnd_env = ModifyNestedDict(
    env=rand_env, obs_info="obs", keys_path=["a", "b"], append_delete="delete"
)
print(f"{mnd_env.observation_space=}")

obs, info = mnd_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = mnd_env.step(mnd_env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

# Test obs/append
print("\n   Test obs/append")
mnd_env = ModifyNestedDict(
    env=rand_env,
    obs_info="obs",
    keys_path=["a", "c"],
    append_delete="append",
    value_path=["a", "b"],
)
print(f"{mnd_env.observation_space=}")

obs, info = mnd_env.reset()
print(f"obs (reset) = {obs}")
print(f"info (reset) = {info}")

obs, _, _, _, info = mnd_env.step(mnd_env.action_space.sample())
print(f"obs (step) = {obs}")
print(f"info (step) = {info}")

# %% Done
print("done")
