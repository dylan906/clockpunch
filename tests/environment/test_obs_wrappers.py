"""Tests for obs_wrappers.py."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from collections import OrderedDict

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from numpy import array, array_equal, inf
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils import check_env

# Punch Clock Imports
from punchclock.common.custody_tracker import DebugCustody
from punchclock.environment.obs_wrappers import (
    Convert2dTo3dObsItems,
    ConvertObsBoxToMultiBinary,
    DiagonalObsItems,
    FlatDict,
    FloatObs,
    LinScaleDictObs,
    MakeDict,
    MakeObsSpaceMultiBinary,
    MinMaxScaleDictObs,
    MultiplyObsItems,
    NestObsItems,
    SplitArrayObs,
    SqueezeObsItems,
    SumArrayWrapper,
    TransformDictObsWithNumpy,
    VisMap2ActionMask,
    WastedActionsMask,
)

# %% Test FloatObs
print("\nTest FloatObs...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "intspace": Box(low=0, high=1, shape=(3, 2), dtype=int),
                "floatspace": Box(low=0, high=1, shape=(3, 2), dtype=float),
                "mbspace": MultiBinary(3),
            }
        )
    }
)

# Test wrapper instantiation
env_float = FloatObs(rand_env)
print(f"unwrapped obs space = {rand_env.observation_space}")
print(f"wrapped obs space ={env_float.observation_space}")

# Test .observation()
obs = rand_env.observation_space.sample()
obs_float = env_float.observation(obs)
print(f"unwrapped obs = {obs}")
print(f"wrapped obs = {obs_float}")
assert env_float.observation_space.contains(obs_float)
# %% Test NestObsItems
print("Test NestObsItems...")
env_prenest = RandomEnv(
    {
        "observation_space": Dict(
            {"a": MultiBinary(3), "b": Box(0, 1), "c": Discrete(4)}
        )
    }
)

# set key_to_nest to different order from observation_space; wrapped env should
# maintain original order of env, regardless of order in keys_to_nest
env_postnest = NestObsItems(
    env=env_prenest, new_key="top_level", keys_to_nest=["c", "a"]
)
obs_nested = env_postnest.observation(env_prenest.observation_space.sample())
assert env_postnest.observation_space.contains(obs_nested)

env_postnest.reset()

# reverse nest
env_postnest = NestObsItems(
    env=env_prenest,
    new_key="top_level",
    keys_to_nest=["c", "a"],
    reverse=True,
)
obs_nested = env_postnest.observation(env_prenest.observation_space.sample())
assert env_postnest.observation_space.contains(obs_nested)

env_postnest.reset()


# %% Test MultiplyObsItems
print("\nTest MultiplyObsItems...")
env_randmask = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "a1": MultiBinary(4),
                "a2": Box(0, 1, shape=(4,), dtype=int),
                "a3": Box(-2, -1, shape=(4,), dtype=int),
            }
        )
    }
)
env_doublemask = MultiplyObsItems(
    env=env_randmask, keys=["a1", "a2"], new_key="foo"
)
obs_randmask = env_randmask.observation_space.sample()
obs_doublemask = env_doublemask.observation(obs=obs_randmask)
print(f"unwrapped obs space = {env_randmask.observation_space}")
print(f"wrapped obs space = {env_doublemask.observation_space}")
assert env_doublemask.observation_space.contains(obs_doublemask)

# %% Test FlatDict
print("\nTest FlatDict...")
# create a separate test env that has a nested dict obs space
env_deep_dict = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "a": gym.spaces.Dict(
                    {
                        "aa": gym.spaces.Box(0, 1, shape=[2, 3], dtype=float),
                        "ab": gym.spaces.Box(0, 1, shape=[2, 3], dtype=int),
                        "ac": gym.spaces.MultiDiscrete([1, 1, 1]),
                    }
                ),
                "b": gym.spaces.Box(0, 1, shape=[2, 3], dtype=int),
                "c": gym.spaces.MultiDiscrete([2, 3, 2]),
            }
        ),
        "action_space": gym.spaces.MultiDiscrete([3, 4, 3]),
    }
)
env_flat = FlatDict(env=env_deep_dict)
print(f"pre-flattend obs space: {env_deep_dict.observation_space}")
print(f"flattend obs space: {env_flat.observation_space}")

[obs, _, _, _, _] = env_flat.step(env_flat.action_space.sample())
print(f"post-step obs = {obs}")

obs = env_deep_dict.observation_space.sample()
print(f"unwrapped obs = {obs}")
new_obs = env_flat.observation(obs=obs)
print(f"wrapped obs = {new_obs}")
check_env(env_flat)

# %% Test MakeDict
print("\nTest MakeDict...")
env_cartpole = gym.make("CartPole-v1")
env_cartpole_dict = MakeDict(env=env_cartpole)
print(f"env_cartpole_dict = {type(env_cartpole_dict)}")
print(f"obs = {env_cartpole_dict.observation_space.sample()}")


# %% Test Rescale obs
print("\nTest LinScaleDictObs...")
print("Test baseline case")

rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(low=0, high=1, shape=(3, 2)),
                "b": Box(low=0, high=1, shape=(3,)),
                "c": MultiBinary(3),
                "d": Box(low=0, high=inf, dtype=int),
            }
        )
    }
)

rescale_env = LinScaleDictObs(rand_env, rescale_config={"a": 1e-4})
print(f"rescale_env = {rescale_env}")

unscaled_obs = rescale_env.observation_space.sample()
scaled_obs = rescale_env.observation(obs=unscaled_obs)
print(f"pre-scaled obs = {unscaled_obs}")
print(f"post-scaled obs = {scaled_obs}")

# Test with empty config
print("Test with default (empty) config")
rescale_env = LinScaleDictObs(rand_env)
print(f"rescale_env = {rescale_env}")

unscaled_obs = rescale_env.observation_space.sample()
scaled_obs = rescale_env.observation(obs=unscaled_obs)
print(f"pre-scaled obs = {unscaled_obs}")
print(f"post-scaled obs = {scaled_obs}")

# Check non-dict observation space
print("Check non-dict observation space")
env_rand_box = RandomEnv(
    {
        "observation_space": gym.spaces.Box(low=1, high=2, shape=(1,)),
        "action_space": gym.spaces.Box(low=1, high=2, shape=(1,)),
    }
)
try:
    rescale_env = LinScaleDictObs(env_rand_box)
except Exception as err:
    print(err)

# Check dict obs space, but non-Box entry
print("Check non-Box entry")
env_non_box = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {"state_a": gym.spaces.MultiDiscrete([2, 2, 2])}
        ),
        "action_space": gym.spaces.Box(low=1, high=2, shape=(1,)),
    }
)
try:
    rescale_env = LinScaleDictObs(env_non_box, {"state_a": 2})
except Exception as err:
    print(err)

# %% Test MinMaxScaleDict wrapper
# test with 2d box, 1d box, and multibinary
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(low=0, high=1, shape=(3, 2)),
                "b": Box(low=0, high=1, shape=(3,)),
                "c": MultiBinary(3),
                "d": Box(low=0, high=inf, dtype=int),
            }
        )
    }
)

env_minmax = MinMaxScaleDictObs(env=rand_env)

obs_nomask = rand_env.observation_space.sample()
obs_nomask["c"] = array([1, 1, 1], dtype=int)
obs_mask = env_minmax.observation(obs_nomask)
print(f"unwrapped obs  = \n{obs_nomask}")
print(f"wrapped obs = \n{obs_mask}")
assert env_minmax.observation_space.contains(obs_mask)

# %% Test SplitArrayObs
rand_env = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "a": gym.spaces.Box(low=0, high=1, shape=(2, 2)),
                "b": gym.spaces.Box(low=0, high=1, shape=(3, 2, 2)),
            }
        )
    }
)
env_split = SplitArrayObs(
    env=rand_env,
    keys=["a", "b"],
    new_keys=[["a1", "a2"], ["b1", "b2"]],
    indices_or_sections=[2],
    axes=[0, 1],
)
obs_unwrapped = rand_env.observation_space.sample()
obs_wrapped = env_split.observation(obs=obs_unwrapped)
print(
    f"""unwrapped obs shapes:
'a' = {obs_unwrapped['a'].shape}
'b' = {obs_unwrapped['b'].shape}"""
)
print(
    f"""wrapped obs shapes:
'a1'= {obs_wrapped['a1'].shape}
'a2'= {obs_wrapped['a2'].shape}
'b1'= {obs_wrapped['b1'].shape}
'b2'= {obs_wrapped['b2'].shape}
"""
)
assert env_split.observation_space.contains(obs_wrapped)


# %% Test SumArrayWrapper
print("\nTest SumArrayWrapper...")

rand_env = RandomEnv(
    {"observation_space": gym.spaces.Dict({"a": Box(0, 1, shape=(2, 3))})}
)

# Test with defaults
saw = SumArrayWrapper(rand_env, keys=["a"])
saw_uw_obs = rand_env.observation_space.sample()
saw_wr_obs = saw.observation(saw_uw_obs)
print(f"unwrapped obs 1 = {saw_uw_obs['a']}")
print(f"wrapped obs 1 = {saw_wr_obs['a']}")

# Test with specifying axis
saw = SumArrayWrapper(rand_env, keys=["a"], axis=0)
saw_uw_obs = rand_env.observation_space.sample()
saw_wr_obs = saw.observation(saw_uw_obs)
print(f"unwrapped obs 2 = {saw_uw_obs['a']}")
print(f"wrapped obs 2 = {saw_wr_obs['a']}")

# Test with MultiBinary space
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": MultiBinary([2, 3, 4])}),
    },
)
saw_rand = SumArrayWrapper(env=rand_env, keys=["a"], axis=0)
sawrand_uw_obs = rand_env.observation_space.sample()
sawrand_wr_obs = saw_rand.observation(sawrand_uw_obs)
print(f"unwrapped obs rand = \n{sawrand_uw_obs['a']}")
print(f"wrapped obs rand = \n{sawrand_wr_obs['a']}")
# %% Test Convert2dTo3dObsItems
print("\nTest Convert2dTo3dObsItems...")
env_rand = rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1, shape=[2, 3]),
                "b": Box(0, 1, shape=[6, 2]),
            }
        ),
    },
)

env_3d = Convert2dTo3dObsItems(env=env_rand, keys=["a"], diag_on_0_or_1=[1])
obs2d = env_rand.observation_space.sample()
obs3d = env_3d.observation(obs2d)
print(f"unwrapped obs 2d = \n{obs2d}")
print(f"wrapped obs 3d = \n{obs3d}")
assert env_3d.observation_space.contains(obs3d)

# %% Test DiagonalObsItems
print("\nTest DiagonalObsItems...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1, shape=[3, 3, 2]),
                "b": Box(0, 1, shape=[2, 3, 3], dtype=int),
                "c": MultiBinary(n=[3, 3, 2]),
            }
        ),
    },
)

diag_env = DiagonalObsItems(
    rand_env, keys=["a", "b", "c"], axis1=[0, 1, 0], axis2=[1, 2, 1]
)
unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = diag_env.observation(unwrapped_obs)
print(f"wrapped observation space = {diag_env.observation_space}")
print(
    f"""Shape of unwrapped obs:
    'a' = {unwrapped_obs['a'].shape}
    'b' = {unwrapped_obs['b'].shape}"""
)
print(
    f"""Shape of wrapped obs:
    'a' = {wrapped_obs['a'].shape}
    'b' = {wrapped_obs['b'].shape}"""
)
print(f"wrapped obs dtype of 'b' = {wrapped_obs['b'].dtype}")
assert diag_env.observation_space.contains(wrapped_obs)

# %% Test combo VisMap2ActionMask, ConvertCustody2ActionMask
# This test checks that the action masks generated by ConvertCustody2ActionMask
# and VisMap2ActionMask are consistent with each other. Meaning that the sensor-action
# pair from one class corresponds to the same pair from the other class.
print("\nTest combo VisMap2ActionMask and ConvertCustody2ActionMask...")

# Test env: 4 targets, 2 sensors
num_targets = 2
num_sensors = 2
rand_env = RandomEnv(
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

# Wrap with CustodyWrapper, ConvertCustody2ActionMask, VisMap2ActionMask
env_1wrap = CustodyWrapper(
    rand_env,
    key="debug_custody",
    config={"func": DebugCustody(num_targets=num_targets).update},
)
env_2wrap = ConvertCustody2ActionMask(env_1wrap, key="custody")
env_3wrap = VisMap2ActionMask(
    env_2wrap,
    vis_map_key="debug_vismap",
    rename_key="vis_action_mask",
)

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

# %% Test ConvertObsBoxToMultiBinary
print("\nTest ConvertObsBoxToMultiBinary...")
box_env = RandomEnv(
    {
        "observation_space": Dict(
            {"a": Box(low=0, high=1, shape=(2, 2), dtype=int)}
        )
    }
)
mb_env = ConvertObsBoxToMultiBinary(box_env, key="a")
obs_unwrapped = box_env.observation_space.sample()
obs_wrapped = mb_env.observation(obs_unwrapped)
print(f"unwrapped space  = {box_env.observation_space}")
print(f"unwrapped obs  = \n{obs_unwrapped}")
print(f"unwrapped dtype  = {obs_unwrapped['a'].dtype}")
print(f"wrapped space  = {mb_env.observation_space}")
print(f"wrapped obs = \n{obs_wrapped}")
print(f"wrapped dtype = {obs_wrapped['a'].dtype}")
assert mb_env.observation_space.contains(obs_wrapped)

# %% Test SqueezeObsItems
print("\nTest SqueezeObsItems...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(low=0, high=1, shape=(2, 1, 2)),
                "b": MultiBinary((2, 1, 2)),
            }
        )
    }
)
squeeze_env = SqueezeObsItems(rand_env, keys=["a", "b"])

unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = squeeze_env.observation(unwrapped_obs)
print(f"wrapped obs space = {squeeze_env.observation_space}")
print(f"unwrapped obs  = \n{unwrapped_obs}")
print(f"wrapped obs = \n{wrapped_obs}")
assert squeeze_env.observation_space.contains(wrapped_obs)

# %% Test WastedActionsMask
print("\nTest WastedActionsMask...")
print("\n  Test with null action row excluded")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"vis_map": MultiBinary((3, 2))}),
        "action_space": MultiDiscrete([4, 4]),
    }
)

nam_env = WastedActionsMask(
    rand_env, vis_map_key="vis_map", mask_key="null_mask"
)

unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = nam_env.observation(unwrapped_obs)
print(f"unwrapped obs  = \n{unwrapped_obs}")
print(f"wrapped obs = \n{wrapped_obs}")
assert nam_env.observation_space.contains(wrapped_obs)

print("\n  Test with null action row included")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"vis_map": MultiBinary((4, 2))}),
        "action_space": MultiDiscrete([4, 4]),
    }
)

nam_env = WastedActionsMask(
    rand_env, vis_map_key="vis_map", mask_key="null_mask"
)
unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = nam_env.observation(unwrapped_obs)
print(f"unwrapped obs  = \n{unwrapped_obs}")
print(f"wrapped obs = \n{wrapped_obs}")
assert nam_env.observation_space.contains(wrapped_obs)

# %% Test TransformDictObsWithNumpy
print("\nTest TransformDictObsWithNumpy...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(low=10, high=100, shape=(3,)),
                "b": MultiBinary((3, 2)),
            }
        ),
    }
)

tdo_env = TransformDictObsWithNumpy(rand_env, "mean", "a")
unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = tdo_env.observation(unwrapped_obs)
print(f"unwrapped obs  = \n{unwrapped_obs}")
print(f"wrapped obs = \n{wrapped_obs}")
assert tdo_env.observation_space.contains(wrapped_obs)

tdo_env = TransformDictObsWithNumpy(rand_env, "mean", "b", axis=1)
unwrapped_obs = rand_env.observation_space.sample()
wrapped_obs = tdo_env.observation(unwrapped_obs)
print(f"unwrapped obs  = \n{unwrapped_obs}")
print(f"wrapped obs = \n{wrapped_obs}")
assert tdo_env.observation_space.contains(wrapped_obs)

# %% MakeObsSpaceMultiBinary
print("\nTest MakeObsSpaceMultiBinary...")
rand_env = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(low=0, high=1, shape=(3,), dtype=int),
            }
        ),
    }
)
mb_env = MakeObsSpaceMultiBinary(rand_env, "a")
print(f"unwrapped obs space = {rand_env.observation_space}")
print(f"wrapped obs space = {mb_env.observation_space}")

# %% Done
print("done")
