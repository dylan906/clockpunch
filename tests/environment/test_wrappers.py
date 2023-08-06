"""Tests for wrappers.py."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from copy import deepcopy

# Third Party Imports
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.wrappers.filter_observation import FilterObservation
from numpy import array, diag, float32, inf, int64, sum
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils import check_env

# Punch Clock Imports
from punchclock.common.math import getCircOrbitVel
from punchclock.common.transforms import ecef2eci, lla2ecef
from punchclock.environment.env import SSAScheduler
from punchclock.environment.env_parameters import SSASchedulerParams
from punchclock.environment.wrappers import (
    ActionMask,
    Convert2dTo3dObsItems,
    ConvertCustody2ActionMask,
    CopyObsItem,
    CustodyWrapper,
    FlatDict,
    FloatObs,
    LinScaleDictObs,
    MakeDict,
    MinMaxScaleDictObs,
    MultiplyObsItems,
    NestObsItems,
    SelectiveDictObsWrapper,
    SplitArrayObs,
    SumArrayWrapper,
    VisMap2ActionMask,
)

# %% Build environment
# satellite initial conditions-- uniform distribution
ref_vel = getCircOrbitVel(8000)
sat_ICs_kernel = array(
    [
        [8000, 8000],
        [0, 0],
        [0, 0],
        [0, 0],
        [ref_vel, ref_vel],
        [0, 0],
    ],
)
# sensor initial conditions-- 2 equatorial terrestrial sensors spaces 90deg
sens_a = ecef2eci(lla2ecef([0, 0, 0]), 0)
sens_b = ecef2eci(lla2ecef([3.14 / 2, 0, 0]), 0)
sensor_ICs = array([sens_a, sens_b])

# agent parameters
agent_params = {
    "num_sensors": 2,
    "num_targets": 4,
    "sensor_starting_num": 1000,
    "target_starting_num": 5000,
    "sensor_dynamics": "terrestrial",
    "target_dynamics": "satellite",
    "sensor_dist": None,
    "target_dist": "uniform",
    "sensor_dist_frame": None,
    "target_dist_frame": "ECI",
    "sensor_dist_params": None,
    "target_dist_params": sat_ICs_kernel,
    "fixed_sensors": sensor_ICs,
    "fixed_targets": None,
    "init_num_tasked": None,
    "init_last_time_tasked": None,
}
# target_filter parameters
diag_matrix = diag([1, 1, 1, 0.01, 0.01, 0.01])
filter_params = {
    "Q": 0.001 * diag_matrix,
    "R": 0.01 * diag_matrix,
    "p_init": 1 * diag_matrix,
}

reward_params = {
    "reward_func": "Threshold",
    "obs_or_info": "obs",
    "metric": "obs_staleness",
    "preprocessors": ["max"],
    "metric_value": 5000,
}

# environment parameters
env_params = {
    "time_step": 100,
    "horizon": 10,
    "agent_params": agent_params,
    "filter_params": filter_params,
    "reward_params": reward_params,
}

env_builder = SSASchedulerParams(**env_params)
env = SSAScheduler(env_builder)
env.reset()
act_sample = env.action_space.sample()

# %% Test FloatObs
print("\nTest FloatObs...")
# Copy original environment and add a dict to obs space to test nested-dict handling
copy_env = deepcopy(env)
copy_env.observation_space["a_dict"] = gym.spaces.Dict(
    {
        "param_int": gym.spaces.Box(0, 1, shape=(2, 2), dtype=int64),
        "param_float": gym.spaces.Box(0, 1, shape=(2, 2), dtype=float32),
    }
)
# Test wrapper instantiation
env_float = FloatObs(copy_env)
# env should have ints
# env_float.unwrapped should have ints
# env_float should have floats
print(f"original env obs space = {copy_env.observation_space}")
print(f"unwrapped obs space = {env_float.unwrapped.observation_space}")
print(f"wrapped obs space ={env_float.observation_space}")

# Test .observation()
obs = copy_env.observation_space.sample()
obs_float = env_float.observation(obs)
print(f"original obs = {obs}")
print(f"wrapped observation = {obs_float}")
print(
    f"wrapped obs in wrapped obs space? "
    f"{env_float.observation_space.contains(obs_float)}"
)
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

# %% Test ActionMask wrapper
print("\nTest ActionMask...")
env.reset()
env_masked = ActionMask(env)

print(f"observation_space = {env_masked.observation_space}")
obs = env_masked.observation_space.sample()
print(f"obs.keys() = {obs.keys()}")
print(f"obs['action_mask'].shape = {obs['action_mask'].shape}")
print(f"type(obs['observations']) = {type(obs['observations'])}")
print(f"'observations' length = {len(obs['observations'])}")
# Check to make sure obs is contained in observation_space
print(f"obs in observation_space: {env_masked.observation_space.contains(obs)}")
act = env_masked.action_space.sample()
print(f"act in action_space: {env_masked.action_space.contains(act)}")

[obs, _, _, _, _] = env_masked.step(act)

# check env conforms to GYM API
check_env(env_masked)

# test with action mask turned off
env_masked_off = ActionMask(env=env, action_mask_on=False)
[obs, _, _, _, _] = env_masked_off.step(act_sample)
print("Now with action mask off...")
print(f"obs['action_mask'] = {obs['action_mask']}")
print(
    f"obs in observation_space: {env_masked_off.observation_space.contains(obs)}"
)
# %% Test CopyObsItem
print("\nTest CopyObsItem...")
env_premask = RandomEnv(
    {
        "observation_space": Dict(
            {"a": MultiBinary(2), "b": Box(0, 1, shape=(2, 3))}
        )
    }
)
env_postmask = CopyObsItem(env=env_premask, key="a")
obs_mask = env_postmask.observation(env_premask.observation_space.sample())
print(f"pre-mask obs space = {env_premask.observation_space}")
print(f"post-mask obs space = {env_postmask.observation_space}")
assert env_postmask.observation_space.contains(obs_mask)

# %% Test VisMap2ActionMask
print("\nTest VisMap2ActionMask...")
env_premask = RandomEnv(
    {
        "observation_space": Dict(
            {
                "a": Box(0, 1, shape=(2, 3)),
                "b": Box(0, 1, shape=(1, 1)),
            }
        ),
        "action_space": MultiDiscrete([3, 3, 3]),
    }
)
env_postmask = VisMap2ActionMask(
    env_premask, vis_map_key="a", renamed_key="a_mask"
)
unmasked_obs = env_premask.observation_space.sample()
obs_mask = env_postmask.observation(unmasked_obs)
print(f"pre-mask obs space = {env_premask.observation_space}")
print(f"post-mask obs space = {env_postmask.observation_space}")
assert env_postmask.observation_space.contains(obs_mask)

# %% Test MultiplyObsItems
print("\nTest MultiplyObsItems...")
env_randmask = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "a1": MultiBinary(4),
                "a2": Box(0, 1, shape=(4,), dtype=int),
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

[obs, _, _, _, _] = env_flat.step(act_sample)
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
rescale_env = LinScaleDictObs(env=env, rescale_config={"est_cov": 1e-4})
print(f"rescale_env = {rescale_env}")

unscaled_obs = rescale_env.observation_space.sample()
scaled_obs = rescale_env.observation(obs=unscaled_obs)
print(f"pre-scaled obs = {unscaled_obs}")
print(f"post-scaled obs = {scaled_obs}")

# Test with empty config
print("Test with default (empty) config")
rescale_env = LinScaleDictObs(env=env)
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
env_minmax = MinMaxScaleDictObs(env=env)
obs = env_minmax.observation(obs=env.observation_space.sample())

# Test with 1d/non-Box obs
rand_env = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {
                "a": gym.spaces.MultiBinary([3]),
            },
        ),
        "action_space": gym.spaces.Box(0, 1),
    }
)
rand_minmax_env = MinMaxScaleDictObs(rand_env)
new_obs = rand_minmax_env.observation(rand_env.observation_space.sample())

# %% Test SplitArrayObs
env_split = SplitArrayObs(
    env=env,
    keys=["est_cov"],
    new_keys=[["est_cov_pos", "est_cov_vel"]],
    indices_or_sections=[2],
)
obs_presplit = env.observation_space.sample()
obs_split = env_split.observation(obs=obs_presplit)
print(f"pre-split obs['est_cov'] = \n{obs_presplit['est_cov']}")
print(f"post-split obs['est_cov_pos'] = \n{obs_split['est_cov_pos']}")
print(f"post-split obs['est_cov_vel'] = \n{obs_split['est_cov_vel']}")

# %% Test SelectiveDictObsWrapper
print("\nTest SelectiveDictObsWrapper...")


def testFunc(x):
    """Test function."""
    return sum(x).reshape((1,))


new_obs_space = deepcopy(env.observation_space)
new_obs_space["est_cov"] = gym.spaces.Box(-inf, inf)
sdow = SelectiveDictObsWrapper(
    env=env,
    funcs=[testFunc],
    keys=["est_cov"],
    new_obs_space=new_obs_space,
)
print(f"unwrapped obs space = {env.observation_space['est_cov']}")
print(f"wrapped obs space = {sdow.observation_space['est_cov']}")

# %% Test CustodyWrapper
print("\nTest CustodyWrapper...")
env_custody = RandomEnv(
    {
        "observation_space": gym.spaces.Dict(
            {"a": gym.spaces.Box(0, 1, shape=(3, 6, 6))}
        ),
        "action_space": gym.spaces.MultiDiscrete([4, 4]),
    }
)
cw = CustodyWrapper(
    env_custody,
    key="a",
    num_targets=3,
    config={
        "func": "tr_cov",
        "threshold": 1,
    },
)

print(f"unwrapped obs space = \n{env_custody.observation_space.keys()}")
print(f"wrapped obs space = \n{cw.observation_space.keys()}")
obs = env_custody.observation_space.sample()
# Make covariance all positive to be ensure diagonals are properly conditioned.
# obs["est_cov"] = abs(obs["est_cov"])
print(f"wrapped obs = {cw.observation(obs)['custody']}")

# %% Test SumArrayWrapper
print("\nTest SumArrayWrapper...")
saw = SumArrayWrapper(env=env, keys=["est_cov"])
saw_uw_obs = env.observation_space.sample()
saw_wr_obs = saw.observation(saw_uw_obs)
print(f"unwrapped obs 1 = {saw_uw_obs['est_cov']}")
print(f"wrapped obs 1 = {saw_wr_obs['est_cov']}")

saw = SumArrayWrapper(env=env, keys=["est_cov"], axis=0)
saw_uw_obs = env.observation_space.sample()
saw_wr_obs = saw.observation(saw_uw_obs)
print(f"unwrapped obs 2 = {saw_uw_obs['est_cov']}")
print(f"wrapped obs 2 = {saw_wr_obs['est_cov']}")

# Test with 3d observation space
rand_env = RandomEnv(
    {
        "observation_space": Dict({"a": Box(0, 1, shape=[2, 3, 4])}),
        "action_space": Discrete(1),
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

# %% Test ConvertCustody2ActionMask
print("\nTest ConvertCustody2ActionMask...")
rand_env = RandomEnv(
    {
        "observation_space": Dict({"custody": MultiBinary(3)}),
    },
)

env_custody2am = ConvertCustody2ActionMask(
    rand_env, key="custody", num_sensors=2
)

# %% Test multiple wrappers
print("\nTest multiple wrappers...")
env_multi_wrapped = FilterObservation(
    env=env, filter_keys=["est_cov", "vis_map_est"]
)
env_multi_wrapped = SplitArrayObs(
    env=env_multi_wrapped,
    keys=["est_cov"],
    new_keys=[["est_cov_pos", "est_cov_vel"]],
    indices_or_sections=[2],
)
env_multi_wrapped = FilterObservation(
    env=env_multi_wrapped, filter_keys=["vis_map_est", "est_cov_pos"]
)
env_multi_wrapped = ActionMask(env=env_multi_wrapped)
env_multi_wrapped = FloatObs(env=env_multi_wrapped)
env_multi_wrapped = FlatDict(env=env_multi_wrapped)
env_multi_wrapped.reset()
# check_env(env_multi_wrapped)

[obs, _, _, _, _] = env_multi_wrapped.step(
    env_multi_wrapped.action_space.sample()
)

print(f"observation_space = {env_multi_wrapped.observation_space}")
print(f"obs = {obs}")
# Check to make sure obs is contained in observation_space
print(
    f"obs in observation_space: {env_multi_wrapped.observation_space.contains(obs)}"
)


# %% Done
print("done")
