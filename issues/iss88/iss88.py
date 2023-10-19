"""Demo MaskedLSTM on simple env."""
# https://github.com/ray-project/ray/blob/1845f1c9f2533955574a6b946e932511c1e02c48/rllib/examples/env/repeat_after_me_env.py#L6
# %% Imports
# Third Party Imports
import ray.rllib.algorithms.ppo as ppo
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gymnasium.spaces.utils import flatten, flatten_space
from numpy import array, ones
from ray import air, tune
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.models import ModelCatalog

# Punch Clock Imports
from punchclock.nets.lstm_mask import MaskedLSTM


# %% Build env
class wrappedRepeat(Env):
    def __init__(self, config=None):
        self.internal_env = RepeatAfterMeEnv()
        self.observation_space = Dict(
            {
                "observations": flatten_space(
                    self.internal_env.observation_space
                ),
                "action_mask": flatten_space(self.internal_env.action_space),
            }
        )
        self.action_space = self.internal_env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.internal_env.reset()
        new_obs = self._wrapObs(obs)
        return new_obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.internal_env.step(action)
        new_obs = self._wrapObs(obs)
        return new_obs, reward, done, trunc, info

    def _wrapObs(self, unwrapped_obs):
        wrapped_obs = {
            "observations": flatten(
                self.internal_env.observation_space, unwrapped_obs
            ),
            # "action_mask": self.observation_space.spaces[
            #     "action_mask"
            # ].sample(),
            "action_mask": ones(2, dtype=int),
        }

        return wrapped_obs


env = wrappedRepeat()
env.reset()
obs, r, _, _, _ = env.step(env.action_space.sample())
print(f"obs = {obs}")
assert env.observation_space.contains(env.observation_space.sample())
assert env.action_space.contains(env.action_space.sample())
# %% Build Tuner
ModelCatalog.register_custom_model("MaskedLSTM", MaskedLSTM)

config = (
    ppo.PPOConfig()
    .environment(wrappedRepeat)
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedLSTM",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {
                "fcnet_hiddens": [6, 6],
                "fcnet_activation": "relu",
                "lstm_state_size": 10,
            },
        }
    )
)

param_space = {
    "framework": "torch",
    "env": wrappedRepeat,
    "model": {
        # Specify our custom model from above.
        "custom_model": "MaskedLSTM",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {
            "fcnet_hiddens": [6, 6],
            "fcnet_activation": "relu",
            "lstm_state_size": 10,
        },
    },
}
tuner = tune.Tuner(
    trainable="PPO",
    param_space=param_space,
    run_config=air.RunConfig(
        stop={"training_iteration": 100},
    ),
)
results = tuner.fit()
# # %% Test policy
# policy = algo.get_policy()
# state = policy.get_initial_state()
# # action = policy.action_space.sample()
# obs, info = env.reset()
# r_cum = 0
# for _ in range(100):
#     action, state, _ = policy.compute_single_action(obs=obs, state=state)
#     obs, r, _, _, _ = env.step(action)
#     r_cum += r

# print(f"r_cum = {r_cum}")

# %% Done
print("done")
