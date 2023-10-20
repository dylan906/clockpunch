"""Action Mask Repeat After Me Env."""
# %% Imports
# Third Party Imports
from gymnasium import Env
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten, flatten_space
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv


class MaskRepeatAfterMe(Env):
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
        self.last_obs = new_obs
        return new_obs, info

    def step(self, action):
        trunc = self._checkMaskViolation(action)
        obs, reward, done, _, info = self.internal_env.step(action)
        new_obs = self._wrapObs(obs)
        self.last_obs = new_obs
        return new_obs, reward, done, trunc, info

    def _wrapObs(self, unwrapped_obs):
        wrapped_obs = {
            "observations": flatten(
                self.internal_env.observation_space, unwrapped_obs
            ),
            "action_mask": self.observation_space.spaces[
                "action_mask"
            ].sample(),
            # "action_mask": ones(2, dtype=int),
        }
        return wrapped_obs

    def _checkMaskViolation(self, action):
        flat_action = flatten(self.action_space, action)
        diff = self.last_obs["action_mask"] - flat_action
        if any([i < 0 for i in diff]):
            truncate = True
            print("mask violation")
        else:
            truncate = False

        return truncate
