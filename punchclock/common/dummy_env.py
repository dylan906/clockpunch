"""Action Mask Repeat After Me Env to use in testing."""
# %% Imports
# Third Party Imports
from gymnasium import Env
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten, flatten_space
from numpy import ones
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv


class MaskRepeatAfterMe(Env):
    """RepeatAfterMeEnv with action masking.

    There are three options for mask_config:
        "viable_random": The action mask is a random sample from the action space,
            with the first action always available.
        "full_random": The action mask is a random sample from the action space.
        "off": All actions are available (the action mask is an array of ones).
    """

    def __init__(self, config=None):
        """Instantiate MaskRepeatAfterMe."""
        self.internal_env = RepeatAfterMeEnv()
        self.observation_space = Dict(
            {
                "observations": flatten_space(self.internal_env.observation_space),
                "action_mask": flatten_space(self.internal_env.action_space),
            }
        )
        self.action_space = self.internal_env.action_space
        if config is None:
            config = {}
        self.mask_config = config.get("mask_config", "viable_random")

    def reset(self, *, seed=None, options=None):
        """Reset env."""
        obs, info = self.internal_env.reset()
        new_obs = self._wrapObs(obs)
        self.last_obs = new_obs
        return new_obs, info

    def step(self, action):
        """Step env."""
        trunc = self._checkMaskViolation(action)
        obs, reward, done, _, info = self.internal_env.step(action)
        new_obs = self._wrapObs(obs)
        self.last_obs = new_obs
        return new_obs, reward, done, trunc, info

    def _wrapObs(self, unwrapped_obs):
        if self.mask_config in ["viable_random"]:
            mask = self.observation_space.spaces["action_mask"].sample()
            mask[0] = 1
        elif self.mask_config == "full_random":
            mask = self.observation_space.spaces["action_mask"].sample()
        elif self.mask_config == "off":
            mask = ones(self.observation_space.spaces["action_mask"].shape, dtype=int)

        wrapped_obs = {
            "observations": flatten(self.internal_env.observation_space, unwrapped_obs),
            "action_mask": mask,
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
