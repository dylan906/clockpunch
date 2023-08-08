__version__ = "0.6.1"

# Third Party Imports
# from gymnasium.envs.registration import register
import gymnasium as gym

gym.register(
    id="SSAScheduler-v0",
    entry_point="punchclock.environment.env:SSAScheduler",
)
