"""Demonstration of LSTM wrapped around action mask model."""
# LSTM example: https://github.com/ray-project/ray/blob/efc23677a2dcfebb08b919ab37bdf319d50ce6b1/rllib/examples/lstm_auto_wrapping.py
# Action mask example: https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py

# %% Imports
# Third Party Imports
import gymnasium as gym
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
from gymnasium.spaces.utils import flatdim, flatten, flatten_space
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.examples.models.action_mask_model import ActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env

# Punch Clock Imports
from punchclock.nets.action_mask_model import MyActionMaskModel

torch, _ = try_import_torch()


# %% Custom Model
class MyCustomModel(TorchModelV2):
    # Copied from Ray example, see link above
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.num_outputs = int(np.product(self.obs_space.shape))
        self._last_batch_size = None

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"]
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        # Return 2x the obs (and empty states).
        # This will further be sent through an automatically provided
        # LSTM head (b/c we are setting use_lstm=True below).
        return obs * 2.0, []

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))


# %% Random Environment with action mask
# Action mask models require an env with a Dict observation space that has
# "action_mask" as an item.
env_config = {
    "observation_space": gym.spaces.Dict(
        {
            "observations": gym.spaces.Box(0, 1, shape=(10,)),
            "action_mask": gym.spaces.MultiBinary(1),
        }
    ),
    "action_space": gym.spaces.MultiDiscrete([2]),
}
num_outputs = flatdim(env_config["action_space"])

# %% Run script
if __name__ == "__main__":
    ray.init()

    # Register models-- choose one to put int "custom_model" in ppo config
    ModelCatalog.register_custom_model("custom_mask_model", MyActionMaskModel)
    ModelCatalog.register_custom_model("ray_mask_model", ActionMaskModel)
    ModelCatalog.register_custom_model("my_torch_model", MyCustomModel)

    # config, build, and train algo
    config = (
        ppo.PPOConfig()
        .environment(RandomEnv, env_config=env_config)
        # .environment("CartPole-v1")
        .framework("torch")
        .training(
            model={
                # Auto-wrap the custom(!) model with an LSTM.
                "use_lstm": True,
                # To further customize the LSTM auto-wrapper.
                "lstm_cell_size": 64,
                # Specify our custom model from above.
                "custom_model": "custom_mask_model",
                # Extra kwargs to be passed to your model's c'tor.
                "custom_model_config": {},
                "fcnet_hiddens": [10],
                # "num_outputs": num_outputs,
            }
        )
    )
    algo = config.build()
    algo.train()
    algo.stop()

    print("done")
