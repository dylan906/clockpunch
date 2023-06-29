"""Basic Neural Net model for use with SSAScheduler."""
# %% Useful links
# https://docs.ray.io/en/latest/rllib/package_ref/models.html
# %% Imports
from __future__ import annotations

# Third Party Imports
import gymnasium as gym
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict


# %% Custom Neural Net
class MultiDiscreteFeedForward(TorchModelV2, nn.Module):
    """Feed forward network for a MultiDiscrete action space.

    Attributes:
        obs_dim: Number of dimensions in observation space.
        act_dim: Number of dimensions in action space.
        num_actions: Number of outputs of the network.
        num_action_vals: Number of values that can be taken by actions.

    Notes:
        - Hidden layers and nodes are static 200/100/50.
        - Activation function is ReLU.
        - Output is (1, num_actions) tensor.
        - Output entries take on values (0 to (num_action_vals-1)).
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space.MultiDiscrete,
        num_outputs: int = None,
        model_config: ModelConfigDict = {},
        name: str = "my_model",
    ):
        """Initialize.

        Args:
            obs_space (`gym.spaces.Space`): Observation space.
            action_space (`gym.spaces.Space.MultiDiscrete`): Action space. All entries must
                be identical.
            num_outputs (`int`, optional): Here for Ray compatibility only. Do not override
                default. Defaults to None.
            model_config (`ModelConfigDict`, optional): Required for Ray register. Defaults
                to {}.
            name (`str`, optional): Name of model. Required for Ray register. Defaults to
                "my_model".

        Example:
            obs_space = gym.spaces.Box(-1, 1, shape=[5])
            act_space = gym.spaces.MultiDiscrete([3, 3, 3])

            my_model = MultiDiscreteFeedForward(
                obs_space=obs_space,
                action_space=act_space)
        """
        # Attributes
        self.obs_dim = gym.spaces.utils.flatdim(obs_space)
        self.act_dim = gym.spaces.utils.flatdim(action_space)
        # get number of possible values each entry action_space can take
        # NOTE: Assumes all actions can take identical range of values
        self.num_action_vals = action_space.nvec[0]
        # number of actions taken (number of outputs of model)
        self.num_actions = action_space.shape[0]

        # initialize ray version of Torch model
        TorchModelV2.__init__(
            self, obs_space, action_space, self.num_actions, model_config, name
        )

        # initialize Torch NN
        nn.Module.__init__(self)

        print(f"act_dim = {self.act_dim}")

        # ---Layers---
        # all layers fully connected
        layers = [
            nn.Linear(self.obs_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, self.act_dim),
        ]
        self._hidden_layers = nn.Sequential(*layers)

        # Initialize MaxPool1d layer for output processing (kernel = stride)
        self.maxpool = nn.MaxPool1d(
            self.num_action_vals,
            return_indices=True,
        )

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType] = None,
        seq_lens: TensorType = None,
    ) -> (TensorType, list[TensorType]):
        """Forward propagate model.

        Args:
            input_dict (`Dict[str, TensorType]`): _description_
            state (`List[TensorType]`, optional): Here for Ray compatibility. Defaults to None
            seq_lens (`TensorType`, optional): Here for Ray compatibility. Defaults to None.

        Returns:
            `Tuple`: Model output tensor and list of new RNN state(s) if any.
        """

        # extract observations from input_dict
        # convert to float likely unnecessary
        obs = input_dict["obs"].float()

        # propagate observations through hidden layers
        x = self._hidden_layers.forward(obs)
        print(f"x = {x}")
        print(f"x.size() = {x.size()}")
        # Convert outputs (from hidden layers) from 1 -> 2-dim
        x = torch.reshape(x, (1, -1))
        print(f"x after reshape = {x}")
        # Get indices of maxPool
        self._features = self.maxPoolIndex1d(x)

        return self._features, []

    def maxPoolIndex1d(self, input: TensorType) -> TensorType:
        """Get the indices of max pooled values from a 1d tensor.

        Args:
            input (`TensorType`): A tensor.

        Returns:
            `TensorType`: Indices of maximum inputs in pools of `self.num_action_vals`.
        """
        [_, indices] = self.maxpool(input)
        # Modulo-- get relative indices of actions
        indices = indices % self.num_action_vals
        return indices
