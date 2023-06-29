"""Model with action masking."""
# Based on Ray examples:
# https://github.com/ray-project/ray/blob/3e1cdeb117c945ba42df93d629f9a70189e38db9/rllib/examples/models/action_mask_model.py
# https://github.com/ray-project/ray/blob/3e1cdeb117c945ba42df93d629f9a70189e38db9/rllib/examples/action_masking.py
# %% Imports
from __future__ import annotations

# Standard Library Imports
from typing import Any

# Third Party Imports
import torch
import torch.nn as nn
from gymnasium.spaces import Dict, Space
from numpy import stack
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from torch import Tensor, tensor

# from ray.rllib.models.modelv2 import restore_original_dimensions

# %% Class


class MyActionMaskModel(TorchModelV2, nn.Module):
    """Model that handles simple discrete action masking.

    Include all regular model configuration parameters (fcnet_hiddens,
    fcnet_activation, etc.) in model_config.

    Custom model parameters are set in model_config["custom_model_config"]. The
    only custom model parameter is "no_masking" (default False).
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        """Initialize action masking model.

        Args:
            obs_space (`Space`): A gym space.
            action_space (`Space`): A gym space.
            num_outputs (`int`): Number of outputs of neural net. Should be the
                size of the flattened action space.
            model_config (`dict`): Model configuration. Required inputs are:
                {
                    "fcnet_hiddens" (`list[int]`): Fully connected hidden layers.
                }
            name (`str`): Name of model.

        To disable action masking, set:
            model_config["custom_model_config"]["no_masking"] = True.
        """
        # Check that the observation space is a dict that contains "action_mask"
        # and "observations" as keys.
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces

        # Boilerplate Torch stuff.
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        # Build feed-forward layers
        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        last_layer_size = model_config["fcnet_hiddens"][-1]
        self.action_head = nn.Linear(last_layer_size, num_outputs)
        self.value_head = nn.Linear(last_layer_size, 1)

        # disable action masking --> will likely lead to invalid actions
        custom_config = model_config.get("custom_model_config", {})
        self.no_masking = False
        if "no_masking" in custom_config:
            self.no_masking = custom_config["no_masking"]

    def forward(
        self,
        input_dict: dict[dict],
        state: Any,
        seq_lens: Any,
    ) -> [Tensor, Any]:
        """Forward propagate observations through the model.

        Takes a `dict` as an argument with the only key being "obs", which is either
        a sample from the observation space or a list of samples from the observation
        space.

        Can input either a single observation or multiple observations. If using
        a single observation, the input is a dict[dict[dict]]]. If using
        multiple observations, the input is a dict[dict[list_of_dicts]].

        Args:
            input_dict (`dict`[`dict`]):
                {
                    "obs": {
                        "action_mask": `Tensor`,
                        "observations": `Tensor`,
                    }
                }
                or
                {
                    "obs": list[
                        {
                        "action_mask": `Tensor`,
                        "observations": `Tensor`
                        },
                        ...]
                }
            state (`Any`): _description_
            seq_lens (`Any`): _description_

        Returns:
            logits (`Tensor`): Logits in shape of (num_outputs, ).
            state (`Any`): _description_
        """
        # Potential solution to sample batch errors.
        # https://discuss.ray.io/t/dict-observation-space-flattened/541
        # original_obs = restore_original_dimensions(
        #     obs=input_dict["obs"],
        #     obs_space=self.obs_space,
        #     tensorlib="torch",
        # )

        # Extract the action mask and observations from the input dict and convert
        # to tensor, if necessary. Stack action masks and observations into larger
        # tensor if multiple obs are passed in. The action mask and observation
        # are different sizes depending on if multiple or single observations are
        # passed in. Convert tensors to floats if not already to input to torch
        # Linear layers
        # (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear).
        if type(input_dict["obs"]) is list:
            # For multiple observations
            # action_mask is a [num_observations, len_mask] tensor
            # observation is a [num_observations, len_obs] tensor
            array_of_masks = stack(
                [a["action_mask"] for a in input_dict["obs"]], axis=0
            )
            action_mask = tensor(array_of_masks)
            array_of_obs = stack(
                [a["observations"] for a in input_dict["obs"]], axis=0
            )
            observation = tensor(array_of_obs).float()
        else:
            action_mask = input_dict["obs"]["action_mask"]
            observation = input_dict["obs"]["observations"].float()

        # print(f"model: mask = {action_mask}")

        # Compute the unmasked logits.
        # Using self.internal_model() is the more "appropriate" way to do the
        # the forward pass, but it doesn't seem to work. Instead, just use
        # _hidden_layers.forward().
        # appropriate_features = self.internal_model({"obs": observation})
        self.internal_model._features = (
            self.internal_model._hidden_layers.forward(observation)
        )
        # print(
        #     f"internal_model._features.size() = {self.internal_model._features.size()}"
        # )
        logits = self.action_head(self.internal_model._features)

        # If action masking is disabled, skip masking and return unmasked logits.
        # Otherwise, step into masking block.
        if self.no_masking is False:
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            # print(f"logits.size() = {logits.size()}")
            # print(f"inf_mask.size() = {inf_mask.size()}")
            masked_logits = logits + inf_mask
            logits = masked_logits

        return logits, state

    def value_function(self) -> Tensor:
        """Get current value of value function.

        Returns:
            `Tensor[torch.float32]`: Value function value.
        """
        # get features and squeeze extra dimensions out.
        y = self.value_head(self.internal_model._features)
        y = y.squeeze(-1)
        return y
