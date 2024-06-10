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
from ray.rllib.utils.typing import TensorType
from torch import Tensor, tensor

# %% Class


class MyActionMaskModel(TorchModelV2, nn.Module):
    """Model that handles simple discrete action masking."""

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: dict = None,
        name: str = None,
        **custom_model_kwargs,
    ):
        """Initialize action masking model.

        Args:
            obs_space (Space): A gym space.
            action_space (Space): A gym space.
            num_outputs (int): Number of outputs of neural net. Should be the
                size of the flattened action space.
            model_config (dict, optional): Model configuration for TorchFC.
                Defaults to None.
            name (str, optional): Name of model. Defaults to "MyActionMaskModel".
            custom_model_kwargs (dict): Custom model configuration. Required.

        Expected items in custom_model_kwargs:
            fcnet_hiddens (list[int]): Number and size of FC layers.
            fcnet_activation (str): Activation function for FC layers. See Ray
                SlimFC documentation for recognized args.
            no_masking (bool, optional): Disables masking if True. Defaults to
                False.

        """
        # Check that the observation space is a dict that contains "action_mask"
        # and "observations" as keys.
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, Dict)
        assert "action_mask" in orig_space.spaces
        assert "observations" in orig_space.spaces
        assert len(orig_space.spaces) == 2
        assert len(orig_space["action_mask"].shape) == 1
        assert (
            orig_space["action_mask"].shape[0] == num_outputs
        ), f"""
        orig_space['action_mask'].shape[0] = {orig_space['action_mask'].shape[0]}\n
        num_outputs = {num_outputs}
        """
        assert "fcnet_hiddens" in custom_model_kwargs
        assert "fcnet_activation" in custom_model_kwargs
        assert isinstance(custom_model_kwargs["fcnet_hiddens"], list)
        assert all([isinstance(i, int) for i in custom_model_kwargs["fcnet_hiddens"]])

        # Defaults
        if model_config is None:
            model_config = {}
        if name is None:
            name = "MyActionMaskModel"

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = custom_model_kwargs.get("no_masking", False)

        # Inheritance
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Build feed-forward layers
        model_config["fcnet_hiddens"] = custom_model_kwargs["fcnet_hiddens"]
        model_config["fcnet_activation"] = custom_model_kwargs["fcnet_activation"]
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

        print(f"MyActionMaskModel built: \n{self}")

    def forward(
        self,
        input_dict: dict[dict],
        state: Any,
        seq_lens: Any,
    ) -> [Tensor, Any]:
        """Forward propagate observations through the model.

        Takes a dict as an argument with the only key being "obs", which is either
        a sample from the observation space or a list of samples from the observation
        space.

        Can input either a single observation or multiple observations. If using
        a single observation, the input is a dict[dict[dict]]]. If using
        multiple observations, the input is a dict[dict[list_of_dicts]].

        Args:
            input_dict (dict[dict]):
                {
                    "obs": {
                        "action_mask": Tensor,
                        "observations": Tensor,
                    }
                }
                or
                {
                    "obs": list[
                        {
                        "action_mask": Tensor,
                        "observations": Tensor
                        },
                        ...]
                }
            state (Any): _description_
            seq_lens (Any): _description_

        Returns:
            logits (Tensor): Logits in shape of (num_outputs, ).
            state (Any): _description_
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
            array_of_obs = stack([a["observations"] for a in input_dict["obs"]], axis=0)
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
        self.internal_model._features = self.internal_model._hidden_layers.forward(
            observation
        )
        # print(
        #     f"internal_model._features.size() = {self.internal_model._features.size()}"
        # )
        logits = self.action_head(self.internal_model._features)

        # Apply mask
        logits = self.maskLogits(logits, action_mask)

        return logits, state

    def value_function(self) -> Tensor:
        """Get current value of value function.

        Returns:
            Tensor[torch.float32]: Value function value.
        """
        # get features and squeeze extra dimensions out.
        y = self.value_head(self.internal_model._features)
        y = y.squeeze(-1)
        return y

    def maskLogits(self, logits: TensorType, mask: TensorType):
        """Apply mask over raw logits."""
        assert all([i in [0, 1] for i in mask.detach().numpy().flatten()])

        # If action masking is disabled, skip masking and return unmasked logits.
        # Otherwise, step into masking block.
        if self.no_masking is False:
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(mask), min=FLOAT_MIN)

            # print(f"logits.size() = {logits.size()}")
            # print(f"inf_mask.size() = {inf_mask.size()}")
            masked_logits = logits + inf_mask

            return masked_logits

        else:
            return logits
