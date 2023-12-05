"""GTrXL with action masking.

Ray implementation: https://github.com/ray-project/ray/blob/master/rllib/models/torch/attention_net.py#L261
"""
# Standard Library Imports
from typing import Dict, Optional, Union

# Third Party Imports
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Space
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.attention_net import GTrXLNet
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules import (
    GRUGate,
    RelativeMultiHeadAttention,
    SkipConnection,
)
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.torch_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import List, ModelConfigDict, TensorType
from ray.util import log_once

torch, nn = try_import_torch()
# Punch Clock Imports
from punchclock.nets.utils import maskLogits


# %% MaskedGTrXL
class MaskedGTrXL(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        *,
        num_transformer_units: int = 1,
        attention_dim: int = 64,
        num_heads: int = 2,
        memory_inference: int = 50,
        memory_training: int = 50,
        head_dim: int = 32,
        position_wise_mlp_dim: int = 32,
        init_gru_gate_bias: float = 2.0,
    ):
        # Convert space to proper gymnasium space if handed is as a different
        # type. Happens during training.
        orig_space = getattr(observation_space, "original_space", observation_space)
        assert isinstance(orig_space, Dict)
        assert "observations" in orig_space.spaces
        assert "action_mask" in orig_space.spaces
        # Check that required items are in model_config
        assert "max_seq_len" in model_config

        super().__init__(
            observation_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)

        self.gtrxl = GTrXLNet(
            observation_space=orig_space["observations"],
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            num_transformer_units=num_transformer_units,
            attention_dim=attention_dim,
            num_heads=num_heads,
            memory_inference=memory_inference,
            memory_training=memory_training,
            head_dim=head_dim,
            position_wise_mlp_dim=position_wise_mlp_dim,
            init_gru_gate_bias=init_gru_gate_bias,
        )

        self._value_out = None

    def forward(
        self, input_dict, state: List[TensorType], seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"]["observations"]
        action_mask = input_dict["obs"]["action_mask"]

        logits, state = self.gtrxl.forward(
            input_dict={"obs": obs},
            state=state,
            seq_lens=seq_lens,
        )
        masked_logits = maskLogits(logits=logits, mask=action_mask)
        self._value_out = self.gtrxl._value_out

        return masked_logits, state

    @override(ModelV2)
    def get_initial_state(self):
        return [
            torch.zeros(
                (1, self.gtrxl.attention_dim)
                # self.gtrxl.view_requirements["state_in_{}".format(i)].space.shape
                # (self.gtrxl.attention_dim, self.gtrxl.memory_training),
            )
            for i in range(self.gtrxl.num_transformer_units)
        ]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert (
            self._value_out is not None
        ), "Must call forward first AND must have value branch!"
        # return torch.reshape(self._value_out, [-1])
        return self.gtrxl.value_function()
