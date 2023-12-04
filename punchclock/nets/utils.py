"""Nets utils."""
# Standard Library Imports
from warnings import warn

# Third Party Imports
import torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from torch import all as tensorall


def maskLogits(logits: TensorType, mask: TensorType):
    """Apply mask over raw logits."""
    # Resolve edge case where Policy.build() can pass in mask values <0 and
    # non-integers. Clamp values < 0  to 0, and values > 0 to 1.
    mask_binary = torch.clamp(mask, min=0, max=1)
    mask_binary[mask_binary > 0] = 1

    # check for binary action mask so error doesn't happen in action distribution
    # creation.
    assert all([i in [0, 1] for i in mask_binary.detach().numpy().flatten()])
    if tensorall(mask_binary == 0):
        warn("All actions masked")

    # Mask logits
    inf_mask = torch.clamp(torch.log(mask_binary), min=FLOAT_MIN)
    masked_logits = logits + inf_mask
    return masked_logits
