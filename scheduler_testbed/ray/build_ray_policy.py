"""Build Ray Policies module."""
# %% Imports
from __future__ import annotations

# Third Party Imports
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

# Punch Clock Imports
from scheduler_testbed.nets.action_mask_model import MyActionMaskModel


# %% Functions
def buildRayActionMaskPolicy(
    checkpoint_path: str,
) -> dict | Policy:
    """Build Ray policy using ActionMask model from a checkpoint path.

    Args:
        checkpoint_path (`str`): Checkpoint path.

    Returns:
        `Policy`: A Ray policy.

    - Registers MyActionMaskModel.
    - checkpoint_path should end in the directory of the json file; do not include
        the file itself.
    """
    print("register model")
    # Register model (assumes MyActionMaskModel is used in policies)
    ModelCatalog.register_custom_model("action_mask_model", MyActionMaskModel)
    print("build policy from checkpoint")
    print(f"checkpoint path = {checkpoint_path}")
    policy = Policy.from_checkpoint(checkpoint_path)
    print("...policy built")
    return policy
