"""Retrain a pretrained model."""
# Standard Library Imports
from pathlib import Path

# Third Party Imports
from ray.rllib.algorithms.algorithm import Algorithm

fpath = Path(__file__).parent
checkpoint_path = Path(
    "issues/iss92/data/training_run_PRU/PPO_CartPole-v1_843ef_00000_0_2023-10-26_17-56-29/checkpoint_000001/policies/default_policy"
)

tuner = Algorithm.restore(str(checkpoint_path))

print("done")
