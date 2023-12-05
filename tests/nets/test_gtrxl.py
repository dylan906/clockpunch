"""Test gtrxl.py."""
# %% Imports
# Third Party Imports
import gymnasium as gym
import ray.rllib.algorithms.ppo as ppo
import torch
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import GTrXLNet
from torch import reshape, tensor, transpose

# Punch Clock Imports
from punchclock.nets.gtrxl import MaskedGTrXL
from punchclock.nets.utils import preprocessDictObs

# # %% Ray example model
# env2 = RandomEnv(
#     {
#         "observation_space": gym.spaces.Box(0, 1, shape=[4]),
#         "action_space": gym.spaces.Box(0, 1, dtype=float),
#     }
# )
# env2.reset()
# model = GTrXLNet(
#     observation_space=env2.observation_space,
#     action_space=env2.action_space,
#     num_outputs=1,
#     attention_dim=1,
#     num_heads=1,
#     model_config={"max_seq_len": 10},
#     name="derp",
# )

# # obs = env2.observation_space.sample()
# # obs = {"obs": tensor(obs)}
# input_dict = {"obs": torch.rand(env2.observation_space.shape)}
# [logits, state] = model.forward(
#     input_dict=input_dict,
#     state=model.get_initial_state(),
#     seq_lens=tensor(array([1])),
# )

# %% Make test Env
print("\nBuild test env...")
obs_space = gym.spaces.Dict(
    {
        "observations": gym.spaces.Box(0, 1, shape=[4]),
        "action_mask": gym.spaces.Box(0, 1, shape=[2], dtype=int),
    }
)
action_space = gym.spaces.MultiDiscrete([2])
env_config = {
    "observation_space": obs_space,
    "action_space": action_space,
}
env = RandomEnv(env_config)
print(f"observation_space = {env.observation_space}")
print(f"action_space = {env.action_space}")
env.reset()


# # %% Model
model = MaskedGTrXL(
    observation_space=env.observation_space,
    action_space=env.action_space,
    num_outputs=2,
    # memory_training=4,
    attention_dim=1,
    model_config={"max_seq_len": 10},
    name="derp",
)

print(f"{model=}")

# %% Test model (basic)
print("\nTest model (basic)...")
# obs = env.observation_space.sample()
# # override action mask to make sure we don't have all same values
# obs["action_mask"][0] = 0
# obs["action_mask"][1] = 1
# print(f"obs (raw) = {obs}")
# obs = preprocessDictObs(obs)
# obs["obs"]["observations"] = reshape(obs["obs"]["observations"], (1, -1))
# print(f"obs (preprocessed) = {obs}")
input_dict = {
    "obs": {
        "observations": torch.rand(obs_space["observations"].shape),
        "action_mask": torch.ones(obs_space["action_mask"].shape),
    }
}

seq_lens = tensor(array([1]))
init_state = model.get_initial_state()
# [logits, state] = model.forward(
#     input_dict=input_dict, state=init_state, seq_lens=seq_lens
# )
# print(f"logits = {logits}")
# print(f"state = {state}")

# %% Test as algo
ModelCatalog.register_custom_model("MaskedGTrXL", MaskedGTrXL)

config = (
    ppo.PPOConfig()
    .environment(RandomEnv, env_config=env_config)
    .framework("torch")
    .training(
        model={
            # Specify our custom model from above.
            "custom_model": "MaskedGTrXL",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {"max_seq_len": 10},
        }
    )
)

algo = config.build()
policy = algo.get_policy()
print(f"policy = {policy}")

state = policy.get_initial_state()
action, state_out, _ = policy.compute_single_action(
    obs=policy.observation_space.sample(),
    state=state,
)
print(f"{action=}")
print(f"{state_out=}")

# %% done
print("done")
