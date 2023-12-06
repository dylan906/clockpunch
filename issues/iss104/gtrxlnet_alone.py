# Third Party Imports
import gymnasium as gym
import torch
from numpy import array
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models.torch.attention_net import GTrXLNet
from torch import tensor

env2 = RandomEnv(
    {
        "observation_space": gym.spaces.Box(0, 1, shape=[4]),
        "action_space": gym.spaces.MultiDiscrete([2]),
    }
)
env2.reset()
# %% Ray example mode
model = GTrXLNet(
    observation_space=env2.observation_space,
    action_space=env2.action_space,
    num_outputs=2,
    model_config={"max_seq_len": 10},
    name="foo",
)

initial_state = [
    torch.zeros(
        1,
        model.view_requirements["state_in_{}".format(i)].space.shape[0],
    )
    for i in range(model.num_transformer_units)
]
[logits, state] = model.forward(
    input_dict={"obs": tensor(env2.observation_space.sample())},
    # state=model.get_initial_state(),
    state=initial_state,
    seq_lens=tensor(array([1])),
)
