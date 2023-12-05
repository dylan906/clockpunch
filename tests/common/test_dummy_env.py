"""Test dummy_env."""
# Punch Clock Imports
from punchclock.common.dummy_env import MaskRepeatAfterMe

# %% Tests
for c in ["off", "viable_random", "full_random"]:
    print(f"{c=}")
    env = MaskRepeatAfterMe(config={"mask_config": c})
    obs, info = env.reset()
    print(f"{obs=}")
    print(f"{info=}")
    obs, rew, term, trunc, info = env.step(action=env.action_space.sample())
    print(f"{obs=}")
    print(f"{rew=}")
    print(f"{term=}")
    print(f"{trunc=}")
    print(f"{info=}")
    print("\n")
