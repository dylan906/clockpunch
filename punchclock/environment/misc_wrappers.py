"""Misc wrappers."""
# %% Imports
# Third Party Imports
from gymnasium import Env, Wrapper


# %% Identity Wrapper
class IdentityWrapper(Wrapper):
    """Wrapper does not modify environment, used for construction."""

    def __init__(self, env: Env):
        super().__init__(env)
        return
