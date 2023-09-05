"""Misc wrappers."""
# %% Imports

# Third Party Imports
from gymnasium import Env, Wrapper


# %% Identity Wrapper
class IdentityWrapper(Wrapper):
    """Wrapper does not modify environment, used for construction."""

    # NOTE: SimRunner is hugely dependent on this class. Be careful about modifying
    # it.

    def __init__(self, env: Env, id=None):
        super().__init__(env)

        self.id = id
        return

    def observation(self, obs):
        return obs
