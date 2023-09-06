"""Misc wrappers."""
# %% Imports

# Third Party Imports
from gymnasium import Env, Wrapper


# %% Identity Wrapper
class IdentityWrapper(Wrapper):
    """Wrapper does not modify environment, used for construction."""

    # NOTE: SimRunner is hugely dependent on this class. Be careful about modifying
    # it.

    def __init__(self, env: Env, id: Any = None):  # noqa
        """Wrap environment with IdentityWrapper.

        Args:
            env (Env): A Gymnasium environment.
            id (Any, optional): Mainly used to distinguish between multiple instances
                of IdentityWrapper. Defaults to None.
        """
        super().__init__(env)

        self.id = id
        return

    def observation(self, obs):
        """Pass-through observation."""
        return obs
