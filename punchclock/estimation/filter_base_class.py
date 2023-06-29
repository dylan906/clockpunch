"""Filter base class."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from abc import ABCMeta, abstractmethod

# Third Party Imports
from numpy import ndarray


# %% Class definition
class Filter(metaclass=ABCMeta):
    """Abstract base class for generic filter object."""

    def __init__(
        self,
        time: float,
        est_x: ndarray,
        est_p: ndarray,
    ):
        """Attributes for base class.

        Args:
            time (`float`): _description_
            est_x (`ndarray`): _description_
            est_p (`ndarray`): _description_
        """
        self.time = time
        self.est_x = est_x
        self.est_p = est_p

    # Required methods for base class
    @abstractmethod
    def reset():
        """Resets filter."""
        pass
