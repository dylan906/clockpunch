"""Target custody tracker."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Callable

# Third Party Imports
from numpy import bool_, diagonal, ndarray, ones, sqrt, trace


# %% CustodyTracker class
class CustodyTracker:
    """Keeps track of target custody as a bool."""

    def __init__(
        self,
        num_targets: int,
        config: dict = None,
        target_names: list = None,
        initial_status: list[bool] = None,
    ):
        """Initialize CustodyTracker.

        Args:
            num_targets (int): Number of targets (>0).
            config (dict, optional): If using a supported function, structure as:
                {
                    "func": name (str),
                    "threshold": value (float),
                }.
                If using a custom function, structure as:
                {
                    "func": func (Callable),
                }.
                Defaults to:
                {
                    "func": "max_pos_std",
                    "threshold": 1000,
                }.
            target_names (list, optional): Target names for use in detailed return
                map. Defaults to sequential integers starting at 0.
            initial_status (list[bool], optional): Initial custody status for all
                targets. Defaults to all True.

        Available preset functions ("func"):
            "max_pos_std": Maximum positional STD. See MaxPosStd for details.
            "tr_cov": Trace of covariance. See TrCov for details.
            "tr_pos_cov": Trace of positional covariance. See TrCov for details.
            "tr_vel_cov": Trace of velocity covariance. See TrCov for details.

        If input, target_names and initial_status must have length num_targets.

        If using a custom custody function, recommend using partial() to fix as
        many arguments as possible. Keyword arguments can be used via
        CustodyTracker.update(..., `**kwargs`).

        Custom custody function must return list of bools.
        """
        assert num_targets > 0, "num_targets must be >0."

        # Defaults
        if initial_status is None:
            initial_status = ones((num_targets), dtype=bool)
        if target_names is None:
            target_names = [i for i in range(num_targets)]
        if config is None:
            # Default config
            config = {
                "func": "max_pos_std",
                "threshold": 1000,
            }

        assert (
            len(target_names) == num_targets
        ), "len(target_names) must equal num_targets"
        assert (
            len(initial_status) == num_targets
        ), "len(initial_status) must equal num_targets"

        self.num_targets = num_targets
        self.target_names = target_names
        self.custody_status = initial_status
        self.custody_map = dict(zip(target_names, initial_status))
        self.config = config
        self.custodyFunc = self.getCustodyFunc(config)

    def getCustodyFunc(self, config: dict) -> Callable[..., list[bool]]:
        """Get the custody function to be used on self.update()."""
        # Custody function must output a list of bools, but can take any type input(s)

        config_func = config["func"]
        if isinstance(config_func, Callable):
            # If custom function provided
            custodyFunc = config_func
        else:
            # supported default functions.
            custody_func_map = {
                "max_pos_std": partial(
                    MaxPosStd(), threshold=config["threshold"]
                ),
                "tr_cov": partial(TrCov(), threshold=config["threshold"]),
                "tr_pos_cov": partial(
                    TrCov("pos"), threshold=config["threshold"]
                ),
                "tr_vel_cov": partial(
                    TrCov("vel"), threshold=config["threshold"]
                ),
            }
            custodyFunc = custody_func_map[config_func]

        return custodyFunc

    def update(
        self,
        obs: Any,
        return_map: bool = False,
        **kwargs,
    ) -> list[bool] | dict:
        """Update target custody status (see CustodyTracker.custody_status).

        Args:
            obs (Any): Must comply with previously-defined interface of
                CustodyTracker.custodyFunc.
            return_map (bool, optional): If True, returns a dict with custody status
                mapped to target name. Otherwise, returns a list of custody statuses
                (True/False). Defaults to False.
            `**kwargs`: Used for custom custody functions.

        Returns:
            list[bool] | dict: Custody status as a list or a mapping of target
                names to status.
        """
        custody_status = self.custodyFunc(obs, **kwargs)

        assert (
            len(custody_status) == self.num_targets
        ), "custodyFunc must return an N-long list."
        assert all(
            isinstance(x, (bool, bool_)) for x in custody_status
        ), "custodyFunc must return a list of bools."

        self.custody_status = custody_status
        self.custody_map = dict(zip(self.target_names, self.custody_status))

        if return_map is True:
            out = deepcopy(self.custody_map)
        else:
            out = deepcopy(self.custody_status)

        return out


# %% Debug custody tracking function
class DebugCustody:
    """Used to debug CustodyTracker.

    The input to DebugCustody.update() is the same as the output.
    """

    def __init__(self, num_targets: int):
        """Initialize."""
        self.num_targets = num_targets

    def update(self, custody: ndarray) -> list[bool]:
        """Input a 1d custody array, output a list of bools.

        Args:
            custody (ndarray): 1s and 0s. Must be 1d.

        Returns:
            list[bool]: 1s are converted to True, 0s to False.
        """
        assert custody.ndim == 1
        assert all([c in [0, 1] for c in custody])
        assert len(custody) == self.num_targets

        return list(custody.astype(bool))


# %% Covariance-based custody functions
class BaseCovCustody(ABC):
    """Base class for covariance-based custody functions.

    Enforces standardized args/returns for a custody function.
        - Args must include cov (an (N, 6, 6) array) and threshold (float | int).
        - Return must be a N-long list of bools.
        - Child class must have ._customLogic() method.

    Example use:
        childFunc = ChildFuncClass()
        custody = childFunc(
            cov=rand(2, 6, 6),
            threshold=1,
        )
        print(custody) # prints a 2-long list of bools

    Example of correct child class:
        class CorrectFunc(BaseCovCustody):
            def _uniqueLogic(self, cov: ndarray, threshold: float):
                return [True for i in range(cov.shape[0])]

    Example of incorrect child class (return list is too long):
        class BadFunc(BaseCovCustody):
            def _uniqueLogic(self, cov: ndarray, threshold: float):
                return [True for i in range(cov.shape[0] + 1)]
    """

    def checkCovArgs(self, cov: Any, threshold: Any):
        """Check if covariance meets shape and type requirements."""
        assert cov.ndim == 3, "cov must be a 3-dimensional array."
        assert cov.shape[1] == 6, "cov must be a (N, 6, 6) array."
        assert cov.shape[2] == 6, "cov must be a (N, 6, 6) array."
        assert isinstance(
            threshold, (int, float)
        ), "threshold must be one of [int, float]."

    def checkReturns(self, y: Any, cov: Any):
        """Check if return list meets type and size requirements."""
        assert isinstance(y, list), "Return value must be a list."
        assert all(
            isinstance(x, bool) for x in y
        ), "All entries in return value must be bools."
        assert (
            len(y) == cov.shape[0]
        ), """Length of return list must be same as 0-th dimension of cov
         (len(return) == cov.shape[0])."""

    def __call__(self, cov: ndarray, threshold: float):
        """Check arguments, run unique logic, and check returns."""
        self.checkCovArgs(cov, threshold)
        retval = self._uniqueLogic(cov, threshold)
        self.checkReturns(retval, cov)
        return retval

    @abstractmethod
    def _uniqueLogic(self, cov: ndarray, threshold: float):
        """Required for all child classes."""
        raise NotImplementedError()


class MaxPosStd(BaseCovCustody):
    """Target in custody if max position STD is below threshold."""

    def _uniqueLogic(self, cov: ndarray, threshold: float):
        """Evaluate the function.

        x =  sqrt(max(diagonal(cov[i, :3, :3])))

        custody[i] = True if x < threshold, False otherwise.
        """
        custody = [True for i in range(cov.shape[0])]
        for i in range(cov.shape[0]):
            c = cov[i, :, :]
            c_diags = diagonal(c)
            std_diags = sqrt(c_diags)
            pos_std_diags = std_diags[:3]
            if max(pos_std_diags) > threshold:
                custody[i] = False

        return custody


class TrCov(BaseCovCustody):
    """Target in custody if trace of covariance is below threshold.

    Optionally use positional or velocity covariance instead of whole matrix.
    """

    def __init__(self, pos_vel: str = "both"):
        """Set evaluation mode to position/velocity/both.

        Args:
            pos_vel (str, optional): ["pos" | "vel" | "both"]. Sets evaluation
                function to use positional, velocity, or both components of covariance
                matrix. Positional components are assumed to be upper-left quadrant
                of covariance matrix. Defaults to "both".
        """
        assert pos_vel in [
            "both",
            "pos",
            "vel",
        ], "pos_vel must be one of 'both', 'pos', or 'vel'."

        # Set array slicing indices to get either upper-left or lower-right 3x3
        # sub-array from covariance.
        if pos_vel == "both":
            self.slice_indices = [None, None]
        elif pos_vel == "pos":
            self.slice_indices = [None, 3]
        elif pos_vel == "vel":
            self.slice_indices = [3, None]

    def _uniqueLogic(self, cov: ndarray, threshold: float) -> list[bool]:
        """Evaluate the function.

        x =  tr(cov[i, a:b, a:b])

        custody[i] = True if x < threshold, False otherwise.

        Variables a and b are set at initialization depending on which components
            (positional, velocity, both) of covariance matrix to use.
        """
        custody = [True for i in range(cov.shape[0])]
        for i in range(cov.shape[0]):
            c = cov[
                i,
                self.slice_indices[0] : self.slice_indices[1],
                self.slice_indices[0] : self.slice_indices[1],
            ]
            if trace(c) > threshold:
                custody[i] = False
        return custody
