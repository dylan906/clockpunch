"""Dynamics classes module."""
# %% Imports
from __future__ import annotations

# Standard Library Imports
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

# Third Party Imports
from numpy import ndarray, reshape, zeros

# Punch Clock Imports
from punchclock.dynamics.dynamics import (
    satelliteDynamics,
    terrestrialDynamicsAlt,
)
from punchclock.dynamics.propagator import simplePropagate


class DynamicsModel(ABC):
    """Abstract base class for dynamics models."""

    def __init__(
        self,
        dynamics_func: Callable[[ndarray, float, float], ndarray],
    ):
        self._dynamics_func = dynamics_func

    @abstractmethod
    def propagate(
        self,
        start_state: ndarray,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> ndarray:
        return self._dynamics_func(
            start_state,
            start_time,
            end_time,
            **kwargs,
        )


class SatDynamicsModel(DynamicsModel):
    """Class for Earth satellite dynamics."""

    def __init__(self):
        """Initialize satellite dynamics instance."""
        dynamics_func = partial(simplePropagate, satelliteDynamics)
        super().__init__(dynamics_func)

    def propagate(
        self,
        start_state: ndarray,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> ndarray:
        """Propagate satellite dynamics.

        Args:
            start_state (`ndarray`): [6x1] ECI state vector (km, km/s).
            start_time (`float`): Time of starting state (sec).
            end_time (`float`): Time to propagate to (sec).

        Returns:
            ndarray: [6x1] ECI state vector at time `end_time`.
        """
        return super().propagate(start_state, start_time, end_time, **kwargs)


class StaticTerrestrial(DynamicsModel):
    """Class for static terrestrial dynamics (Earth-fixed)."""

    def __init__(self):
        """Initialize `StaticTerrestrial` dynamics object."""
        dynamics_func = terrestrialDynamicsAlt
        super().__init__(dynamics_func)

    def propagate(
        self,
        start_state: ndarray,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> ndarray:
        """Propagate static terrestrial dynamics.

        Args:
            start_state (`ndarray`): (6, N) or (6,) ECI state vector (km, km/s).
            start_time (`float`): Julian date (or generic absolute measure of time).
            end_time (`float`): Julian date (or generic absolute measure of time).

        Returns:
            ndarray: (6, N) or (6,) ECI state vector at time `end_time`.
        """
        # reshape to standardize single- and multi-input
        if start_state.ndim == 1:
            start_state = reshape(start_state, (6, 1))

        end_state = zeros(start_state.shape)
        for i, vec in enumerate(start_state.transpose()):
            end_state[:, i] = (
                super()
                .propagate(
                    vec,
                    start_time,
                    end_time,
                    **kwargs,
                )
                .squeeze()
            )

        return end_state.squeeze()
