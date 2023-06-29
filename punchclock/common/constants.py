"""Constants: Module for constants used throughout repo."""
# %% Imports
from __future__ import annotations


# %% Function
def getConstants() -> dict:
    """Returns constants used throughout Punch Clock repo.

    Returns:
        dict: {
            earth_radius (`float`): Spherical Earth radius (km),
            mu (`float`): Earth's gravitational parameter (km^3/s^2)
        }
    """
    params = {}

    # Earth radius (km)
    params["earth_radius"] = 6378.1363

    # gravitational parameter of Earth (km^3/s^2)
    params["mu"] = 398600

    # Earth's rotation rate around K-axis (rad/s)
    params["earth_rotation_rate"] = 7.27e-5

    return params
