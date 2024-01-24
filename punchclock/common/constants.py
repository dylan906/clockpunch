"""Constants: Module for constants used throughout repo."""
# Reference Vallado, 3rd Edition
# %% Imports
# Third Party Imports
from numpy import pi


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

    # Day length (sec)
    params["sidereal_day"] = 86164.090517
    params["solar_day"] = 86400

    # Radius (Semi-major axis) of geosynchronous orbit (km)
    params["gso_radius"] = (
        (params["sidereal_day"] ** 2) / (4 * (pi**2)) * params["mu"]
    ) ** (1 / 3)

    # Altitude of geosynchronous orbit (km)
    params["gso_altitude"] = params["gso_radius"] - params["earth_radius"]

    return params
