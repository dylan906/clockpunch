"""Visibility map and associated functions."""
# %% Imports
# Third Party Imports

# Third Party Imports
from numpy import ndarray, where, zeros
from satvis.visibility_func import calcVisAndDerVis, isVis, visibilityFunc

# Punch Clock Imports
from punchclock.common.constants import getConstants
from punchclock.common.orbits import getRadialRate

RE = getConstants()["earth_radius"]


# %% Functions
def _prepVisMapInputs(
    sensor_states: ndarray, target_states: ndarray, body_radius: float = None
) -> tuple[float, int, int, ndarray, ndarray]:
    """Prepare inputs for visibility map calculation.

    Used by calcVisMap and calcVisMapAndDerivative.

    Args:
        sensor_states (ndarray): Array of sensor states with shape (6, M).
        target_states (ndarray): Array of target states with shape (6, M).
        body_radius (float, optional): Radius of the body. If None, is assigned
            to Earth's radius (km). Defaults to None.

    Returns:
        Tuple: A tuple containing the body radius, number of sensors, number of targets,
        sensor states, and target states.
    """
    if body_radius is None:
        body_radius = RE

    # Check that 0th dimension of state arrays is 6-long.
    # Doesn't catch errors if M or N == 6.
    if sensor_states.shape[0] != 6:
        raise ValueError("Bad input: sensor_states must be (6, M)")
    if target_states.shape[0] != 6:
        raise ValueError("Bad input: target_states must be (6, M)")

    # Reshape if 1-d arrays passed in
    if sensor_states.ndim == 1:
        sensor_states = sensor_states.reshape((6, 1))
    if target_states.ndim == 1:
        target_states = target_states.reshape((6, 1))

    # get numbers of agents
    num_sensors = sensor_states.shape[1]
    num_targets = target_states.shape[1]
    return body_radius, num_sensors, num_targets, sensor_states, target_states


def calcVisMap(
    sensor_states: ndarray,
    target_states: ndarray,
    body_radius: float = None,
    binary: bool = True,
) -> ndarray[int | float]:
    """Calculate visibility map between M sensors and N targets.

    This function calculates a visibility map that indicates whether each
    sensor-target pair can see each other. The visibility is determined based
    on the states of the sensors and targets and the radius of the celestial body.

    This function anc calcVisMapAndDerivative are similar. Use this if you want
    to save a bit on computation.

    Args:
        sensor_states (ndarray): A 2D array of shape (6, M) representing the
            states of M sensors. Each state is a 6D vector.
        target_states (ndarray): A 2D array of shape (6, N) representing the
            states of N targets. Each state is a 6D vector.
        body_radius (float, optional): The radius of the celestial body. The units
            should match those of the state vectors. Defaults to Earth's radius.
        binary (bool, optional): If True, the visibility map will contain 1s and 0s.
            If False, the visibility map will contain the actual visibility values.
            Defaults to True.

    Returns:
        ndarray[int | float]: A 2D array of shape (N, M) representing the
            visibility map. If binary is True, the map contains 1s and 0s, where 1
            indicates that the corresponding sensor-target pair can see each other.
            If binary is False, the map contains the actual visibility values,
            where values >0 indicate that the corresponding sensor-target pair can
            see each other.
    """
    # use external function for code reuse
    (
        body_radius,
        num_sensors,
        num_targets,
        sensor_states,
        target_states,
    ) = _prepVisMapInputs(sensor_states, target_states, body_radius)

    # initialize visibility map
    vis_map = zeros((num_targets, num_sensors))
    # Loop through sensors and targets, record visibility in vis_map.
    for col, sens in enumerate(sensor_states.T):
        for row, targ in enumerate(target_states.T):
            if binary is True:
                # isVis outputs a bool, but is converted to float by assigning to vis_map
                vis_map[row, col] = isVis(sens[:3], targ[:3], body_radius)
            else:
                vis_map[row, col], _, _, _ = visibilityFunc(
                    r1=sens[:3], r2=targ[:3], RE=body_radius, hg=0, tol=1e-6
                )

    if binary is True:
        # convert vis_map from floats to ints
        vis_map = vis_map.astype("int")

    return vis_map


def calcVisMapAndDerivative(
    sensor_states: ndarray,
    target_states: ndarray,
    body_radius: float = None,
    binary=True,
) -> tuple[ndarray[float], ndarray[float]]:
    """Calculate the visibility map and its derivative.

    This function is the more comprehensive version of calcVisMap. It's a bit
    more computationally expensive, but returns the derivative of the visibility.

    Args:
        sensor_states (ndarray): Array of sensor states.
        target_states (ndarray): Array of target states.
        body_radius (float, optional): Radius of the body. Defaults to Earth's
            radius (km).
        binary (bool, optional): If True, the visibility map will contain 1s and 0s.
            If False, the visibility map will contain the actual visibility values.
            Defaults to True.

    Returns:
        ndarray[float]: [N, M] array. The visibility map as continuous values.
        ndarray[float]: [N, M] array. The derivative of the visibility map. If position
            vectors of any sensor-target pair are aligned, the value is Inf or
            -Inf, dependent on other variables.
    """
    # Calculating the vis map derivative requires calculating the vis map, so we
    # return both values in this function.

    # use external function for code reuse
    (
        body_radius,
        num_sensors,
        num_targets,
        sensor_states,
        target_states,
    ) = _prepVisMapInputs(sensor_states, target_states, body_radius)

    # initialize visibility map
    vis_map = zeros((num_targets, num_sensors))
    vis_map_der = zeros((num_targets, num_sensors))

    for col, sens in enumerate(sensor_states.T):
        for row, targ in enumerate(target_states.T):
            # Get position magnitude rates
            r1mag_dot = getRadialRate(r_vec=sens[:3], v_vec=sens[3:])
            r2mag_dot = getRadialRate(r_vec=targ[:3], v_vec=targ[3:])
            vis_map[row, col], vis_map_der[row, col] = calcVisAndDerVis(
                r1=sens[:3],
                r1dot=sens[3:],
                r1mag_dot=r1mag_dot,
                r2=targ[:3],
                r2dot=targ[3:],
                r2mag_dot=r2mag_dot,
                RE=body_radius,
            )

    if binary is True:
        # convert vis_map from floats to binary
        vis_map = where(vis_map > 0, 1, 0)

    return vis_map, vis_map_der
