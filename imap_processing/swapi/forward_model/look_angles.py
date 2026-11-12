"""
Placeholder: Conversions between SWAPI look angles and flow velocities in SWAPI's frame.

From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Integral"): "The flow
direction in RTN coordinates, d(theta, phi), is obtained by applying the
measurement-specific rotation matrix to d^XYZ(theta, phi) = v/||v|| = (-cos theta sin
phi, -cos theta cos phi, -sin theta)^T."
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def look_angles_to_velocity(
    speed: float, azimuth: float, elevation: float
) -> npt.NDArray[np.float64]:
    r"""
    Placeholder: Compute a velocity in the SWAPI frame from a look direction.

    The formula is:

    .. math::

        \mathbf{v} = -v \begin{bmatrix}
            \cos\theta \sin\phi \\
            \cos\theta \cos\phi \\
            \sin\theta
        \end{bmatrix}

    where :math:`v` is the speed, :math:`\phi` the azimuth and :math:`\theta`
    the elevation.

    (0 deg, 0 deg) implies a flow straight into the boresight, along -y.

    Parameters
    ----------
    speed : float
        Bulk flow speed [km/s].
    azimuth : float
        Azimuth angle [degrees].
    elevation : float
        Elevation angle [degrees].

    Returns
    -------
    numpy.ndarray, shape (3,)
        Velocity vector in the SWAPI frame [km/s].
    """
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    cos_el = np.cos(elevation_rad)
    return -speed * np.array(
        [
            cos_el * np.sin(azimuth_rad),
            cos_el * np.cos(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )


@numba.njit
def velocity_to_look_angles(vx: float, vy: float, vz: float) -> tuple[float, float]:
    """
    Placeholder: Convert a flow velocity in the SWAPI frame to its look angles.

    The input is the direction the particles move; the look direction is
    opposite it. Inverse of :func:`look_angles_to_velocity`.

    Parameters
    ----------
    vx : float
        Flow velocity x component.
    vy : float
        Flow velocity y component.
    vz : float
        Flow velocity z component.

    Returns
    -------
    azimuth : float
        Look azimuth in deg.
    elevation : float
        Look elevation in deg.
    """
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    return (
        np.degrees(np.arctan2(-vx, -vy)),
        np.degrees(np.arcsin(-vz / speed)),
    )
