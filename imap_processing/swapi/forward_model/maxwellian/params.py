"""Maxwellian distribution parameters."""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    METERS_PER_KILOMETER,
)
from imap_processing.swapi.forward_model.look_angles import velocity_to_look_angles


class MaxwellianParams(NamedTuple):
    """Parameters for Maxwellian distribution (some derived from others)."""

    density: float
    velocity_xyz: npt.NDArray[np.float64]
    bulk_speed: float
    thermal_speed_sigma: float
    bulk_azimuth: float
    bulk_elevation: float


@numba.njit
def make_maxwellian(
    density: float,
    temperature: float,
    velocity_xyz: npt.NDArray[np.float64],
    mass_kg: float,
) -> MaxwellianParams:
    """
    Build a MaxwellianParams object from the basic parameters and species mass.

    Parameters
    ----------
    density : float
        Number density in cm^-3.
    temperature : float
        Temperature in K.
    velocity_xyz : numpy.ndarray
        Bulk flow velocity in the SWAPI frame in km/s (3,).
    mass_kg : float
        Species mass in kg.

    Returns
    -------
    params : MaxwellianParams
    """
    v = velocity_xyz
    bulk_azimuth, bulk_elevation = velocity_to_look_angles(v[0], v[1], v[2])
    bulk_speed = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    thermal_speed_sigma = (
        math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * temperature / mass_kg)
        / METERS_PER_KILOMETER
    )
    return MaxwellianParams(
        density=density,
        velocity_xyz=velocity_xyz,
        bulk_speed=bulk_speed,
        thermal_speed_sigma=thermal_speed_sigma,
        bulk_azimuth=bulk_azimuth,
        bulk_elevation=bulk_elevation,
    )
