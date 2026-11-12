"""Placeholder: A drifting Maxwellian seen by one ESA step's species."""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import BOLTZMANN_CONSTANT_J_PER_K
from imap_processing.swapi.forward_model.look_angles import velocity_to_look_angles
from imap_processing.swapi.forward_model.response.instrument_response import (
    METERS_PER_KILOMETER,
)


class Maxwellian(NamedTuple):
    """
    Placeholder: A drifting Maxwellian with the quantities derived from its moments.

    Build it with :func:`make_maxwellian` so the derived fields agree with the
    moments; numba can't run a custom ``__init__`` on a ``NamedTuple``.
    """

    density: float
    velocity_xyz: npt.NDArray[np.float64]
    bulk_speed: float
    thermal_speed: float
    bulk_azimuth: float
    bulk_elevation: float


@numba.njit
def make_maxwellian(
    density: float,
    temperature: float,
    velocity_xyz: npt.NDArray[np.float64],
    mass_kg: float,
) -> Maxwellian:
    """
    Placeholder: Build a Maxwellian from its moments and a species mass.

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
    Maxwellian
        The moments with the bulk speed (km/s), thermal speed sqrt(kT/m) (km/s)
        and the look azimuth and elevation (deg) of the bulk flow.
    """
    v = velocity_xyz
    bulk_azimuth, bulk_elevation = velocity_to_look_angles(v[0], v[1], v[2])
    return Maxwellian(
        density=density,
        velocity_xyz=velocity_xyz,
        bulk_speed=math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2),
        thermal_speed=math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * temperature / mass_kg)
        / METERS_PER_KILOMETER,
        bulk_azimuth=bulk_azimuth,
        bulk_elevation=bulk_elevation,
    )
