"""Maxwellian moments and the thermal-speed conversions (L3a ``params.py``)."""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import BOLTZMANN_CONSTANT_J_PER_K
from imap_processing.swapi.forward_model.response.instrument_response import (
    METERS_PER_KILOMETER,
    StepResponse,
)


class _MaxwellianParams(NamedTuple):
    """Moments of a Maxwellian population."""

    density: float
    velocity_xyz: npt.NDArray[np.float64]
    temperature: float


@numba.njit
def bulk_speed(sw_params: _MaxwellianParams) -> float:  # numpydoc ignore=PR01,RT01
    """Compute the bulk speed in km/s."""
    v = sw_params.velocity_xyz
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


@numba.njit
def thermal_speed(
    sw_params: _MaxwellianParams, response: StepResponse
) -> float:  # numpydoc ignore=PR01,RT01
    """Compute the thermal speed sqrt(kT/m) in km/s."""
    return temperature_to_thermal_speed(response.mass_kg, sw_params.temperature)


@numba.njit
def temperature_to_thermal_speed(
    mass: float, temperature: float
) -> float:  # numpydoc ignore=PR01,RT01
    """Convert a temperature (K) to a thermal speed sqrt(kT/m) (km/s)."""
    return (
        math.sqrt(BOLTZMANN_CONSTANT_J_PER_K * temperature / mass)
        / METERS_PER_KILOMETER
    )


@numba.njit
def thermal_speed_to_temperature(
    thermal_speed: float, mass: float
) -> float:  # numpydoc ignore=PR01,RT01
    """Convert a thermal speed sqrt(kT/m) (km/s) to a temperature (K)."""
    return (
        mass * (thermal_speed * METERS_PER_KILOMETER) ** 2 / BOLTZMANN_CONSTANT_J_PER_K
    )
