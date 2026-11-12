"""Placeholder: Conversion of the response integral to counts/s (L3a ``utils.py``)."""

from __future__ import annotations

import math

import numba
import numpy as np

from imap_processing.swapi.forward_model.response.instrument_response import (
    StepResponse,
)


@numba.njit
def count_rate_conversion_factor(
    density: float, thermal_speed: float, response: StepResponse
) -> float:  # numpydoc ignore=PR01,RT01
    """Placeholder: Compute the factor converting the response integral to counts/s."""
    return (
        response.central_effective_area
        * density
        * (np.sqrt(2 * np.pi) * thermal_speed) ** -3
        * (math.pi / 180.0) ** 2
        * 1e5  # Placeholder: km -> cm
    )
