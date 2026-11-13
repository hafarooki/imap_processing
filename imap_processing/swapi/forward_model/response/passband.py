"""
Placeholder: SWAPI's gridded energy-elevation passband.

From the L3a docs (``docs/swapi/response.md``, "Energy-Angle Passbands"): "The
``PassbandGrid`` class stores the interpolated values for a given V and region(phi)
in addition to an integration contour of 1% relative to the maximum value."
"""

from __future__ import annotations

from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    PASSBAND_BOUNDARY_THRESHOLD,
    PASSBAND_ELEVATIONS_DEG,
    PASSBAND_SPEED_RATIOS,
)

# Placeholder: Passbands are tabulated on a uniform (elevation, speed ratio) grid
_MIN_ELEVATION = float(PASSBAND_ELEVATIONS_DEG[0])
_ELEVATION_SPACING = float(PASSBAND_ELEVATIONS_DEG[1] - PASSBAND_ELEVATIONS_DEG[0])
_MIN_SPEED_RATIO = float(PASSBAND_SPEED_RATIOS[0])
_SPEED_RATIO_SPACING = float(PASSBAND_SPEED_RATIOS[1] - PASSBAND_SPEED_RATIOS[0])


class Passband(NamedTuple):
    """
    Placeholder: Normalized energy-elevation passband of an aperture, on a uniform grid.

    From the L3a docs (``docs/swapi/response.md``, "Energy-Angle Passbands"): "The
    ``PassbandGrid`` class stores the interpolated values for a given V and region(phi)
    in addition to an integration contour of 1% relative to the maximum value."
    """

    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    values: npt.NDArray[np.float64]
    min_boundary: npt.NDArray[np.float64]
    max_boundary: npt.NDArray[np.float64]
    elevation_range: tuple[float, float]


def grid_passband(values: npt.NDArray[np.float64]) -> Passband:
    """
    Placeholder: Build a passband, with its integration contour, from gridded values.

    Parameters
    ----------
    values : numpy.ndarray
        Normalized passband on the ``PASSBAND_ELEVATIONS_DEG`` x
        ``PASSBAND_SPEED_RATIOS`` grid.

    Returns
    -------
    Passband
        The passband, with the contour at ``PASSBAND_BOUNDARY_THRESHOLD`` of
        its maximum.
    """
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * float(values.max())
    speed_ratio_spans = np.array(
        [_span_above(PASSBAND_SPEED_RATIOS, row, cutoff) for row in values]
    )
    return Passband(
        min_elevation=_MIN_ELEVATION,
        elevation_spacing=_ELEVATION_SPACING,
        min_speed_ratio=_MIN_SPEED_RATIO,
        speed_ratio_spacing=_SPEED_RATIO_SPACING,
        values=values,
        min_boundary=speed_ratio_spans[:, 0],
        max_boundary=speed_ratio_spans[:, 1],
        elevation_range=_span_above(
            PASSBAND_ELEVATIONS_DEG, values.max(axis=1), cutoff
        ),
    )


def _span_above(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], cutoff: float
) -> tuple[float, float]:
    """
    Placeholder: Find where a sampled curve first rises to and last falls from a cutoff.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissae.
    y : numpy.ndarray
        Ordinates, at or below ``cutoff`` at both ends.
    cutoff : float
        Level of the crossings.

    Returns
    -------
    lower : float
        Linearly interpolated first crossing; NaN if ``y`` never reaches
        ``cutoff``.
    upper : float
        Linearly interpolated last crossing; NaN if ``y`` never reaches
        ``cutoff``.
    """

    def crossing(i_inside: int, i_outside: int) -> float:
        """
        Placeholder: Interpolate where ``y`` crosses ``cutoff`` between two samples.

        Parameters
        ----------
        i_inside : int
            Index of the sample at or above ``cutoff``.
        i_outside : int
            Index of the neighboring sample below ``cutoff``.

        Returns
        -------
        float
            The crossing's abscissa.
        """
        return float(
            x[i_inside]
            + (cutoff - y[i_inside])
            / (y[i_outside] - y[i_inside])
            * (x[i_outside] - x[i_inside])
        )

    above = np.flatnonzero(y >= cutoff)
    if above.size == 0:
        return np.nan, np.nan

    first, last = int(above[0]), int(above[-1])
    return crossing(first, first - 1), crossing(last, last + 1)


@numba.njit
def interpolate_passband(
    passband: Passband, elevation: float, speed_ratio: float
) -> float:
    """
    Placeholder: Bilinearly interpolate a passband at an elevation and speed ratio.

    Parameters
    ----------
    passband : Passband
        The passband.
    elevation : float
        Look elevation in deg.
    speed_ratio : float
        Speed over the central speed.

    Returns
    -------
    float
        The passband; 0 outside its grid.
    """
    values = passband.values
    i_float = (elevation - passband.min_elevation) / passband.elevation_spacing
    if i_float < 0 or i_float + 1 >= values.shape[0]:
        return 0.0
    j_float = (speed_ratio - passband.min_speed_ratio) / passband.speed_ratio_spacing
    if j_float < 0 or j_float + 1 >= values.shape[1]:
        return 0.0

    i_lower = int(i_float)
    i_upper = i_lower + 1
    i_weight = i_float - i_lower
    j_lower = int(j_float)
    j_upper = j_lower + 1
    j_weight = j_float - j_lower

    return (1 - i_weight) * (
        (1 - j_weight) * values[i_lower, j_lower] + j_weight * values[i_lower, j_upper]
    ) + i_weight * (
        (1 - j_weight) * values[i_upper, j_lower] + j_weight * values[i_upper, j_upper]
    )


# Placeholder: Without the explicit override `fastmath=False`, isnan() misbehaves.
@numba.njit(fastmath=False)
def speed_ratio_range_at_elevation(
    passband: Passband, elevation: float
) -> tuple[float, float]:
    """
    Placeholder: Find the speed-ratio span of a passband at an elevation.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Speed Limits"): "the
    passband's integration contour, denoted [r_min(theta)v_0, r_max(theta)v_0]"

    Parameters
    ----------
    passband : Passband
        The passband.
    elevation : float
        Look elevation in deg.

    Returns
    -------
    lower : float
        Lowest speed ratio of the span over the grid rows either side.
    upper : float
        Highest speed ratio of the span over the grid rows either side.

    Raises
    ------
    ValueError
        If neither grid row has a span.
    """
    n_elevations = passband.min_boundary.shape[0]
    i_float = (elevation - passband.min_elevation) / passband.elevation_spacing
    if i_float < 0.0:
        lower_row = 0
    elif i_float >= n_elevations - 1:
        lower_row = n_elevations - 1
    else:
        lower_row = int(i_float)
    upper_row = lower_row + 1 if lower_row + 1 < n_elevations else n_elevations - 1

    lower_min = passband.min_boundary[lower_row]
    upper_min = passband.min_boundary[upper_row]
    lower_max = passband.max_boundary[lower_row]
    upper_max = passband.max_boundary[upper_row]

    lower_is_nan = np.isnan(lower_min) or np.isnan(lower_max)
    upper_is_nan = np.isnan(upper_min) or np.isnan(upper_max)
    if lower_is_nan and upper_is_nan:
        raise ValueError("Elevation out of range.")
    if lower_is_nan:
        return upper_min, upper_max
    if upper_is_nan:
        return lower_min, lower_max
    return min(upper_min, lower_min), max(upper_max, lower_max)
