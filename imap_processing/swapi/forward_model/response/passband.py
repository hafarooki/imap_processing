"""Energy-angle passband for SWAPI at a fixed ESA voltage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numba
import numba.typed
import numpy as np
import numpy.typing as npt
import xarray as xr


class PassbandGrid(NamedTuple):
    """
    Energy-angle passband for SWAPI at a fixed ESA voltage for a particular aperture.

    Stores the interpolated values for a given V and region(phi) in addition to an
    integration contour of 1% relative to the maximum value.

    ``speed_ratio_spans`` maps an elevation row index to the (lowest, highest)
    speed-ratio contour values on that row. Rows that are entirely below the contour
    threshold have no entry.
    """

    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    values: npt.NDArray[np.float64]
    speed_ratio_spans: Mapping[int, tuple[float, float]]
    elevation_range: tuple[float, float]

    @classmethod
    def from_values(
        cls,
        raw_values: xr.DataArray,
        *,
        min_elevation: float = -15.0,
        max_elevation: float = 15.0,
        elevation_spacing: float = 0.5,
        min_speed_ratio: float = 0.9,
        max_speed_ratio: float = 1.1,
        speed_ratio_spacing: float = 0.002,
        contour_threshold: float = 1e-2,
    ) -> PassbandGrid:
        """
        Construct PassbandGrid object from unnormalized values.

        Handles interpolation, normalization, and computing the intergation contours.

        Parameters
        ----------
        raw_values : xarray.DataArray
            Unnormalized passband values with dimensions and coordinates
            ``elevation`` [degrees] and ``speed_ratio``.
        min_elevation : float
            First elevation of the uniform grid [degrees].
        max_elevation : float
            Last elevation of the uniform grid [degrees].
        elevation_spacing : float
            Elevation spacing of the uniform grid [degrees].
        min_speed_ratio : float
            First speed ratio of the uniform grid.
        max_speed_ratio : float
            Last speed ratio of the uniform grid.
        speed_ratio_spacing : float
            Speed ratio spacing of the uniform grid.
        contour_threshold : float
            Level of the integration contour, relative to the passband's maximum.

        Returns
        -------
        PassbandGrid
            Processed passband object.

        Raises
        ------
        ValueError
            If a problem is identified with the input values.
        """
        grid_elevations = np.linspace(
            min_elevation,
            max_elevation,
            round((max_elevation - min_elevation) / elevation_spacing) + 1,
        )
        grid_speed_ratios = np.linspace(
            min_speed_ratio,
            max_speed_ratio,
            round((max_speed_ratio - min_speed_ratio) / speed_ratio_spacing) + 1,
        )
        grid = (
            raw_values
            # sort first bc interp assumed sorted values
            .sortby(["elevation", "speed_ratio"])
            .interp(
                elevation=grid_elevations,
                speed_ratio=grid_speed_ratios,
                kwargs={"fill_value": 0.0},
            )
            .transpose("elevation", "speed_ratio")
        )

        center_value = float(grid.interp(elevation=0.0, speed_ratio=1.0))
        if not (np.isfinite(center_value) and center_value > 0):
            raise ValueError(
                f"Passband must be positive and finite at its center, not "
                f"{center_value}."
            )
        # ensure consistent numba type: array(float64, 2d, C)
        normalized_values = np.ascontiguousarray(grid.values / center_value)

        cutoff = contour_threshold * float(normalized_values.max())
        elevation_range = _span_above(
            grid_elevations, normalized_values.max(axis=1), cutoff
        )
        if elevation_range is None:
            raise ValueError("Passband contour could not be identified.")
        speed_ratio_spans = numba.typed.Dict.empty(
            numba.types.int64, numba.types.UniTuple(numba.types.float64, 2)
        )
        for i, row in enumerate(normalized_values):
            span = _span_above(grid_speed_ratios, row, cutoff)
            if span is not None:
                speed_ratio_spans[i] = span
        return cls(
            min_elevation=float(grid_elevations[0]),
            elevation_spacing=float(grid_elevations[1] - grid_elevations[0]),
            min_speed_ratio=float(grid_speed_ratios[0]),
            speed_ratio_spacing=float(grid_speed_ratios[1] - grid_speed_ratios[0]),
            values=normalized_values,
            speed_ratio_spans=speed_ratio_spans,
            elevation_range=elevation_range,
        )


@numba.njit
def interpolate_passband(
    passband: PassbandGrid, elevation: float, speed_ratio: float
) -> float:
    """
    Interpolate the passband values to a particular point.

    Parameters
    ----------
    passband : Passband
        The passband to be interpolated.
    elevation : float
        Look direction elevation angle in degrees.
    speed_ratio : float
        Ratio of particle speed to passband central speed.

    Returns
    -------
    float
        The passband value at the specified point.
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


@numba.njit
def speed_ratio_range_at_elevation(
    passband: PassbandGrid, elevation: float
) -> tuple[float, float] | None:
    """
    Compute the speed ratio integration limits for a particular elevation angle.

    This is the passband's integration contour, denoted mathematically
    ``[r_min(theta)v_0, r_max(theta)v_0]``.

    Parameters
    ----------
    passband : Passband
        The passband.
    elevation : float
        Look direction elevation angle in degrees.

    Returns
    -------
    tuple of (float, float) or None
        (lowest, highest) speed ratio contour values, or None if ``elevation`` is
        outside ``passband.elevation_range`` or neither row bracketing it reaches
        the contour level.
    """
    min_elevation, max_elevation = passband.elevation_range
    if elevation < min_elevation or elevation > max_elevation:
        return None

    spans = passband.speed_ratio_spans
    lower_row = int(
        np.floor((elevation - passband.min_elevation) / passband.elevation_spacing)
    )
    upper_row = lower_row + 1

    has_lower = lower_row in spans
    has_upper = upper_row in spans
    if not has_lower and not has_upper:
        return None
    if not has_lower:
        return spans[upper_row]
    if not has_upper:
        return spans[lower_row]
    lower_min, lower_max = spans[lower_row]
    upper_min, upper_max = spans[upper_row]
    return min(lower_min, upper_min), max(lower_max, upper_max)


def _span_above(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], cutoff: float
) -> tuple[float, float] | None:
    """
    Use interpolation to identify where a sampled curve first and last crosses a cutoff.

    The returned values are the linearly interpolated first and last crossings.
    If ``y`` starts (ends) at or above ``cutoff`` for the lower (upper) bound, then
    the first (last) value of ``x`` is returned. If ``y`` never reaches ``cutoff``, then
    None is returned.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissae.
    y : numpy.ndarray
        Ordinates.
    cutoff : float
        Level of the crossings.

    Returns
    -------
    tuple of (float, float) or None
        First and last crossings, or None if there is no crossing.
    """
    above = np.flatnonzero(y >= cutoff)
    if above.size == 0:
        return None
    first, last = int(above[0]), int(above[-1])

    if first == 0:
        lower = x[0]
    else:
        # np.interp needs xp (here y) to be increasing
        # y necessarily increases from first - 1 to first
        lower = np.interp(cutoff, y[[first - 1, first]], x[[first - 1, first]])

    if last == len(y) - 1:
        upper = x[-1]
    else:
        # np.interp needs xp (here y) to be increasing
        # reverse order since y necessarily decreases from last to last + 1
        upper = np.interp(cutoff, y[[last + 1, last]], x[[last + 1, last]])

    return float(lower), float(upper)
