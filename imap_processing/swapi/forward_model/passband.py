"""SWAPI's energy-elevation passbands: the fits, and the gridded passband."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt
import xarray as xr

from imap_processing.swapi.constants import (
    PASSBAND_BOUNDARY_THRESHOLD,
    PASSBAND_ELEVATIONS_DEG,
    PASSBAND_SPEED_RATIOS,
    SIMION_K_FACTOR,
)

# Passbands are tabulated on a uniform (elevation, speed ratio) grid
_MIN_ELEVATION = float(PASSBAND_ELEVATIONS_DEG[0])
_ELEVATION_SPACING = float(PASSBAND_ELEVATIONS_DEG[1] - PASSBAND_ELEVATIONS_DEG[0])
_MIN_SPEED_RATIO = float(PASSBAND_SPEED_RATIOS[0])
_SPEED_RATIO_SPACING = float(PASSBAND_SPEED_RATIOS[1] - PASSBAND_SPEED_RATIOS[0])


class Aperture(Enum):
    """Aperture whose passband is fit; values match the ancillary file."""

    SUNGLASSES = "SG"
    OPEN_APERTURE = "OA"


class Passband(NamedTuple):
    """
    Normalized energy-elevation passband of one aperture, on a uniform grid.

    Attributes
    ----------
    min_elevation : float
        Elevation of the first row of ``values`` in deg.
    elevation_spacing : float
        Elevation spacing of the rows of ``values`` in deg.
    min_speed_ratio : float
        Speed ratio (speed over the central speed) of the first column of
        ``values``.
    speed_ratio_spacing : float
        Speed ratio spacing of the columns of ``values``.
    values : numpy.ndarray
        Passband (elevation, speed_ratio), >= 0, equal to 1 at elevation 0,
        speed ratio 1. C-contiguous.
    min_boundary : numpy.ndarray
        Speed ratio at which each row of ``values`` first rises to 1% of the
        peak; NaN where it never does.
    max_boundary : numpy.ndarray
        Speed ratio at which each row of ``values`` last falls from 1% of the
        peak; NaN where it never does.
    elevation_range : tuple[float, float]
        The same bounds along elevation in deg, for the passband maximized over
        speed ratio.
    """

    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    values: npt.NDArray[np.float64]
    min_boundary: npt.NDArray[np.float64]
    max_boundary: npt.NDArray[np.float64]
    elevation_range: tuple[float, float]


class Passbands(NamedTuple):
    """
    One aperture's passbands at several ESA energies, stacked on a shared grid.

    Attributes
    ----------
    min_elevation : float
        Elevation of the first row of each passband in deg.
    elevation_spacing : float
        Elevation spacing of the rows in deg.
    min_speed_ratio : float
        Speed ratio of the first column of each passband.
    speed_ratio_spacing : float
        Speed ratio spacing of the columns.
    values : numpy.ndarray
        Passbands (energy, elevation, speed_ratio); see :class:`Passband`.
    min_boundary : numpy.ndarray
        Lower speed-ratio boundaries (energy, elevation).
    max_boundary : numpy.ndarray
        Upper speed-ratio boundaries (energy, elevation).
    elevation_range : numpy.ndarray
        Elevation bounds (energy, 2) in deg.
    """

    min_elevation: float
    elevation_spacing: float
    min_speed_ratio: float
    speed_ratio_spacing: float
    values: npt.NDArray[np.float64]
    min_boundary: npt.NDArray[np.float64]
    max_boundary: npt.NDArray[np.float64]
    elevation_range: npt.NDArray[np.float64]

    @classmethod
    def stack(cls, passbands: list[Passband]) -> Passbands:
        """
        Stack passbands evaluated on the shared grid.

        Parameters
        ----------
        passbands : list[Passband]
            Passbands, e.g. from :meth:`PassbandFit.evaluate`.

        Returns
        -------
        Passbands
            The passbands in the order given.
        """
        return cls(
            min_elevation=_MIN_ELEVATION,
            elevation_spacing=_ELEVATION_SPACING,
            min_speed_ratio=_MIN_SPEED_RATIO,
            speed_ratio_spacing=_SPEED_RATIO_SPACING,
            values=np.stack([p.values for p in passbands]),
            min_boundary=np.stack([p.min_boundary for p in passbands]),
            max_boundary=np.stack([p.max_boundary for p in passbands]),
            elevation_range=np.array(
                [p.elevation_range for p in passbands], dtype=float
            ).reshape(-1, 2),
        )


@numba.njit  # type: ignore[untyped-decorator]
def passband_at(passbands: Passbands, index: int) -> Passband:
    """
    Select one passband from a stack, without copying its arrays.

    Parameters
    ----------
    passbands : Passbands
        The stacked passbands.
    index : int
        Index of the passband along the energy axis.

    Returns
    -------
    Passband
        The passband, viewing the stack's arrays.
    """
    return Passband(
        passbands.min_elevation,
        passbands.elevation_spacing,
        passbands.min_speed_ratio,
        passbands.speed_ratio_spacing,
        passbands.values[index],
        passbands.min_boundary[index],
        passbands.max_boundary[index],
        (passbands.elevation_range[index, 0], passbands.elevation_range[index, 1]),
    )


@dataclass(frozen=True)
class PassbandFit:
    """
    Polynomial fit of one aperture's passband in ln(beam energy).

    Attributes
    ----------
    coefficients : xarray.DataArray
        Coefficients (power, energy_ratio, elevation) of ln(passband), highest
        power first; NaN where there is no fit at an (energy_ratio, elevation).
        Coordinates ``min_esa_energy`` and ``max_esa_energy``
        (energy_ratio, elevation) give the range of ESA energies per charge in
        eV/q over which each fit is valid; NaN where there is no fit.
    """

    coefficients: xr.DataArray

    def evaluate(self, esa_energy: float) -> Passband:
        """
        Evaluate the normalized passband at one ESA energy.

        Parameters
        ----------
        esa_energy : float
            ESA energy per charge in eV/q, ``SIMION_K_FACTOR`` times the ESA
            voltage; it is clamped to each fit's valid range.

        Returns
        -------
        Passband
            The passband on the ``PASSBAND_ELEVATIONS_DEG`` x
            ``PASSBAND_SPEED_RATIOS`` grid.
        """
        coefficients = (
            self.coefficients.dropna("energy_ratio", how="all")
            .dropna("elevation", how="all")
            .transpose("power", "elevation", "energy_ratio")
        )
        # (elevation, energy_ratio); NaN where there is no fit
        clamped_energy = np.clip(
            esa_energy,
            coefficients.min_esa_energy.values,
            coefficients.max_esa_energy.values,
        )
        # we use energy-angle passbands for modeling the coincidence rate,
        # but fits are to voltage-angle passbands as a function of natural log
        # of beam energy; effectively equivalent as long as the x-axis is E/|V|
        # and not E or |V|
        log_beam_energy = np.log(clamped_energy)
        fitted = np.exp(np.polyval(coefficients.values, log_beam_energy))
        fitted = np.nan_to_num(fitted, nan=0.0)  # no fit at this (elevation, ratio)
        fitted_speed_ratios = np.sqrt(
            coefficients.energy_ratio.values / SIMION_K_FACTOR
        )

        values = np.zeros((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)))
        fitted_elevations = list(coefficients.elevation.values)
        for i, elevation in enumerate(PASSBAND_ELEVATIONS_DEG):
            if elevation in fitted_elevations:
                values[i] = np.interp(
                    PASSBAND_SPEED_RATIOS,
                    fitted_speed_ratios,
                    fitted[fitted_elevations.index(elevation)],
                    left=0.0,
                    right=0.0,
                )
        i_center = int(np.argmin(np.abs(PASSBAND_ELEVATIONS_DEG)))
        values /= np.interp(1.0, PASSBAND_SPEED_RATIOS, values[i_center])

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
    Find where a tabulated curve first rises to and last falls from a cutoff.

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
        Interpolate where ``y`` crosses ``cutoff`` between two samples.

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


@numba.njit  # type: ignore[untyped-decorator]
def interpolate_passband(
    passband: Passband, elevation: float, speed_ratio: float
) -> float:
    """
    Bilinearly interpolate a passband at an elevation and speed ratio.

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


# Without the explicit override `fastmath=False`, isnan() misbehaves.
@numba.njit(fastmath=False)  # type: ignore[untyped-decorator]
def speed_ratio_range_at_elevation(
    passband: Passband, elevation: float
) -> tuple[float, float]:
    """
    Find the speed-ratio span of a passband at an elevation.

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
