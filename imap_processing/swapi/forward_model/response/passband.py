"""
Placeholder: SWAPI's energy-elevation passbands: the fits, and the gridded passband.

From the L3a docs (``docs/swapi/response.md``, "Energy-Angle Passbands"): "The passband
coefficient CSV is indexed by ``region``, ``energy_ratio``, and ``elevation``, where
``energy_ratio`` is (E/q)/|V| (particle energy per charge versus ESA voltage
magnitude)."
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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

# Placeholder: Passbands are tabulated on a uniform (elevation, speed ratio) grid
_MIN_ELEVATION = float(PASSBAND_ELEVATIONS_DEG[0])
_ELEVATION_SPACING = float(PASSBAND_ELEVATIONS_DEG[1] - PASSBAND_ELEVATIONS_DEG[0])
_MIN_SPEED_RATIO = float(PASSBAND_SPEED_RATIOS[0])
_SPEED_RATIO_SPACING = float(PASSBAND_SPEED_RATIOS[1] - PASSBAND_SPEED_RATIOS[0])

# Placeholder: number of sweeps whose passbands each PassbandFits keeps cached
PASSBAND_CACHE_SIZE = 32


# TODO consolidate with AzimuthalRegion
class Aperture(Enum):
    """Placeholder: Aperture whose passband is fit; values match the ancillary file."""

    SUNGLASSES = "SG"
    OPEN_APERTURE = "OA"


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


@dataclass(frozen=True)
class PassbandFit:
    """
    Placeholder: Polynomial fit of one aperture's passband in ln(beam energy).

    From the L3a docs (``docs/swapi/response.md``, "Energy-Angle Passbands"): "For each
    pixel, the coefficients c give a polynomial fit to ln P as a function of ln(k*|V|).
    The current ancillary file contains quadratic coefficients, ordered from highest
    degree to constant term. In the future, a higher-order polynomial may be used."

    Attributes
    ----------
    coefficients : xarray.DataArray
        Coefficients (power, energy_ratio, elevation) of ln(passband), highest
        power first; NaN where there is no fit at an (energy_ratio, elevation).
        Coordinates ``min_esa_voltage`` and ``max_esa_voltage``
        (energy_ratio, elevation) give the range of ESA voltages in V over
        which each fit is valid; NaN where there is no fit.
    """

    coefficients: xr.DataArray

    def evaluate(self, esa_voltage: float) -> Passband:
        """
        Placeholder: Evaluate the normalized passband at one ESA voltage.

        From the L3a docs (``docs/swapi/response.md``, "Energy-Angle Passbands"): "When
        initializing the passband cache for a given |V|, for each region,
        ``SwapiResponse`` clamps |V| to the ``min_esa_voltage``/``max_esa_voltage``
        limits from the CSV for a given region, then evaluates P = exp[polyval(c,
        ln(k*|V_clamped|))]. The ``energy_ratio`` axis is converted to speed ratio using
        v/v_0^s = sqrt(((E/q)/|V|) / k*) [...] After interpolating to the uniform grid,
        the passband is divided by its value at (v/v_0 = 1, theta = 0) so that P_r(1, 0,
        V) = 1 by construction."

        Parameters
        ----------
        esa_voltage : float
            ESA voltage magnitude in V; it is clamped to each fit's valid
            range.

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
        # Placeholder: (elevation, energy_ratio); NaN where there is no fit
        clamped_voltage = np.clip(
            esa_voltage,
            coefficients.min_esa_voltage.values,
            coefficients.max_esa_voltage.values,
        )
        # Placeholder: we use energy-angle passbands for modeling the coincidence rate,
        # but fits are to voltage-angle passbands as a function of natural log of beam
        # energy; effectively equivalent as long as the x-axis is E/|V| and not E or |V|
        # the fits' x-axis is ln of the central beam energy, k |V|
        log_beam_energy = np.log(SIMION_K_FACTOR * clamped_voltage)
        fitted = np.exp(np.polyval(coefficients.values, log_beam_energy))
        # Placeholder: no fit at this (elevation, ratio)
        fitted = np.nan_to_num(fitted, nan=0.0)
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


@dataclass(frozen=True, eq=False)
class PassbandFits:
    """
    Placeholder: Passband fits of both apertures, with a cache of evaluated sweeps.

    :meth:`sweep_passbands` caches the passbands of each sweep, so chunks that
    share ESA voltages reuse them.

    Attributes
    ----------
    fits : dict[Aperture, PassbandFit]
        Passband fit of each aperture.
    """

    fits: dict[Aperture, PassbandFit]
    _cache: OrderedDict[bytes, tuple[list[Passband], list[Passband]]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def passband(self, esa_voltage: float, aperture: Aperture) -> Passband:
        """
        Placeholder: Evaluate an aperture's passband at an ESA voltage, uncached.

        Parameters
        ----------
        esa_voltage : float
            ESA voltage magnitude in V; it is clamped to the fit's valid
            range.
        aperture : Aperture
            Aperture.

        Returns
        -------
        Passband
            The passband.
        """
        return self.fits[aperture].evaluate(esa_voltage)

    def sweep_passbands(
        self, esa_voltages: npt.NDArray[np.float64]
    ) -> tuple[list[Passband], list[Passband]]:
        """
        Claude: Evaluate the passbands of each sample of a sweep, with caching.

        The passbands are shared between calls with the same voltages and must
        not be modified.

        Evaluating the passbands dominates building the measurement species
        responses (~60 ms for a 5 x 72 chunk, against ~1.5 ms for the rest), and
        they depend on the voltages alone.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitude of each sample in V (n,), positive and
            finite.

        Returns
        -------
        sg_passbands : list[Passband]
            Sunglasses passband of each sample.
        oa_passbands : list[Passband]
            Open aperture passband of each sample.
        """
        key = esa_voltages.tobytes()
        cache = self._cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        # Placeholder: evaluate each distinct voltage's passbands once
        unique_voltages = np.unique(esa_voltages)
        sg_by_voltage = {
            v: self.passband(v, Aperture.SUNGLASSES) for v in unique_voltages
        }
        oa_by_voltage = {
            v: self.passband(v, Aperture.OPEN_APERTURE) for v in unique_voltages
        }
        passbands = (
            [sg_by_voltage[v] for v in esa_voltages],
            [oa_by_voltage[v] for v in esa_voltages],
        )
        cache[key] = passbands
        if len(cache) > PASSBAND_CACHE_SIZE:
            cache.popitem(last=False)
        return passbands


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
