"""Polynomial fit to SWAPI's energy-angle passband as a function of ESA voltage."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.swapi.constants import (
    PASSBAND_ELEVATIONS_DEG,
    PASSBAND_SPEED_RATIOS,
    SIMION_K_FACTOR,
)
from imap_processing.swapi.forward_model.response.passband import PassbandGrid

# TODO consolidate with AzimuthalRegion
# Claude: sunglasses (SG) or open aperture (OA), as in the ancillary file's "region"
Aperture = Literal["SG", "OA"]
APERTURES: tuple[Aperture, ...] = get_args(Aperture)


@dataclass(frozen=True, eq=False)  # eq=False helps for caching purposes
class PassbandFitTable:
    """
    Claude: Polynomial fits of each aperture's passband in ln(beam energy).

    From the L3a docs (``docs/swapi/response.md``, "Energy-Angle Passbands"): "For each
    pixel, the coefficients c give a polynomial fit to ln P as a function of ln(k*|V|).
    The current ancillary file contains quadratic coefficients, ordered from highest
    degree to constant term. In the future, a higher-order polynomial may be used."

    Attributes
    ----------
    coefficients : xarray.DataArray
        Coefficients (aperture, power, energy_ratio, elevation) of ln(passband),
        highest power first; NaN where there is no fit at an (energy_ratio,
        elevation). The ``aperture`` coordinate holds ``APERTURES``.
        Coordinates ``min_esa_voltage`` and ``max_esa_voltage``
        (aperture, energy_ratio, elevation) give the range of ESA voltages in V
        over which each fit is valid; NaN where there is no fit.
    """

    coefficients: xr.DataArray

    @classmethod
    def from_csv(cls, path: Path) -> PassbandFitTable:
        """
        Placeholder: Read the passband fit coefficients table.

        Parameters
        ----------
        path : pathlib.Path
            The ``passband-fit-coefficients`` ancillary CSV.

        Returns
        -------
        PassbandFitTable
            Fit of each aperture, with coefficients (aperture, power,
            energy_ratio, elevation) on the grid shared by all apertures,
            highest power first, and each fit's valid ESA voltage range in V
            (aperture, energy_ratio, elevation).

        Raises
        ------
        ValueError
            If an aperture has no passband fit in the table.
        """
        fit_table = pd.read_csv(path)
        fit_table["region"] = fit_table["region"].astype(str)
        apertures = list(APERTURES)
        for aperture in apertures:
            if not (fit_table["region"] == aperture).any():
                raise ValueError(f"No passband fit for aperture {aperture}.")
        energy_ratios = np.unique(fit_table["energy_ratio"].to_numpy(dtype=float))
        elevations = np.unique(fit_table["elevation"].to_numpy(dtype=float))
        powers = sorted(
            (int(name) for name in fit_table.columns if name.isdigit()), reverse=True
        )

        # Claude: fill the full (aperture, energy_ratio, elevation) grid, NaN where
        # there is no fit, then put power after aperture
        fit_grid = fit_table.set_index(["region", "energy_ratio", "elevation"]).reindex(
            pd.MultiIndex.from_product([apertures, energy_ratios, elevations])
        )
        grid_shape = (len(apertures), len(energy_ratios), len(elevations))
        coefficient_values = (
            fit_grid[[str(power) for power in powers]]
            .to_numpy(dtype=float)
            .reshape(*grid_shape, len(powers))
            .transpose(0, 3, 1, 2)
        )

        voltage_bounds = {
            f"{bound}_esa_voltage": (
                ("aperture", "energy_ratio", "elevation"),
                fit_grid[f"{bound}_esa_voltage"]
                .to_numpy(dtype=float)
                .reshape(grid_shape),
                {"units": "V"},
            )
            for bound in ("min", "max")
        }
        return cls(
            coefficients=xr.DataArray(
                coefficient_values,
                dims=["aperture", "power", "energy_ratio", "elevation"],
                coords={
                    "aperture": apertures,
                    "power": powers,
                    "energy_ratio": ("energy_ratio", energy_ratios, {"units": "eV/V"}),
                    "elevation": ("elevation", elevations, {"units": "deg"}),
                    **voltage_bounds,
                },
            )
        )


@functools.cache
def evaluate_passband(
    table: PassbandFitTable, esa_voltage: float, aperture: Aperture
) -> PassbandGrid:
    """
    Placeholder: Evaluate an aperture's normalized passband at one ESA voltage.

    Results are cached without a size limit. ``table`` hashes by identity and ESA
    voltages come from a finite sweep table, so there are few distinct keys.
    Repeated calls return the same ``PassbandGrid``, which must not be mutated.

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
    table : PassbandFitTable
        The passband fit table.
    esa_voltage : float
        ESA voltage magnitude in V; it is clamped to each fit's valid range.
    aperture : {"SG", "OA"}
        Sunglasses or open aperture.

    Returns
    -------
    PassbandGrid
        The passband on the ``PASSBAND_ELEVATIONS_DEG`` x
        ``PASSBAND_SPEED_RATIOS`` grid, normalized to 1 at its center by
        ``PassbandGrid.from_values``.
    """
    coefficients = (
        table.coefficients.sel(aperture=aperture)
        .dropna("energy_ratio", how="all")
        .dropna("elevation", how="all")
        .transpose("power", "elevation", "energy_ratio")
    )
    fitted_elevations = list(coefficients.elevation.values)
    fitted_speed_ratios = np.sqrt(coefficients.energy_ratio.values / SIMION_K_FACTOR)

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

    values = np.zeros((len(PASSBAND_ELEVATIONS_DEG), len(PASSBAND_SPEED_RATIOS)))
    for i, elevation in enumerate(PASSBAND_ELEVATIONS_DEG):
        if elevation in fitted_elevations:
            values[i] = np.interp(
                PASSBAND_SPEED_RATIOS,
                fitted_speed_ratios,
                fitted[fitted_elevations.index(elevation)],
                left=0.0,
                right=0.0,
            )
    return PassbandGrid.from_values(values)
