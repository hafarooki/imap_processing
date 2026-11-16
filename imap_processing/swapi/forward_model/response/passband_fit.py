"""
Polynomial fit to SWAPI's energy-angle passband as a function of ESA voltage.

For each pixel, the coefficients c give a polynomial fit to ln P as a function of
ln(k*|V|). The current ancillary file contains quadratic coefficients, ordered from
highest degree to constant term. In the future, a higher-order polynomial may be
used.

The fits' x-axis is ln of the central beam energy, k |V|. We use energy-angle passbands
for modeling the coincidence rate, but fits were made to voltage-angle passbands as a
function of natural log of beam energy; effectively equivalent as long as the x-axis is
E/|V| and not E or |V|.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.swapi.constants import SIMION_K_FACTOR
from imap_processing.swapi.forward_model.response.passband import PassbandGrid



class PassbandFitTable:
    """
    Passband by ESA voltage.

    Stores the passband-vs-voltage coefficients, evaluates the passband
    values at a given ESA voltage, and caches the result for repeated calls.

    Parameters
    ----------
    path : pathlib.Path
        The ``passband-fit-coefficients`` ancillary CSV.
    """

    def __init__(self, path: Path) -> None:
        self._cached_passbands: dict[tuple[float, str], PassbandGrid] = {}

        fit_table = pd.read_csv(path).rename(columns={"region": "aperture"})
        fit_table["aperture"] = fit_table["aperture"].astype(str)

        fits = (
            fit_table.set_index(["aperture", "energy_ratio", "elevation"])
            .to_xarray()
            .reindex(aperture=["SG", "OA"])
        )
        fits["energy_ratio"].attrs["units"] = "eV/V"
        fits["elevation"].attrs["units"] = "deg"
        for bound in ("min_esa_voltage", "max_esa_voltage"):
            fits[bound].attrs["units"] = "V"
        fits = fits.set_coords(["min_esa_voltage", "max_esa_voltage"])

        self._coefficients = (
            fits.to_dataarray("power")
            .assign_coords(power=lambda da: da.power.astype(int))
            .sortby("power", ascending=False)
        )

    def evaluate_passband(
        self, esa_voltage: float, aperture: Literal["SG", "OA"]
    ) -> PassbandGrid:
        """
        Evaluate the normalized passband for a particular ESA voltage and aperture.

        Parameters
        ----------
        esa_voltage : float
            ESA voltage magnitude in volts.
        aperture : {"SG", "OA"}
            Sunglasses or open aperture, as in the ancillary file's ``region``.

        Returns
        -------
        PassbandGrid
            The passband interpolated onto the uniform grid of
            ``PassbandGrid.from_values`` and normalized to 1 at its center.
        """
        cache_key = (round(float(esa_voltage), 3), aperture)
        if cache_key in self._cached_passbands:
            return self._cached_passbands[cache_key]

        coefficients = (
            self._coefficients.sel(aperture=aperture)
            .dropna("energy_ratio", how="all")
            .dropna("elevation", how="all")
        )
        clamped_voltage = xr.full_like(coefficients.min_esa_voltage, esa_voltage).clip(
            coefficients.min_esa_voltage, coefficients.max_esa_voltage
        )

        log_beam_energy = np.log(SIMION_K_FACTOR * clamped_voltage)

        values = (
            xr.polyval(log_beam_energy, coefficients.reset_coords(drop=True), "power")
            .pipe(np.exp)
            .fillna(0)
            .reset_coords(drop=True)
            .assign_coords(speed_ratio=lambda x: np.sqrt(x.energy_ratio / SIMION_K_FACTOR))
            .swap_dims(energy_ratio="speed_ratio")
            .drop_vars("energy_ratio")
        )
        passband = PassbandGrid.from_values(values)
        self._cached_passbands[cache_key] = passband
        return passband
