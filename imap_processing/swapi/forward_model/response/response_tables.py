"""Placeholder: SWAPI's instrument response ancillary tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.effective_area import (
    CentralEffectiveArea,
)
from imap_processing.swapi.forward_model.response.passband import (
    Aperture,
    PassbandFit,
    PassbandFits,
)


@dataclass(frozen=True, eq=False)
class SwapiResponseTables:
    """
    Placeholder: SWAPI's instrument response model, read from the ancillary tables.

    ESA steps are identified by their ESA voltage magnitude in V, as in the
    ancillary tables. ``SIMION_K_FACTOR`` converts a voltage to the central
    energy per charge of its passband only where an energy or speed is needed
    (:func:`~.measurement_species_response.measurement_species_responses` and the
    passband fits).

    Typical use: read the tables once, then build the responses of each chunk
    from them with
    :func:`~.measurement_species_response.measurement_species_responses`::

        tables = SwapiResponseTables.from_files(...)
        responses = measurement_species_responses(
            tables, chunk_esa_voltages, chunk_epochs, species, chunk_rtn_to_swapi
        )

    :meth:`.PassbandFits.sweep_passbands` caches the passbands of each sweep, so
    chunks that share ESA voltages reuse them.

    From the L3a docs (``docs/swapi/response.md``, "Effective Area Function"): "The
    production code loads them from ancillary files. They are processed and cached in
    the ``SwapiResponse`` class."

    Attributes
    ----------
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission.
    central_effective_area : CentralEffectiveArea
        Central effective area and its time-variable relative efficiency.
    passband_fits : PassbandFits
        Passband fit of each aperture.
    """

    azimuthal_transmission: AzimuthalTransmission
    central_effective_area: CentralEffectiveArea
    passband_fits: PassbandFits

    @classmethod
    def from_files(
        cls,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
        efficiency_path: Path,
    ) -> SwapiResponseTables:
        """
        Placeholder: Load the ancillary response tables.

        Parameters
        ----------
        azimuthal_transmission_path : pathlib.Path
            The ``azimuthal-transmission`` ancillary CSV.
        central_effective_area_path : pathlib.Path
            The ``central-effective-area`` ancillary CSV.
        passband_fit_coefficients_path : pathlib.Path
            The ``passband-fit-coefficients`` ancillary CSV.
        efficiency_path : pathlib.Path
            The ``efficiency-lut`` ancillary file.

        Returns
        -------
        SwapiResponseTables
            The tables.
        """
        return cls(
            azimuthal_transmission=AzimuthalTransmission.from_csv(
                azimuthal_transmission_path
            ),
            central_effective_area=CentralEffectiveArea.from_files(
                central_effective_area_path, efficiency_path
            ),
            passband_fits=_read_passband_fits(passband_fit_coefficients_path),
        )


def _read_passband_fits(
    passband_fit_coefficients_path: Path,
) -> PassbandFits:
    """
    Placeholder: Read the passband fit coefficients table.

    Parameters
    ----------
    passband_fit_coefficients_path : pathlib.Path
        The ``passband-fit-coefficients`` ancillary CSV.

    Returns
    -------
    PassbandFits
        Fit of each aperture, with coefficients (power, energy_ratio,
        elevation) on the grid shared by all apertures, highest power first,
        and each fit's valid ESA voltage range in V (energy_ratio,
        elevation).
    """
    fit_table = pd.read_csv(passband_fit_coefficients_path)
    fit_table["region"] = fit_table["region"].astype(str)
    energy_ratios = np.unique(fit_table["energy_ratio"].to_numpy(dtype=float))
    elevations = np.unique(fit_table["elevation"].to_numpy(dtype=float))
    powers = sorted(
        (int(name) for name in fit_table.columns if name.isdigit()), reverse=True
    )

    fits: dict[Aperture, PassbandFit] = {}
    for aperture in Aperture:
        aperture_table = fit_table[fit_table["region"] == aperture.value]
        if aperture_table.empty:
            raise ValueError(f"No passband fit for aperture {aperture.value}.")
        # Placeholder: fill the full (energy_ratio, elevation) grid, NaN where there is
        # no fit, then put power first
        fit_grid = aperture_table.set_index(["energy_ratio", "elevation"]).reindex(
            pd.MultiIndex.from_product([energy_ratios, elevations])
        )
        grid_shape = (len(energy_ratios), len(elevations))
        coefficient_values = (
            fit_grid[[str(power) for power in powers]]
            .to_numpy(dtype=float)
            .reshape(*grid_shape, len(powers))
            .transpose(2, 0, 1)
        )

        voltage_bounds = {
            f"{bound}_esa_voltage": (
                ("energy_ratio", "elevation"),
                fit_grid[f"{bound}_esa_voltage"]
                .to_numpy(dtype=float)
                .reshape(grid_shape),
                {"units": "V"},
            )
            for bound in ("min", "max")
        }
        fits[aperture] = PassbandFit(
            coefficients=xr.DataArray(
                coefficient_values,
                dims=["power", "energy_ratio", "elevation"],
                coords={
                    "power": powers,
                    "energy_ratio": ("energy_ratio", energy_ratios, {"units": "eV/V"}),
                    "elevation": ("elevation", elevations, {"units": "deg"}),
                    **voltage_bounds,
                },
            ),
        )
    return PassbandFits(fits)
