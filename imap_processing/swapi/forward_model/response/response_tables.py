"""Placeholder: SWAPI's instrument response ancillary tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import Species
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.passband import (
    Aperture,
    PassbandFit,
    PassbandFits,
)

ESA_VOLTAGE_UNITS = "V"


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
    central_effective_area : xarray.DataArray
        Effective area at elevation 0 along an ``esa_voltage`` coordinate in
        V.
    passband_fits : PassbandFits
        Passband fit of each aperture.
    relative_efficiency : xarray.DataArray
        Relative efficiency (efficiency_epoch, ion) with ``efficiency_epoch``
        in TT J2000 ns, sorted.
    """

    azimuthal_transmission: AzimuthalTransmission
    central_effective_area: xr.DataArray
    passband_fits: PassbandFits
    relative_efficiency: xr.DataArray

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
            central_effective_area=_read_central_effective_area(
                central_effective_area_path
            ),
            passband_fits=_read_passband_fits(passband_fit_coefficients_path),
            relative_efficiency=_read_relative_efficiency(efficiency_path),
        )

    def central_effective_area_at(
        self,
        esa_voltages: npt.ArrayLike,
        epochs: npt.ArrayLike,
        species: Species,
    ) -> npt.NDArray[np.float64]:
        """
        Placeholder: Evaluate the central effective area of each sample.

        The tabulated area is interpolated linearly in voltage and held at the
        table's end values outside it, as in the L3a docs (``docs/swapi/response.md``,
        "Central Effective Area and Azimuthal Transmission"): "``SwapiResponse``
        evaluates this curve by linear interpolation in |V| and uses the nearest
        tabulated value outside the CSV range."

        It is then scaled by the species' relative efficiency, interpolated
        linearly in time between the efficiency table's rows and held at the
        first or last row's value outside them. This deviates from the L3a docs,
        which take "the most recent entry whose timestamp precedes t". He+ and He++
        share the helium column.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitude of each sample in V.
        epochs : numpy.ndarray
            Time of each sample in TT J2000 ns, broadcastable against
            ``esa_voltages``.
        species : Species
            Ion species.

        Returns
        -------
        numpy.ndarray
            The effective area of each sample, with the broadcast shape of
            ``esa_voltages`` and ``epochs``.
        """
        efficiency = self.relative_efficiency
        row_epochs = efficiency.efficiency_epoch.values
        epochs = np.asarray(epochs, dtype=np.int64)
        ion = "H" if species is Species.PROTON else "He"
        # Placeholder: interpolate relative to the first row so the int64 ns
        # epochs keep their precision as float64
        relative_efficiencies = np.interp(
            (epochs - row_epochs[0]).astype(float),
            (row_epochs - row_epochs[0]).astype(float),
            efficiency.sel(ion=ion).values,
        )

        table = self.central_effective_area
        areas = np.interp(esa_voltages, table.esa_voltage.values, table.values)
        return np.asarray(areas * relative_efficiencies, dtype=float)


def _read_central_effective_area(central_effective_area_path: Path) -> xr.DataArray:
    """
    Placeholder: Read the central effective area table.

    Parameters
    ----------
    central_effective_area_path : pathlib.Path
        The ``central-effective-area`` ancillary CSV.

    Returns
    -------
    xarray.DataArray
        Effective area at elevation 0 along an ``esa_voltage`` coordinate in
        V.
    """
    table = pd.read_csv(central_effective_area_path)
    return xr.DataArray(
        table["effective_area"].to_numpy(dtype=float),
        dims=["esa_voltage"],
        coords={
            "esa_voltage": (
                "esa_voltage",
                table["esa_voltage"].to_numpy(dtype=float),
                {"units": ESA_VOLTAGE_UNITS},
            )
        },
    )


def _read_relative_efficiency(efficiency_path: Path) -> xr.DataArray:
    """
    Placeholder: Read the relative efficiency table.

    From the L3a docs (``docs/swapi/response.md``, "Central Effective Area and Azimuthal
    Transmission"): "The efficiency calibration table has two columns: epsilon_H for
    hydrogen and epsilon_He for helium."

    Parameters
    ----------
    efficiency_path : pathlib.Path
        The ``efficiency-lut`` ancillary file: whitespace-separated columns of
        UTC time, MET, H efficiency and He efficiency, with ``#`` comments.

    Returns
    -------
    xarray.DataArray
        Relative efficiency (efficiency_epoch, ion) with ``efficiency_epoch``
        in TT J2000 ns.
    """
    table = pd.read_csv(
        efficiency_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["utc", "met", "H", "He"],
    ).sort_values("utc")
    epochs = et_to_ttj2000ns(str_to_et(table["utc"].to_numpy(dtype=str)))
    return xr.DataArray(
        table[["H", "He"]].to_numpy(dtype=float),
        dims=["efficiency_epoch", "ion"],
        coords={
            "efficiency_epoch": (
                "efficiency_epoch",
                np.asarray(epochs, dtype=np.int64),
                {"units": "ns"},
            ),
            "ion": ["H", "He"],
        },
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
                {"units": ESA_VOLTAGE_UNITS},
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
