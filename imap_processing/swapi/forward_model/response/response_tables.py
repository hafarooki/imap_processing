"""Placeholder: SWAPI's instrument response ancillary tables."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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
    Passband,
    PassbandFit,
)

ESA_VOLTAGE_UNITS = "V"

# Placeholder: number of sweeps whose passbands each SwapiResponse keeps cached
PASSBAND_CACHE_SIZE = 32


@dataclass(frozen=True, eq=False)
class SwapiResponse:
    """
    Placeholder: SWAPI's instrument response model, read from the ancillary tables.

    ESA steps are identified by their ESA voltage magnitude in V, as in the
    ancillary tables. ``SIMION_K_FACTOR`` converts a voltage to the central
    energy per charge of its passband only where an energy or speed is needed
    (:func:`~.step_response.step_responses` and the passband fits).

    Typical use: read the tables once, then build the responses of each chunk
    from them with :func:`~.step_response.step_responses`::

        response = SwapiResponse.from_files(...)
        responses = step_responses(
            response, chunk_esa_voltages, epoch, species, rtn_to_swapi
        )

    :meth:`sweep_passbands` caches the passbands of each sweep, so chunks that
    share ESA voltages reuse them.

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
    passband_fits : dict[Aperture, PassbandFit]
        Passband fit of each aperture.
    relative_efficiency : xarray.DataArray
        Relative efficiency (efficiency_epoch, ion) with ``efficiency_epoch``
        in TT J2000 ns, sorted.
    """

    azimuthal_transmission: AzimuthalTransmission
    central_effective_area: xr.DataArray
    passband_fits: dict[Aperture, PassbandFit]
    relative_efficiency: xr.DataArray
    _passbands_cache: OrderedDict[bytes, tuple[list[Passband], list[Passband]]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    @classmethod
    def from_files(
        cls,
        azimuthal_transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
        efficiency_path: Path,
    ) -> SwapiResponse:
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
        SwapiResponse
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

    def relative_efficiency_at(self, epoch: int, species: Species) -> float:
        """
        Placeholder: Look up a species' relative efficiency in effect at an epoch.

        From the L3a docs (``docs/swapi/response.md``, "Central Effective Area and
        Azimuthal Transmission"): "where epsilon_H(t) is the most recent entry whose
        timestamp precedes t. [...] It has also been assumed that He+ and He++ have the
        same efficiency based on theoretical expectations and preliminary analyses of
        SWAPI's measurements in space."

        Parameters
        ----------
        epoch : int
            Time in TT J2000 ns.
        species : Species
            Ion species.

        Returns
        -------
        float
            The efficiency of the latest table row strictly before ``epoch``.

        Raises
        ------
        ValueError
            If ``epoch`` is not after the table's first row.
        """
        efficiency = self.relative_efficiency
        i_row = (
            int(np.searchsorted(efficiency.efficiency_epoch.values, epoch, side="left"))
            - 1
        )
        if i_row < 0:
            raise ValueError(f"No relative efficiency in effect at epoch {epoch} ns.")
        ion = "H" if species is Species.PROTON else "He"
        return float(efficiency.isel(efficiency_epoch=i_row).sel(ion=ion))

    def central_effective_area_at(self, esa_voltage: float) -> float:
        """
        Placeholder: Interpolate the central effective area at an ESA voltage.

        From the L3a docs (``docs/swapi/response.md``, "Central Effective Area and
        Azimuthal Transmission"): "``SwapiResponse`` evaluates this curve by linear
        interpolation in |V| and uses the nearest tabulated value outside the CSV
        range."

        Parameters
        ----------
        esa_voltage : float
            ESA voltage magnitude in V.

        Returns
        -------
        float
            The area, interpolated linearly in voltage and held at the table's
            end values outside it; not scaled by the relative efficiency.
        """
        table = self.central_effective_area
        return float(np.interp(esa_voltage, table.esa_voltage.values, table.values))

    def passband(self, esa_voltage: float, aperture: Aperture) -> Passband:
        """
        Placeholder: Evaluate an aperture's passband at an ESA voltage.

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
        return self.passband_fits[aperture].evaluate(esa_voltage)

    def sweep_passbands(
        self, esa_voltages: npt.NDArray[np.float64]
    ) -> tuple[list[Passband], list[Passband]]:
        """
        Claude: Evaluate the passbands of each sample of a sweep, with caching.

        The passbands are shared between calls with the same voltages and must
        not be modified.

        Evaluating the passbands dominates building the step responses (~60 ms
        for a 5 x 72 chunk, against ~1.5 ms for the rest), and they depend on
        the voltages alone.

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
        cache = self._passbands_cache
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
) -> dict[Aperture, PassbandFit]:
    """
    Placeholder: Read the passband fit coefficients table.

    Parameters
    ----------
    passband_fit_coefficients_path : pathlib.Path
        The ``passband-fit-coefficients`` ancillary CSV.

    Returns
    -------
    dict[Aperture, PassbandFit]
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

    fits = {}
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
    return fits
