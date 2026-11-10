"""SWAPI's instrument response ancillary tables."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import Species
from imap_processing.swapi.forward_model.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.instrument_response import (
    StepResponses,
    central_speed,
)
from imap_processing.swapi.forward_model.passband import (
    Aperture,
    Passband,
    PassbandFit,
    Passbands,
)

ESA_VOLTAGE_UNITS = "V"

# number of distinct step responses each SwapiResponse keeps cached
STEP_RESPONSE_CACHE_SIZE = 32


class SwapiPassbands(NamedTuple):
    """
    Both apertures' passbands at a set of ESA voltages.

    Attributes
    ----------
    esa_voltage : numpy.ndarray
        ESA voltage magnitudes in V, sorted and unique (k,).
    sunglasses : Passbands
        Passbands of the sunglasses at ``esa_voltage``.
    open_aperture : Passbands
        Passbands of the open aperture at ``esa_voltage``.
    """

    esa_voltage: npt.NDArray[np.float64]
    sunglasses: Passbands
    open_aperture: Passbands

    def index_of(self, esa_voltages: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """
        Find the passbands of ESA voltages.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitudes in V, each exactly one of ``esa_voltage``.

        Returns
        -------
        numpy.ndarray
            Index of each voltage's passbands, with the shape of
            ``esa_voltages``.

        Raises
        ------
        ValueError
            If a voltage has no passband.
        """
        esa_voltages = np.asarray(esa_voltages, dtype=float)
        index = np.searchsorted(self.esa_voltage, esa_voltages)
        clipped = np.minimum(index, len(self.esa_voltage) - 1)
        missing = self.esa_voltage[clipped] != esa_voltages
        if np.any(missing):
            raise ValueError(
                "No passband at ESA voltages "
                f"{np.unique(esa_voltages[missing]).tolist()} V; include them "
                "in SwapiResponse.passbands."
            )
        return clipped.astype(np.int64)


@dataclass(frozen=True, eq=False)
class SwapiResponse:
    """
    SWAPI's instrument response model, read from the ancillary tables.

    ESA steps are identified by their ESA voltage magnitude in V, as in the
    ancillary tables. ``SIMION_K_FACTOR`` converts a voltage to the central
    energy per charge of its passband only where an energy or speed is needed
    (:func:`~imap_processing.swapi.forward_model.instrument_response.central_speed`
    and the passband fits).

    Typical use: read the tables once, then build the responses of each chunk
    from them::

        response = SwapiResponse.from_files(...)
        responses = response.step_responses(chunk_esa_voltages, epoch, species)

    :meth:`step_responses` caches its results, so chunks that share ESA
    voltages and relative efficiency reuse the same responses.

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
    _step_responses_cache: OrderedDict[tuple[Species, float, bytes], StepResponses] = (
        field(default_factory=OrderedDict, init=False, repr=False)
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
        Load the ancillary response tables.

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
            azimuthal_transmission=_read_transmission(azimuthal_transmission_path),
            central_effective_area=_read_central_effective_area(
                central_effective_area_path
            ),
            passband_fits=_read_passband_fits(passband_fit_coefficients_path),
            relative_efficiency=_read_relative_efficiency(efficiency_path),
        )

    def relative_efficiency_at(self, epoch: int, species: Species) -> float:
        """
        Look up a species' relative efficiency in effect at an epoch.

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
        Interpolate the central effective area at an ESA voltage.

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
        Evaluate an aperture's passband at an ESA voltage.

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

    def passbands(self, esa_voltages: npt.ArrayLike) -> SwapiPassbands:
        """
        Evaluate both apertures' passbands at each distinct ESA voltage.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitudes in V, of any shape; repeats are evaluated
            once.

        Returns
        -------
        SwapiPassbands
            The passbands, by sorted unique voltage.

        Raises
        ------
        ValueError
            If a voltage is not positive and finite.
        """
        esa_voltages = np.asarray(esa_voltages, dtype=float)
        if not np.all(np.isfinite(esa_voltages) & (esa_voltages > 0)):
            raise ValueError("ESA voltages must be positive and finite.")
        unique_voltages = np.unique(esa_voltages)
        return SwapiPassbands(
            esa_voltage=unique_voltages,
            sunglasses=Passbands.stack(
                [self.passband(v, Aperture.SUNGLASSES) for v in unique_voltages]
            ),
            open_aperture=Passbands.stack(
                [self.passband(v, Aperture.OPEN_APERTURE) for v in unique_voltages]
            ),
        )

    def step_responses(
        self,
        esa_voltages: npt.ArrayLike,
        epoch: int,
        species: Species,
    ) -> StepResponses:
        """
        Compute SWAPI's response at each sample of a chunk for one species.

        Results are cached by species, relative efficiency and ESA voltages,
        so repeated calls for chunks with the same sweep (e.g. consecutive
        chunks) don't re-evaluate the passbands. The returned responses are
        shared between calls and must not be modified.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitude of each sample in V; flattened in C order.
        epoch : int
            Time in TT J2000 ns, for the relative efficiency lookup.
        species : Species
            Ion species.

        Returns
        -------
        StepResponses
            The response of each sample. The central effective area is
            interpolated linearly in voltage and held at the table's end values
            outside it, then scaled by the species' relative efficiency in
            effect at ``epoch``.

        Raises
        ------
        ValueError
            If a voltage is not positive and finite.
        """
        esa_voltages = np.ascontiguousarray(esa_voltages, dtype=float).ravel()
        efficiency = self.relative_efficiency_at(epoch, species)
        key = (species, efficiency, esa_voltages.tobytes())
        cache = self._step_responses_cache
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        passbands = self.passbands(esa_voltages)
        table = self.central_effective_area
        responses = StepResponses(
            mass_kg=species.mass_kg,
            central_speed=np.array(
                [central_speed(v, species) for v in esa_voltages], dtype=float
            ),
            central_effective_area=np.interp(
                esa_voltages, table.esa_voltage.values, table.values
            )
            * efficiency,
            passband_index=passbands.index_of(esa_voltages),
            sg_passbands=passbands.sunglasses,
            oa_passbands=passbands.open_aperture,
            azimuthal_transmission=self.azimuthal_transmission,
        )
        cache[key] = responses
        if len(cache) > STEP_RESPONSE_CACHE_SIZE:
            cache.popitem(last=False)
        return responses


def _read_transmission(transmission_path: Path) -> AzimuthalTransmission:
    """
    Read the azimuthal transmission table.

    The input CSV must have format `(abs_azimuth, transmission)`.
    NaNs and missing values will be treated as 0.

    Parameters
    ----------
    transmission_path : pathlib.Path
        The ``azimuthal-transmission`` ancillary CSV.

    Returns
    -------
    AzimuthalTransmission
        The transmission.
    """
    table = pd.read_csv(transmission_path)
    return AzimuthalTransmission.from_table(
        table["abs_azimuth"].to_numpy(dtype=float),
        table["transmission"].fillna(0).to_numpy(dtype=float),
    )


def _read_central_effective_area(central_effective_area_path: Path) -> xr.DataArray:
    """
    Read the central effective area table.

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
    Read the relative efficiency table.

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
    Read the passband fit coefficients table.

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
        # fill the full (energy_ratio, elevation) grid, NaN where there is no
        # fit, then put power first
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
