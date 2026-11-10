"""SWAPI's instrument response ancillary tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import SIMION_K_FACTOR, Species
from imap_processing.swapi.forward_model.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.instrument_response import (
    InstrumentResponses,
    central_speed,
)
from imap_processing.swapi.forward_model.passband import (
    Aperture,
    Passband,
    PassbandFit,
    Passbands,
)

ESA_ENERGY_UNITS = "eV/q"


class SwapiPassbands(NamedTuple):
    """
    Both apertures' passbands at a set of ESA energies.

    Attributes
    ----------
    esa_energy : numpy.ndarray
        ESA energies per charge in eV/q, sorted and unique (k,).
    sunglasses : Passbands
        Passbands of the sunglasses at ``esa_energy``.
    open_aperture : Passbands
        Passbands of the open aperture at ``esa_energy``.
    """

    esa_energy: npt.NDArray[np.float64]
    sunglasses: Passbands
    open_aperture: Passbands

    def index_of(self, esa_energies: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """
        Find the passbands of ESA energies.

        Parameters
        ----------
        esa_energies : numpy.ndarray
            ESA energies per charge in eV/q, each exactly one of
            ``esa_energy``.

        Returns
        -------
        numpy.ndarray
            Index of each energy's passbands, with the shape of
            ``esa_energies``.

        Raises
        ------
        ValueError
            If an energy has no passband.
        """
        esa_energies = np.asarray(esa_energies, dtype=float)
        index = np.searchsorted(self.esa_energy, esa_energies)
        clipped = np.minimum(index, len(self.esa_energy) - 1)
        missing = self.esa_energy[clipped] != esa_energies
        if np.any(missing):
            raise ValueError(
                "No passband at ESA energies "
                f"{np.unique(esa_energies[missing]).tolist()} eV/q; include them "
                "in SwapiCalibration.passbands."
            )
        return clipped.astype(np.int64)


@dataclass(frozen=True, eq=False)
class SwapiCalibration:
    """
    SWAPI's instrument calibration, and the responses derived from it.

    ESA energies are per charge, in eV/q: ``SIMION_K_FACTOR`` times the ESA
    voltage. Tables given in ESA voltage are converted when read.

    Typical use: read the tables and evaluate the passbands at every ESA
    energy once, then build the responses of each chunk from them::

        calibration = SwapiCalibration.from_files(...)
        passbands = calibration.passbands(all_esa_energies)
        responses = calibration.responses(passbands, chunk_esa_energies, epoch, species)

    Attributes
    ----------
    azimuthal_transmission : AzimuthalTransmission
        Azimuthal transmission.
    central_effective_area : xarray.DataArray
        Effective area at elevation 0 along an ``esa_energy`` coordinate in
        eV/q.
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

    @classmethod
    def from_files(
        cls,
        transmission_path: Path,
        central_effective_area_path: Path,
        passband_fit_coefficients_path: Path,
        efficiency_path: Path,
    ) -> SwapiCalibration:
        """
        Load the ancillary response tables.

        Parameters
        ----------
        transmission_path : pathlib.Path
            The ``azimuthal-transmission`` ancillary CSV.
        central_effective_area_path : pathlib.Path
            The ``central-effective-area`` ancillary CSV.
        passband_fit_coefficients_path : pathlib.Path
            The ``passband-fit-coefficients`` ancillary CSV.
        efficiency_path : pathlib.Path
            The ``efficiency-lut`` ancillary file.

        Returns
        -------
        SwapiCalibration
            The tables.
        """
        return cls(
            azimuthal_transmission=_read_transmission(transmission_path),
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

    def central_effective_area_at(self, esa_energy: float) -> float:
        """
        Interpolate the central effective area at an ESA energy.

        Parameters
        ----------
        esa_energy : float
            ESA energy per charge in eV/q.

        Returns
        -------
        float
            The area, interpolated linearly in energy and held at the table's
            end values outside it; not scaled by the relative efficiency.
        """
        table = self.central_effective_area
        return float(np.interp(esa_energy, table.esa_energy.values, table.values))

    def passband(self, esa_energy: float, aperture: Aperture) -> Passband:
        """
        Evaluate an aperture's passband at an ESA energy.

        Parameters
        ----------
        esa_energy : float
            ESA energy per charge in eV/q; it is clamped to the fit's valid
            range.
        aperture : Aperture
            Aperture.

        Returns
        -------
        Passband
            The passband.
        """
        return self.passband_fits[aperture].evaluate(esa_energy)

    def passbands(self, esa_energies: npt.ArrayLike) -> SwapiPassbands:
        """
        Evaluate both apertures' passbands at each distinct ESA energy.

        Parameters
        ----------
        esa_energies : numpy.ndarray
            ESA energies per charge in eV/q, of any shape; repeats are
            evaluated once.

        Returns
        -------
        SwapiPassbands
            The passbands, by sorted unique energy.

        Raises
        ------
        ValueError
            If an energy is not positive and finite.
        """
        esa_energies = np.asarray(esa_energies, dtype=float)
        if not np.all(np.isfinite(esa_energies) & (esa_energies > 0)):
            raise ValueError("ESA energies must be positive and finite.")
        unique_energies = np.unique(esa_energies)
        return SwapiPassbands(
            esa_energy=unique_energies,
            sunglasses=Passbands.stack(
                [self.passband(e, Aperture.SUNGLASSES) for e in unique_energies]
            ),
            open_aperture=Passbands.stack(
                [self.passband(e, Aperture.OPEN_APERTURE) for e in unique_energies]
            ),
        )

    def responses(
        self,
        passbands: SwapiPassbands,
        esa_energies: npt.ArrayLike,
        epoch: int,
        species: Species,
    ) -> InstrumentResponses:
        """
        Compute SWAPI's response at each sample of a chunk for one species.

        Parameters
        ----------
        passbands : SwapiPassbands
            Passbands from :meth:`passbands`, including every energy in
            ``esa_energies``.
        esa_energies : numpy.ndarray
            ESA energy per charge of each sample in eV/q; flattened in C order.
        epoch : int
            Time in TT J2000 ns, for the relative efficiency lookup.
        species : Species
            Ion species.

        Returns
        -------
        InstrumentResponses
            The response of each sample, sharing ``passbands``. The central
            effective area is interpolated linearly in energy and held at the
            table's end values outside it, then scaled by the species' relative
            efficiency in effect at ``epoch``.
        """
        esa_energies = np.asarray(esa_energies, dtype=float).ravel()
        efficiency = self.relative_efficiency_at(epoch, species)
        table = self.central_effective_area
        return InstrumentResponses(
            mass_kg=species.mass_kg,
            central_speed=np.array(
                [central_speed(e, species) for e in esa_energies], dtype=float
            ),
            central_effective_area=np.interp(
                esa_energies, table.esa_energy.values, table.values
            )
            * efficiency,
            passband_index=passbands.index_of(esa_energies),
            sg_passbands=passbands.sunglasses,
            oa_passbands=passbands.open_aperture,
            azimuthal_transmission=self.azimuthal_transmission,
        )


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
        Effective area at elevation 0 along an ``esa_energy`` coordinate in
        eV/q, converted from the file's ESA voltages.
    """
    table = pd.read_csv(central_effective_area_path)
    return xr.DataArray(
        table["effective_area"].to_numpy(dtype=float),
        dims=["esa_energy"],
        coords={
            "esa_energy": (
                "esa_energy",
                SIMION_K_FACTOR * table["esa_voltage"].to_numpy(dtype=float),
                {"units": ESA_ENERGY_UNITS},
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
        and each fit's valid ESA energy range in eV/q (energy_ratio,
        elevation), converted from the file's ESA voltages.
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

        energy_bounds = {
            f"{bound}_esa_energy": (
                ("energy_ratio", "elevation"),
                SIMION_K_FACTOR
                * fit_grid[f"{bound}_esa_voltage"]
                .to_numpy(dtype=float)
                .reshape(grid_shape),
                {"units": ESA_ENERGY_UNITS},
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
                    **energy_bounds,
                },
            ),
        )
    return fits
