"""
Handling for the central effective area component of SWAPI's instrument response.

Central effective area ``Aeff(V)`` is defined as the ratio of count rate (C) to particle
flux ``nv`` for a uniform beam of energy ``k * |V|`` incident on SWAPI through the open
aperture at elevation angle ``theta = 0``, where ``k = 1.88`` is the SIMION-calibrated
k-factor, provided that SWAPI's ESA voltage setting is ``V``.

This module handles the ESA voltage-dependent central effective area from before launch,
as well as the time variability as determined through changes in the efficiency table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import Species


@dataclass(frozen=True, eq=False)
class CentralEffectiveArea:
    """
    Central effective area tables and interpolation.

    Handles storing and interpolating the central effective area-related tables: lab
    central effective area (as a function of ESA voltage) and relative efficiency (as a
    function of time).

    Attributes
    ----------
    lab_area : xarray.DataArray
        Lab-calibrated central effective area.
    relative_efficiency : xarray.DataArray
        Relative efficiency for hydrogen and helium.
    """

    lab_area: xr.DataArray
    relative_efficiency: xr.DataArray

    @classmethod
    def from_files(
        cls, central_effective_area_path: Path, efficiency_path: Path
    ) -> CentralEffectiveArea:
        """
        Load the central effective area and efficiency tables.

        Parameters
        ----------
        central_effective_area_path : pathlib.Path
            The ``central-effective-area`` ancillary file path.
        efficiency_path : pathlib.Path
            The ``efficiency-lut`` ancillary file path.

        Returns
        -------
        CentralEffectiveArea
            The central effective area.
        """
        return cls(
            lab_area=_read_central_effective_area(central_effective_area_path),
            relative_efficiency=_read_relative_efficiency(efficiency_path),
        )

    def evaluate_adjusted(
        self,
        esa_voltages: npt.ArrayLike,
        epochs: npt.ArrayLike,
        species: Species,
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the adjusted central effective area of each sample.

        The tabulated area is interpolated linearly in voltage and held at the
        table's end values outside it. It is then multiplied by the species' relative
        efficiency, taken from the most recent row of the efficiency table whose
        timestamp is at or before the sample time.

        The parameters may have any shape but their shapes must match.

        Parameters
        ----------
        esa_voltages : numpy.ndarray
            ESA voltage magnitude of each sample in V.
        epochs : numpy.ndarray
            Time of each sample in TT J2000 ns.
        species : Species
            Species being modeled.

        Returns
        -------
        numpy.ndarray
            The adjusted effective area, matching the input shape.

        Raises
        ------
        ValueError
            If ``esa_voltages`` and ``epochs`` have different shapes.
        KeyError
            If any epoch precedes the first row of the efficiency table.
        """
        esa_voltages = np.asarray(esa_voltages, dtype=float)
        epochs = np.asarray(epochs, dtype=np.int64)
        if esa_voltages.shape != epochs.shape:
            raise ValueError(
                f"esa_voltages.shape {esa_voltages.shape} does not match "
                f"epochs.shape {epochs.shape}."
            )

        element = "H" if species is Species.PROTON else "He"
        relative_efficiencies = self.relative_efficiency.sel(
            element=element,
            # Wrapped in a DataArray in case of multidimensional inputs
            efficiency_epoch=xr.DataArray(epochs),
            method="ffill",
        ).values

        lab_areas = self.lab_area.interp(
            # Wrapped in a DataArray in case of multidimensional inputs
            esa_voltage=xr.DataArray(esa_voltages),
            kwargs={
                "fill_value": (
                    self.lab_area.values[0],
                    self.lab_area.values[-1],
                )
            },
        ).values
        return np.asarray(lab_areas * relative_efficiencies, dtype=float)


def _read_central_effective_area(central_effective_area_path: Path) -> xr.DataArray:
    """Read the central effective area table."""
    table = pd.read_csv(central_effective_area_path)
    return xr.DataArray(
        table["effective_area"].to_numpy(dtype=float),
        dims=["esa_voltage"],
        coords={
            "esa_voltage": (
                "esa_voltage",
                table["esa_voltage"].to_numpy(dtype=float),
                {"units": "V"},
            )
        },
    )


def _read_relative_efficiency(efficiency_path: Path) -> xr.DataArray:
    """Read the relative efficiency table."""
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
        dims=["efficiency_epoch", "element"],
        coords={
            "efficiency_epoch": (
                "efficiency_epoch",
                np.asarray(epochs, dtype=np.int64),
                {"units": "ns"},
            ),
            "element": ["H", "He"],
        },
    )
