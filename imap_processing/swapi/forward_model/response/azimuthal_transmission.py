"""Handling for the azimuthal transmission component of SWAPI's instrument response."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd


class AzimuthalRegion(Enum):
    """
    Azimuthal segment of the aperture.

    Includes the sunglasses (SG, |az| <= 20 deg) and the two sides of the
    open aperture (OA+/-, 20 deg <= |az| <= 150 deg). The values are ``(lower, upper)``
    azimuth ranges in degrees.
    """

    OPEN_APERTURE_NEGATIVE = (-150.0, -20.0)
    SUNGLASSES = (-20.0, 20.0)
    OPEN_APERTURE_POSITIVE = (20.0, 150.0)


class AzimuthalTransmission(NamedTuple):
    """
    Azimuthal transmission tabulated on a uniform grid starting at 0 deg.

    In the ancillary file, T(phi) is tabulated as a function of |phi| from 0 deg to 180
    deg at 0.1 deg spacing.
    """

    values: npt.NDArray[np.float64]
    spacing: float

    @classmethod
    def from_csv(cls, path: Path) -> AzimuthalTransmission:
        """
        Parse ancillary file.

        Parameters
        ----------
        path : pathlib.Path
            The ``azimuthal-transmission`` ancillary file path.

        Returns
        -------
        AzimuthalTransmission
            The azimuthal transmission.

        Raises
        ------
        ValueError
            If the values in the CSV do not follow the assumptions made for interpolation.
        """
        table = pd.read_csv(path)
        azimuths = table["abs_azimuth"].to_numpy(dtype=float)
        values = np.ascontiguousarray(table["transmission"].fillna(0), dtype=float)
        if azimuths.size < 2:
            raise ValueError("Azimuthal transmission needs at least two azimuths.")
        spacing = (azimuths[-1] - azimuths[0]) / (azimuths.size - 1)
        if not np.allclose(
            azimuths, spacing * np.arange(azimuths.size), rtol=0, atol=1e-9
        ):
            raise ValueError(
                "Azimuthal transmission must be on a uniform grid starting at 0 deg."
            )
        plateaus = (
            ("sunglasses", 0.0, 9.0, 1e-3),
            ("open-aperture", 31.0, 115.0, 1.0),
        )
        for name, lo, hi, value in plateaus:
            in_plateau = (azimuths >= lo - 1e-9) & (azimuths <= hi + 1e-9)
            if not np.allclose(values[in_plateau], value):
                raise ValueError(
                    f"Azimuthal transmission must be {value} on the {name} "
                    f"plateau ({lo} to {hi} deg)."
                )
        return cls(values=values, spacing=float(spacing))


@numba.njit
def interpolate_azimuthal_transmission(
    transmission: AzimuthalTransmission, azimuth: float
) -> float:
    """
    Interpolate the azimuthal transmission function.

    The interpolator uses |az| after wrapping to [-180 deg, 180 deg). The flat portions
    of the azimuth response are explicitly included in the interpolator for performance:
    1/1000 for the central sunglasses region (|phi| <= 9 deg), and 100% across the
    main open aperture range (31 deg <= |phi| <= 115 deg).

    Parameters
    ----------
    transmission : AzimuthalTransmission
        The azimuthal transmission table.
    azimuth : float
        Look azimuth in deg.

    Returns
    -------
    float
        Azimuthal transmission value.
    """
    azimuth = (azimuth + 180) % 360 - 180
    abs_azimuth = abs(azimuth)

    if 31.0 <= abs_azimuth <= 115.0:
        return 1.0
    if abs_azimuth <= 9.0:
        return 1e-3

    i_float = abs_azimuth / transmission.spacing
    i_lower = math.floor(i_float)
    i_upper = i_lower + 1

    n = len(transmission.values)
    i_lower = min(i_lower, n - 1)
    i_upper = min(i_upper, n - 1)

    weight_lower = float(i_upper) - i_float
    weight_upper = i_float - float(i_lower)
    return (
        transmission.values[i_lower] * weight_lower
        + transmission.values[i_upper] * weight_upper
    )
