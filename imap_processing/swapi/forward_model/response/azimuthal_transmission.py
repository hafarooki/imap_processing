"""Placeholder: SWAPI's azimuthal transmission."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple

import numba
import numpy as np
import numpy.typing as npt

from imap_processing.swapi.constants import (
    OA_AZIMUTH_MAX_DEG,
    OA_PLATEAU_AZIMUTH_MAX_DEG,
    OA_PLATEAU_AZIMUTH_MIN_DEG,
    OA_PLATEAU_TRANSMISSION,
    SG_AZIMUTH_MAX_DEG,
    SG_PLATEAU_AZIMUTH_MAX_DEG,
    SG_PLATEAU_TRANSMISSION,
)


class Region(IntEnum):
    """
    Placeholder: Azimuthal region of the aperture, named by the sign of its azimuths.

    The sunglasses are centered on the boresight, flanked by the two halves of
    the open aperture.

    From the L3a docs (``docs/swapi/solar-wind-forward-model.md``, "Integral"): "The
    region sum includes the sunglasses (SG, |phi| <= 20 deg) and the two sides of the
    open aperture (OA+/-, 20 deg <= |phi| <= 150 deg)."
    """

    OPEN_APERTURE_NEGATIVE = -1
    SUNGLASSES = 0
    OPEN_APERTURE_POSITIVE = 1


class AzimuthalTransmission(NamedTuple):
    """
    Placeholder: Azimuthal transmission tabulated on a uniform grid starting at 0 deg.

    From the L3a docs (``docs/swapi/response.md``, "Central Effective Area and Azimuthal
    Transmission"): "T(phi) is tabulated as a function of |phi| from 0 deg to 180 deg at
    0.1 deg spacing."
    """

    values: npt.NDArray[np.float64]
    spacing: float

    @classmethod
    def from_table(
        cls, azimuths: npt.ArrayLike, transmission: npt.ArrayLike
    ) -> AzimuthalTransmission:
        """
        Placeholder: Build from a table, checking its grid and plateaus.

        The table must be on a uniform grid from 0 deg, and its values inside
        the sunglasses and open-aperture plateaus must equal the plateau
        transmissions that ``interpolate_azimuthal_transmission`` returns there.

        Parameters
        ----------
        azimuths : numpy.ndarray
            Absolute azimuths in deg.
        transmission : numpy.ndarray
            Transmission at ``azimuths``.

        Returns
        -------
        AzimuthalTransmission
            The transmission.

        Raises
        ------
        ValueError
            If ``azimuths`` is not ``0, spacing, 2 * spacing, ...``, if
            ``transmission`` does not match ``azimuths`` in size, or if the
            table disagrees with a plateau transmission.
        """
        azimuths = np.asarray(azimuths, dtype=float)
        if azimuths.size < 2:
            raise ValueError("Azimuthal transmission needs at least two azimuths.")
        spacing = (azimuths[-1] - azimuths[0]) / (azimuths.size - 1)
        if not np.allclose(
            azimuths, spacing * np.arange(azimuths.size), rtol=0, atol=1e-9
        ):
            raise ValueError(
                "Azimuthal transmission must be on a uniform grid starting at 0 deg."
            )
        values = np.ascontiguousarray(transmission, dtype=float)
        if values.shape != azimuths.shape:
            raise ValueError(
                "Azimuthal transmission must have one value per azimuth, got "
                f"{values.size} values for {azimuths.size} azimuths."
            )
        plateaus = (
            ("sunglasses", 0.0, SG_PLATEAU_AZIMUTH_MAX_DEG, SG_PLATEAU_TRANSMISSION),
            (
                "open-aperture",
                OA_PLATEAU_AZIMUTH_MIN_DEG,
                OA_PLATEAU_AZIMUTH_MAX_DEG,
                OA_PLATEAU_TRANSMISSION,
            ),
        )
        for name, lo, hi, plateau in plateaus:
            in_plateau = (azimuths >= lo - 1e-9) & (azimuths <= hi + 1e-9)
            if not np.allclose(values[in_plateau], plateau):
                raise ValueError(
                    f"Azimuthal transmission must be {plateau} on the {name} "
                    f"plateau ({lo} to {hi} deg)."
                )
        return cls(values=values, spacing=float(spacing))


@numba.njit
def interpolate_azimuthal_transmission(
    transmission: AzimuthalTransmission, azimuth: float
) -> float:
    """
    Placeholder: Interpolate the azimuthal transmission at an azimuth.

    From the L3a docs (``docs/swapi/response.md``, "Central Effective Area and Azimuthal
    Transmission"): "the interpolator uses |phi| after wrapping azimuth into [-180 deg,
    180 deg). The flat portions of the azimuth response are explicitly included in the
    interpolator for performance: T = 10^-3 for the central sunglasses region (|phi| <=
    9 deg), and T = 1 across the main open aperture range (31 deg <= |phi| <= 115 deg)."

    Parameters
    ----------
    transmission : AzimuthalTransmission
        The azimuthal transmission table.
    azimuth : float
        Look azimuth in deg.

    Returns
    -------
    float
        Transmission, the plateau values inside the plateaus and linear
        interpolation in ``abs(azimuth)`` elsewhere.
    """
    azimuth = (azimuth + 180) % 360 - 180
    abs_azimuth = abs(azimuth)

    if OA_PLATEAU_AZIMUTH_MIN_DEG <= abs_azimuth <= OA_PLATEAU_AZIMUTH_MAX_DEG:
        return OA_PLATEAU_TRANSMISSION
    if abs_azimuth <= SG_PLATEAU_AZIMUTH_MAX_DEG:
        return SG_PLATEAU_TRANSMISSION

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


@numba.njit
def azimuth_range(region: Region) -> tuple[float, float]:
    """
    Placeholder: Find the azimuths spanned by an azimuthal region.

    Parameters
    ----------
    region : Region
        Azimuthal region.

    Returns
    -------
    lower : float
        Lowest azimuth in deg.
    upper : float
        Highest azimuth in deg.
    """
    if region == Region.SUNGLASSES:
        return -SG_AZIMUTH_MAX_DEG, SG_AZIMUTH_MAX_DEG
    if region == Region.OPEN_APERTURE_POSITIVE:
        return SG_AZIMUTH_MAX_DEG, OA_AZIMUTH_MAX_DEG
    return -OA_AZIMUTH_MAX_DEG, -SG_AZIMUTH_MAX_DEG
