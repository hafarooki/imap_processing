"""Placeholder: Shared helpers for the Maxwellian count-rate tests, ported from L3a's
``tests/swapi/_helpers.py``."""

from typing import NamedTuple

import numpy as np
import pytest

from imap_processing.swapi.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    PASSBAND_ELEVATIONS_DEG,
    PASSBAND_SPEED_RATIOS,
    PROTON_MASS_KG,
    SIMION_K_FACTOR,
    Species,
)
from imap_processing.swapi.forward_model.maxwellian.integration import (
    temperature_to_thermal_speed,
)
from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.instrument_response import (
    StepResponse,
)
from imap_processing.swapi.forward_model.response.passband import (
    Passband,
    _span_above,
)

# Placeholder: A 1 keV proton ESA voltage gives a comfortably typical central speed
# (~437 km/s) and a passband elevation range that contains 0.
ESA_VOLTAGE_1KEV = 1000.0 / SIMION_K_FACTOR

# Placeholder: Bulk flow straight into the boresight (look azimuth 0 deg, elevation 0
# deg)
BORESIGHT_VELOCITY_XYZ = (0.0, -450.0, 0.0)


def thermal_speed_to_temperature(thermal_speed: float, mass: float) -> float:
    """Placeholder: Convert a thermal speed sqrt(kT/m) (km/s) to a temperature (K)."""
    return mass * (thermal_speed * 1000.0) ** 2 / BOLTZMANN_CONSTANT_J_PER_K


class ProtonParams(NamedTuple):
    """Placeholder: Proton moments, with the bulk and thermal speeds they imply."""

    density: float
    velocity_xyz: np.ndarray
    temperature: float

    @property
    def bulk_speed(self) -> float:
        return float(np.linalg.norm(self.velocity_xyz))

    @property
    def thermal_speed(self) -> float:
        return temperature_to_thermal_speed(PROTON_MASS_KG, self.temperature)


def proton_params(
    *,
    density: float = 5.0,
    velocity_xyz=BORESIGHT_VELOCITY_XYZ,
    temperature: float = 100_000.0,
) -> ProtonParams:
    """Placeholder: Solar-wind proton moments at typical slow-wind values, in the SWAPI
    frame."""
    return ProtonParams(
        density=density,
        velocity_xyz=np.asarray(velocity_xyz, dtype=float),
        temperature=temperature,
    )


def gaussian_passband(
    sigma_elevation: float = 2.0, sigma_speed_ratio: float = 0.026
) -> Passband:
    """Placeholder: Synthetic passband: a Gaussian bump at elevation 0, speed ratio 1,
    on the standard grid, normalized to 1 at its peak."""
    elevation = PASSBAND_ELEVATIONS_DEG[:, np.newaxis]
    speed_ratio = PASSBAND_SPEED_RATIOS[np.newaxis, :]
    values = np.exp(
        -(elevation**2) / (2 * sigma_elevation**2)
        - (speed_ratio - 1) ** 2 / (2 * sigma_speed_ratio**2)
    )
    cutoff = 1e-2 * values.max()
    spans = np.array(
        [_span_above(PASSBAND_SPEED_RATIOS, row, cutoff) for row in values]
    )
    return Passband(
        min_elevation=float(PASSBAND_ELEVATIONS_DEG[0]),
        elevation_spacing=float(
            PASSBAND_ELEVATIONS_DEG[1] - PASSBAND_ELEVATIONS_DEG[0]
        ),
        min_speed_ratio=float(PASSBAND_SPEED_RATIOS[0]),
        speed_ratio_spacing=float(PASSBAND_SPEED_RATIOS[1] - PASSBAND_SPEED_RATIOS[0]),
        values=np.ascontiguousarray(values),
        min_boundary=spans[:, 0],
        max_boundary=spans[:, 1],
        elevation_range=_span_above(
            PASSBAND_ELEVATIONS_DEG, values.max(axis=1), cutoff
        ),
    )


def synthetic_step_response(
    central_speed: float = 450.0,
    central_effective_area: float = 0.5,
    azimuthal_transmission: np.ndarray | None = None,
    azimuthal_transmission_spacing: float = 1.0,
) -> StepResponse:
    """Placeholder: Proton step response with Gaussian passbands for both apertures."""
    if azimuthal_transmission is None:
        # Placeholder: T(|az|) = 1 sampled on [0 deg, 180 deg] at 1 deg spacing
        azimuthal_transmission = np.ones(181)
    passband = gaussian_passband()
    return StepResponse(
        mass_kg=PROTON_MASS_KG,
        central_speed=central_speed,
        central_effective_area=central_effective_area,
        sg_passband=passband,
        oa_passband=passband,
        azimuthal_transmission=AzimuthalTransmission(
            values=azimuthal_transmission,
            spacing=azimuthal_transmission_spacing,
        ),
    )


@pytest.fixture
def proton_step_responses(swapi_response, epoch):
    """
    Placeholder: Return a function giving the real proton responses at ESA voltages.
    """

    def step_responses(esa_voltages):
        return swapi_response.step_responses(
            np.asarray(esa_voltages, dtype=float), epoch, Species.PROTON
        )

    return step_responses
