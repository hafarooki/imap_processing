"""SWAPI instrument response fixtures."""

import pytest

from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
)
from imap_processing.swapi.forward_model.response.central_effective_area import (
    CentralEffectiveAreaTable,
)
from imap_processing.swapi.forward_model.response.passband_fit import PassbandFitTable


@pytest.fixture(scope="session")
def azimuthal_transmission(imap_tests_path):
    return AzimuthalTransmission.from_csv(
        imap_tests_path / "swapi/lut/azimuthal-transmission.csv"
    )


@pytest.fixture(scope="session")
def central_effective_area_table(imap_tests_path):
    return CentralEffectiveAreaTable(
        imap_tests_path / "swapi/lut/central-effective-area.csv",
        imap_tests_path / "swapi/lut/efficiency-lut.dat",
    )


@pytest.fixture(scope="session")
def passband_fit_table(imap_tests_path):
    return PassbandFitTable(imap_tests_path / "swapi/lut/passband-fit-coefficients.csv")
