import pytest

from imap_processing.swapi.forward_model.response.response_tables import (
    SwapiResponse,
)


@pytest.fixture(scope="session")
def swapi_test_data_path(imap_tests_path):
    return imap_tests_path / "swapi/"


@pytest.fixture(scope="session")
def swapi_l0_test_data_path(swapi_test_data_path):
    return swapi_test_data_path / "l0_data/"


@pytest.fixture(scope="session")
def swapi_l0_validation_data_path(swapi_test_data_path):
    return swapi_test_data_path / "l0_validation_data/"


@pytest.fixture(scope="session")
def swapi_response(swapi_test_data_path):
    lut = swapi_test_data_path / "lut"
    return SwapiResponse.from_files(
        azimuthal_transmission_path=lut / "azimuthal-transmission.csv",
        central_effective_area_path=lut / "central-effective-area.csv",
        passband_fit_coefficients_path=lut / "passband-fit-coefficients.csv",
        efficiency_path=lut / "efficiency-lut.dat",
    )
