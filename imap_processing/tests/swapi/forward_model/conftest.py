import pytest

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.forward_model.response_tables import SwapiCalibration


@pytest.fixture(scope="session")
def response_table_paths(swapi_test_data_path):
    lut = swapi_test_data_path / "lut"
    return {
        "transmission_path": lut / "azimuthal-transmission.csv",
        "central_effective_area_path": lut / "central-effective-area.csv",
        "passband_fit_coefficients_path": lut / "passband-fit-coefficients.csv",
        "efficiency_path": lut / "efficiency-lut.dat",
    }


@pytest.fixture(scope="session")
def response_tables(response_table_paths):
    return SwapiCalibration.from_files(**response_table_paths)


@pytest.fixture
def epoch():
    return int(et_to_ttj2000ns(str_to_et("2026-01-01T00:00:00")))
