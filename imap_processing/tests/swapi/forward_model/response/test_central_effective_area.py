import pytest

from imap_processing.spice.time import et_to_ttj2000ns, str_to_et
from imap_processing.swapi.constants import Species
from imap_processing.swapi.forward_model.response.central_effective_area import (
    CentralEffectiveAreaTable,
)


def test_central_effective_area(swapi_test_data_path):
    table = CentralEffectiveAreaTable(
        central_effective_area_path=swapi_test_data_path
        / "lut/central-effective-area.csv",
        efficiency_path=swapi_test_data_path / "lut/efficiency-lut.dat",
    )
    epoch = et_to_ttj2000ns(str_to_et("2026-01-01"))

    proton_1k = table.evaluate(1000 / 1.89, epoch, Species.PROTON)
    assert proton_1k == pytest.approx(0.38, rel=0.5 / 100)

    alpha_1k_per_q = table.evaluate(1000 / 1.89, epoch, Species.ALPHA)
    assert alpha_1k_per_q == pytest.approx(0.38 * 1.05, rel=0.5 / 100)

    min_tabulated_voltage = 53.18792553
    at_min = table.evaluate(min_tabulated_voltage, epoch, Species.PROTON)
    below_min = table.evaluate(min_tabulated_voltage / 2, epoch, Species.PROTON)
    assert below_min == pytest.approx(at_min)

    max_tabulated_voltage = 10512.34043
    at_max = table.evaluate(max_tabulated_voltage, epoch, Species.PROTON)
    above_max = table.evaluate(max_tabulated_voltage * 2, epoch, Species.PROTON)
    assert above_max == pytest.approx(at_max)
