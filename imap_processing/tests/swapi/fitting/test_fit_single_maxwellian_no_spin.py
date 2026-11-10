import numpy as np
import pytest

from imap_processing.swapi.constants import Species
from imap_processing.swapi.fitting.fit_single_maxwellian_no_spin import (
    fit_single_maxwellian_no_spin,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    maxwellian_count_rates,
)

# realistic ESA energies in eV/q
ESA_ENERGIES = np.geomspace(100.0, 20000.0, 62)
N_SWEEPS = 5


@pytest.fixture
def chunk_responses(response_tables, epoch):
    energies = np.tile(ESA_ENERGIES, (N_SWEEPS, 1))
    passbands = response_tables.passbands(ESA_ENERGIES)
    return response_tables.step_responses(passbands, energies, epoch, Species.PROTON)


@pytest.mark.parametrize(
    "temperature", [pytest.param(20_000, id="cold"), pytest.param(200_000, id="hot")]
)
@pytest.mark.parametrize(
    "speed", [pytest.param(350, id="slow"), pytest.param(700, id="fast")]
)
def test_fit_single_maxwellian_no_spin(chunk_responses, speed, temperature):
    true_density = 5.0
    n_samples = N_SWEEPS * len(ESA_ENERGIES)
    # along the boresight, so every sweep sees the same rates
    velocity = np.broadcast_to(np.array([0.0, -float(speed), 0.0]), (n_samples, 3))
    ideal = maxwellian_count_rates(
        chunk_responses, true_density, float(temperature), velocity
    ).rates
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)

    fit = fit_single_maxwellian_no_spin(chunk_responses, count_rates)

    assert fit["density"] == pytest.approx(true_density, rel=0.01)
    assert fit["temperature"] == pytest.approx(temperature, rel=0.01)
    assert fit["speed"] == pytest.approx(speed, rel=0.01)
    assert fit["model_count_rates"] == pytest.approx(count_rates, rel=1e-3, abs=1e-3)


def test_fit_single_maxwellian_no_spin_ignores_invalid_rates(chunk_responses):
    n_samples = N_SWEEPS * len(ESA_ENERGIES)
    velocity = np.broadcast_to(np.array([0.0, -450.0, 0.0]), (n_samples, 3))
    ideal = maxwellian_count_rates(chunk_responses, 5.0, 100_000.0, velocity).rates
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)
    count_rates[0, ::3] = np.nan

    fit = fit_single_maxwellian_no_spin(chunk_responses, count_rates)

    assert fit["density"] == pytest.approx(5.0, rel=0.01)
    assert fit["temperature"] == pytest.approx(100_000.0, rel=0.01)
    assert fit["speed"] == pytest.approx(450.0, rel=0.01)


def test_fit_single_maxwellian_no_spin_checks_shapes(chunk_responses):
    with pytest.raises(ValueError, match="responses"):
        fit_single_maxwellian_no_spin(chunk_responses, np.ones((4, 62)))
