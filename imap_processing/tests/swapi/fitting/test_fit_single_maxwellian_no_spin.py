import numpy as np
import pytest

from imap_processing.swapi.constants import SIMION_K_FACTOR, Species
from imap_processing.swapi.fitting.fit_single_maxwellian_no_spin import (
    fit_single_maxwellian_no_spin,
    mean_sweep,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor
from imap_processing.swapi.forward_model.maxwellian_count_rates import (
    maxwellian_count_rates,
)

# realistic ESA voltages in V, for central energies of 100 eV/q to 20 keV/q
ESA_VOLTAGES = np.geomspace(100.0, 20000.0, 62) / SIMION_K_FACTOR
N_SWEEPS = 5


CHUNK_VOLTAGES = np.tile(ESA_VOLTAGES, (N_SWEEPS, 1))


@pytest.fixture
def chunk_responses(response_tables, epoch):
    return response_tables.step_responses(CHUNK_VOLTAGES, epoch, Species.PROTON)


@pytest.mark.parametrize(
    "temperature", [pytest.param(20_000, id="cold"), pytest.param(200_000, id="hot")]
)
@pytest.mark.parametrize(
    "speed", [pytest.param(350, id="slow"), pytest.param(700, id="fast")]
)
def test_fit_single_maxwellian_no_spin(
    response_tables, epoch, chunk_responses, speed, temperature
):
    true_density = 5.0
    n_samples = N_SWEEPS * len(ESA_VOLTAGES)
    # along the boresight, so every sweep sees the same rates
    velocity = np.broadcast_to(np.array([0.0, -float(speed), 0.0]), (n_samples, 3))
    ideal = maxwellian_count_rates(
        chunk_responses, true_density, float(temperature), velocity
    ).rates
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)

    fit = fit_single_maxwellian_no_spin(
        CHUNK_VOLTAGES, count_rates, swapi_response=response_tables, epoch=epoch
    )

    assert fit["density"] == pytest.approx(true_density, rel=0.01)
    assert fit["temperature"] == pytest.approx(temperature, rel=0.01)
    assert fit["speed"] == pytest.approx(speed, rel=0.01)
    assert fit["model_count_rates"] == pytest.approx(count_rates, rel=1e-3, abs=1e-3)


def test_fit_single_maxwellian_no_spin_ignores_invalid_rates(
    response_tables, epoch, chunk_responses
):
    n_samples = N_SWEEPS * len(ESA_VOLTAGES)
    velocity = np.broadcast_to(np.array([0.0, -450.0, 0.0]), (n_samples, 3))
    ideal = maxwellian_count_rates(chunk_responses, 5.0, 100_000.0, velocity).rates
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)
    count_rates[0, ::3] = np.nan
    # a step with no valid rate in any sweep is dropped
    count_rates[:, 1] = np.nan

    fit = fit_single_maxwellian_no_spin(
        CHUNK_VOLTAGES, count_rates, swapi_response=response_tables, epoch=epoch
    )

    assert fit["density"] == pytest.approx(5.0, rel=0.01)
    assert fit["temperature"] == pytest.approx(100_000.0, rel=0.01)
    assert fit["speed"] == pytest.approx(450.0, rel=0.01)
    # the model is still reported at every sample, including invalid ones
    assert fit["model_count_rates"].shape == count_rates.shape
    assert np.all(np.isfinite(fit["model_count_rates"]))


def test_fit_single_maxwellian_no_spin_checks_shapes(response_tables, epoch):
    with pytest.raises(ValueError, match="same"):
        fit_single_maxwellian_no_spin(
            CHUNK_VOLTAGES,
            np.ones((4, 62)),
            swapi_response=response_tables,
            epoch=epoch,
        )


def test_fit_single_maxwellian_no_spin_checks_sweep_voltages(response_tables, epoch):
    voltages = CHUNK_VOLTAGES.copy()
    voltages[1] *= 1.01
    with pytest.raises(ValueError, match="same ESA voltages"):
        fit_single_maxwellian_no_spin(
            voltages,
            np.ones(voltages.shape),
            swapi_response=response_tables,
            epoch=epoch,
        )


def test_mean_sweep():
    voltages = np.tile([1.0, 2.0, 3.0], (2, 1))
    rates = np.array([[1.0, np.nan, np.nan], [3.0, 4.0, np.nan]])

    mean_voltages, mean_rates = mean_sweep(voltages, rates)

    np.testing.assert_array_equal(mean_voltages, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(mean_rates, [[2.0, 4.0, np.nan]])
