import numpy as np
import pytest

from imap_processing.swapi.constants import Species
from imap_processing.swapi.fitting.fit_dual_maxwellian_no_spin import (
    dual_rates_and_jacobian,
    fit_dual_maxwellian_no_spin,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor

# realistic ESA energies in eV/q
ESA_ENERGIES = np.geomspace(100.0, 20000.0, 62)
N_SWEEPS = 5


@pytest.fixture
def chunk_responses(response_tables, epoch):
    energies = np.tile(ESA_ENERGIES, (N_SWEEPS, 1))
    passbands = response_tables.passbands(ESA_ENERGIES)
    return response_tables.responses(passbands, energies, epoch, Species.PROTON)


def test_dual_rates_and_jacobian(chunk_responses):
    state = np.array(
        [np.log(5.0), np.log(50_000.0), np.log(400.0), np.log(0.2), np.log(5.0)]
    )
    rates, jacobian = dual_rates_and_jacobian(chunk_responses, state)

    # jacobian against central differences of the state. The kernel's analytic
    # derivatives hold the quadrature limits fixed, but the limits move with
    # the temperature and speed, sliding the nodes over kinks in the
    # piecewise-linear passband and transmission tables, so the discretized
    # rates differ from the analytic derivatives by a small fraction of the
    # rate (the analytic ones are the stable ones under quadrature
    # refinement). That is up to ~4% of the peak of the small suprathermal
    # temperature column, where the core dominates the rate, so the tolerance
    # also allows 0.1% of the peak rate.
    for k, step in enumerate([1e-4] * 5):
        delta = np.zeros(5)
        delta[k] = step
        upper, _ = dual_rates_and_jacobian(chunk_responses, state + delta)
        lower, _ = dual_rates_and_jacobian(chunk_responses, state - delta)
        numeric = (upper - lower) / (2 * step)
        assert jacobian[:, k] == pytest.approx(
            numeric,
            rel=1e-3,
            abs=max(1e-2 * np.abs(numeric).max(), 1e-3 * rates.max()),
        )


@pytest.mark.parametrize(
    (
        "speed",
        "core_density",
        "core_temperature",
        "supra_density",
        "supra_temperature",
    ),
    # typical solar wind: core temperature rising with speed (roughly the Lopez &
    # Freeman relation), suprathermal density 1-5% of the core's and temperature
    # 2-20x it (MAX_TEMPERATURE_RATIO)
    [
        pytest.param(450, 6.0, 80_000, 0.2, 400_000, id="nominal"),
        pytest.param(330, 15.0, 30_000, 0.3, 150_000, id="slow_dense"),
        pytest.param(700, 2.5, 250_000, 0.05, 1_500_000, id="fast_hot"),
        pytest.param(400, 5.0, 50_000, 0.05, 500_000, id="faint_suprathermal"),
        pytest.param(500, 4.0, 100_000, 0.2, 300_000, id="strong_suprathermal"),
        pytest.param(400, 5.0, 50_000, 0.25, 100_000, id="small_temperature_ratio"),
        pytest.param(450, 5.0, 70_000, 0.1, 1_200_000, id="large_temperature_ratio"),
        pytest.param(330, 20.0, 20_000, 0.4, 200_000, id="cold_dense_core"),
        pytest.param(600, 1.0, 180_000, 0.03, 1_000_000, id="tenuous"),
    ],
)
def test_fit_dual_maxwellian_no_spin(
    chunk_responses,
    speed,
    core_density,
    core_temperature,
    supra_density,
    supra_temperature,
):
    state = np.array(
        [
            np.log(core_density),
            np.log(core_temperature),
            np.log(speed),
            np.log(supra_density),
            np.log(supra_temperature / core_temperature),
        ]
    )
    # five identical sweeps since the populations are aligned with the spin axis
    ideal = dual_rates_and_jacobian(chunk_responses, state)[0]
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)

    fit = fit_dual_maxwellian_no_spin(chunk_responses, count_rates)

    assert fit["speed"] == pytest.approx(speed, rel=0.1)
    assert fit["core_density"] == pytest.approx(core_density, rel=0.1)
    assert fit["core_temperature"] == pytest.approx(core_temperature, rel=0.1)
    assert fit["suprathermal_density"] == pytest.approx(supra_density, rel=0.1)
    assert fit["suprathermal_temperature"] == pytest.approx(supra_temperature, rel=0.1)
    assert fit["density"] == pytest.approx(core_density + supra_density, rel=0.1)
