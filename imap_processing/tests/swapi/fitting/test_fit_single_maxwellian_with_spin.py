import numpy as np
import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.constants import Species
from imap_processing.swapi.fitting.fit_single_maxwellian_with_spin import (
    SPIN_PERIOD_S,
    SWEEP_PERIOD_S,
    fit_single_maxwellian_with_spin,
    spin_rates_and_jacobian,
    spin_rotations,
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


@pytest.fixture
def rotations():
    return spin_rotations(N_SWEEPS, len(ESA_ENERGIES)).reshape(-1, 3, 3)


def test_spin_rotations():
    rotations = spin_rotations(2, 72)
    assert rotations.shape == (2, 72, 3, 3)
    # proper rotations
    assert_allclose(
        rotations @ np.swapaxes(rotations, -1, -2),
        np.broadcast_to(np.eye(3), (2, 72, 3, 3)),
        atol=1e-12,
    )
    assert_allclose(np.linalg.det(rotations), 1.0)
    # angle 0 at the first step of the first sweep
    assert_allclose(rotations[0, 0], np.eye(3), atol=1e-12)
    # the spin axis, SWAPI +y, is fixed
    assert_allclose(
        rotations[..., :, 1], np.broadcast_to([0.0, 1.0, 0.0], (2, 72, 3)), atol=1e-12
    )

    # 15 bins (2.5 s) in, the spacecraft has spun 60 deg right-handed about +y,
    # so a flow along +x at spin angle 0 is seen by SWAPI rotated toward +z
    seconds = 15 * SWEEP_PERIOD_S / 72
    angle = 2 * np.pi * seconds / SPIN_PERIOD_S
    assert angle == pytest.approx(np.pi / 3)
    rotation = spin_rotations(1, 16)[0, 15]
    assert_allclose(
        rotation.T @ [1.0, 0.0, 0.0], [np.cos(angle), 0.0, np.sin(angle)], atol=1e-12
    )


def test_spin_rates_and_jacobian(chunk_responses, rotations):
    # far enough off the spin axis to modulate the rates strongly
    state = np.array([np.log(5.0), np.log(50_000.0), 30.0, -400.0, 60.0])
    rates, jacobian = spin_rates_and_jacobian(chunk_responses, rotations, state)
    assert rates.shape == (N_SWEEPS * len(ESA_ENERGIES),)
    assert jacobian.shape == (*rates.shape, 5)
    # the sweeps differ as the spacecraft spins
    rates_by_sweep = rates.reshape(N_SWEEPS, -1)
    assert not np.allclose(rates_by_sweep[0], rates_by_sweep[1], rtol=0.05)

    # jacobian against central differences of the state, to within the
    # kernel's own derivatives, which are off by up to ~0.5% of the peak
    for k, step in enumerate([1e-4, 1e-4, 1e-2, 1e-2, 1e-2]):
        delta = np.zeros(5)
        delta[k] = step
        upper, _ = spin_rates_and_jacobian(chunk_responses, rotations, state + delta)
        lower, _ = spin_rates_and_jacobian(chunk_responses, rotations, state - delta)
        numeric = (upper - lower) / (2 * step)
        assert jacobian[:, k] == pytest.approx(
            numeric, rel=1e-3, abs=1e-2 * np.abs(numeric).max()
        )


@pytest.mark.parametrize(
    "temperature", [pytest.param(20_000, id="cold"), pytest.param(200_000, id="hot")]
)
@pytest.mark.parametrize(
    "speed", [pytest.param(350, id="slow"), pytest.param(700, id="fast")]
)
@pytest.mark.parametrize(
    "deflection", [pytest.param(1.0, id="small"), pytest.param(5.0, id="large")]
)
def test_fit_single_maxwellian_with_spin(
    chunk_responses, rotations, speed, temperature, deflection
):
    true_density = 5.0
    # mostly along the boresight, so the flow is off the spin axis by a bit
    true_velocity = np.array([deflection, -float(speed), -deflection])
    state = np.array([np.log(true_density), np.log(temperature), *true_velocity])
    ideal = spin_rates_and_jacobian(chunk_responses, rotations, state)[0]
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)
    # the off-axis flow makes the sweeps differ as the spacecraft spins
    assert not np.allclose(count_rates[0], count_rates[1])

    fit = fit_single_maxwellian_with_spin(chunk_responses, count_rates)

    assert fit["density"] == pytest.approx(true_density, rel=0.05)
    assert fit["temperature"] == pytest.approx(temperature, rel=0.05)
    assert fit["speed"] == pytest.approx(np.linalg.norm(true_velocity), rel=0.01)
