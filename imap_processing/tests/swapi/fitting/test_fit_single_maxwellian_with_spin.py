from unittest import mock

import numpy as np
import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.constants import SIMION_K_FACTOR, Species
from imap_processing.swapi.fitting.fit_single_maxwellian_with_spin import (
    fit_single_maxwellian_with_spin,
    spin_rates_and_jacobian,
    spin_rotations,
)
from imap_processing.swapi.forward_model.deadtime import deadtime_factor

# realistic ESA voltages in V, for central energies of 100 eV/q to 20 keV/q
ESA_VOLTAGES = np.geomspace(100.0, 20000.0, 62) / SIMION_K_FACTOR
N_SWEEPS = 5


CHUNK_VOLTAGES = np.tile(ESA_VOLTAGES, (N_SWEEPS, 1))


def spin_angles(n_sweeps, n_steps, spin_period_s):
    """Spin angles of consecutive 12 s sweeps of 72 steps, 0 at the first."""
    seconds = np.add.outer(12.0 * np.arange(n_sweeps), np.arange(n_steps) / 6.0)
    return 2 * np.pi * seconds / spin_period_s


# a spin period off the nominal 15 s, as in flight
SPIN_ANGLES = spin_angles(N_SWEEPS, len(ESA_VOLTAGES), 14.89)


@pytest.fixture
def chunk_responses(response_tables, epoch):
    return response_tables.step_responses(CHUNK_VOLTAGES, epoch, Species.PROTON)


@pytest.fixture
def rotations():
    return spin_rotations(SPIN_ANGLES).reshape(-1, 3, 3)


def test_spin_rotations():
    rotations = spin_rotations(spin_angles(2, 72, 15.0))
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

    # at 60 deg the spacecraft has spun right-handed about +y, so a flow along
    # +x at spin angle 0 is seen by SWAPI rotated toward +z
    angle = np.pi / 3
    rotation = spin_rotations(angle)
    assert_allclose(
        rotation.T @ [1.0, 0.0, 0.0], [np.cos(angle), 0.0, np.sin(angle)], atol=1e-12
    )


def test_spin_rates_and_jacobian(chunk_responses, rotations):
    # far enough off the spin axis to modulate the rates strongly
    state = np.array([np.log(5.0), np.log(50_000.0), 30.0, -400.0, 60.0])
    rates, jacobian = spin_rates_and_jacobian(chunk_responses, rotations, state)
    assert rates.shape == (N_SWEEPS * len(ESA_VOLTAGES),)
    assert jacobian.shape == (*rates.shape, 5)
    # the sweeps differ as the spacecraft spins
    rates_by_sweep = rates.reshape(N_SWEEPS, -1)
    assert not np.allclose(rates_by_sweep[0], rates_by_sweep[1], rtol=0.05)

    # jacobian against central differences of the state. The kernel's analytic
    # derivatives hold the quadrature limits fixed, but the limits move with
    # the temperature and flow, sliding the nodes over kinks in the
    # piecewise-linear transmission and passband tables. With the flow off the
    # spin axis, that makes the discretized rates differ from the analytic
    # derivatives by up to ~2% of the peak at a few samples (the analytic ones
    # are the stable ones under quadrature refinement). A wrong chain rule
    # through the rotations would be off by order 100%.
    for k, step in enumerate([1e-4, 1e-4, 1e-2, 1e-2, 1e-2]):
        delta = np.zeros(5)
        delta[k] = step
        upper, _ = spin_rates_and_jacobian(chunk_responses, rotations, state + delta)
        lower, _ = spin_rates_and_jacobian(chunk_responses, rotations, state - delta)
        numeric = (upper - lower) / (2 * step)
        assert jacobian[:, k] == pytest.approx(
            numeric, rel=1e-3, abs=3e-2 * np.abs(numeric).max()
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
    response_tables, epoch, chunk_responses, rotations, speed, temperature, deflection
):
    true_density = 5.0
    # mostly along the boresight, so the flow is off the spin axis by a bit
    true_velocity = np.array([deflection, -float(speed), -deflection])
    state = np.array([np.log(true_density), np.log(temperature), *true_velocity])
    ideal = spin_rates_and_jacobian(chunk_responses, rotations, state)[0]
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)
    # the off-axis flow makes the sweeps differ as the spacecraft spins
    assert not np.allclose(count_rates[0], count_rates[1])

    fit = fit_single_maxwellian_with_spin(
        CHUNK_VOLTAGES,
        count_rates,
        spin_angles=SPIN_ANGLES,
        swapi_response=response_tables,
        epoch=epoch,
    )

    assert fit["density"] == pytest.approx(true_density, rel=0.05)
    assert fit["temperature"] == pytest.approx(temperature, rel=0.05)
    assert fit["speed"] == pytest.approx(np.linalg.norm(true_velocity), rel=0.01)


def test_fit_single_maxwellian_with_spin_fits_valid_samples_only(
    response_tables, epoch, chunk_responses, rotations
):
    """The model is only evaluated at valid samples while fitting."""
    state = np.array([np.log(5.0), np.log(100_000.0), 3.0, -450.0, -3.0])
    ideal = spin_rates_and_jacobian(chunk_responses, rotations, state)[0]
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)
    # only the steps around the peak are valid
    peak = int(np.argmax(count_rates.mean(axis=0)))
    masked = np.full_like(count_rates, np.nan)
    masked[:, peak - 4 : peak + 5] = count_rates[:, peak - 4 : peak + 5]
    n_valid = np.isfinite(masked).sum()

    with mock.patch(
        "imap_processing.swapi.fitting.fit_single_maxwellian_with_spin."
        "spin_rates_and_jacobian",
        wraps=spin_rates_and_jacobian,
    ) as spy:
        fit = fit_single_maxwellian_with_spin(
            CHUNK_VOLTAGES,
            masked,
            spin_angles=SPIN_ANGLES,
            swapi_response=response_tables,
            epoch=epoch,
        )

    sizes = [call.args[0].central_speed.shape[0] for call in spy.call_args_list]
    # every evaluation but the final, full-size model is at the valid samples
    assert sizes[:-1] == [n_valid] * (len(sizes) - 1)
    assert sizes[-1] == count_rates.size
    assert fit["model_count_rates"].shape == count_rates.shape
    assert fit["density"] == pytest.approx(5.0, rel=0.05)
    assert fit["temperature"] == pytest.approx(100_000.0, rel=0.05)
    assert fit["speed"] == pytest.approx(np.linalg.norm(state[2:]), rel=0.01)


def test_fit_single_maxwellian_with_spin_uses_spin_angles(
    response_tables, epoch, chunk_responses, rotations
):
    """Fitting with the nominal 15 s spin instead of the true angles is worse."""
    true_velocity = np.array([25.0, -450.0, -25.0])
    state = np.array([np.log(5.0), np.log(100_000.0), *true_velocity])
    ideal = spin_rates_and_jacobian(chunk_responses, rotations, state)[0]
    count_rates = (ideal * deadtime_factor(ideal)).reshape(N_SWEEPS, -1)

    fits = {
        name: fit_single_maxwellian_with_spin(
            CHUNK_VOLTAGES,
            count_rates,
            spin_angles=angles,
            swapi_response=response_tables,
            epoch=epoch,
        )
        for name, angles in [
            ("true", SPIN_ANGLES),
            ("nominal", spin_angles(N_SWEEPS, len(ESA_VOLTAGES), 15.0)),
        ]
    }

    def error(fit):
        return np.mean((fit["model_count_rates"] - count_rates) ** 2)

    assert fits["true"]["speed"] == pytest.approx(
        np.linalg.norm(true_velocity), rel=0.01
    )
    assert error(fits["true"]) < 0.1 * error(fits["nominal"])
