import numpy as np

from imap_processing.swapi.forward_model.response.deadtime import (
    SWAPI_DEADTIME_S,
    deadtime_factor,
)


def test_zero_rate_produces_unit_factor():
    # If the count rate is 0, the deadtime correction factor is 1.
    assert deadtime_factor(0.0) == 1.0


def test_factor_is_bounded_in_unit_interval():
    # The factor must lie between (0, 1] for physical (non-negative) inputs.
    factors = deadtime_factor([0.0, 1e3, 1e6, 1e9, 1e12])
    assert np.all(factors > 0.0)
    assert np.all(factors <= 1.0)


def test_factor_is_monotonically_decreasing():
    # The higher the count rate, the lower the deadtime factor.
    # Test this over a reasonably large range.
    rates = np.logspace(0, 9, 10)
    assert np.all(np.diff(deadtime_factor(rates)) < 0)


def test_factor_is_half_when_rate_equals_one_over_tau():
    # 1 / (1 + tau * (1/tau)) = 1/2.
    rate = 1.0 / SWAPI_DEADTIME_S
    np.testing.assert_allclose(deadtime_factor(rate), 0.5, rtol=1e-12)


def test_226_hz_loss_at_35000_hz_measured_rate():
    # A 35.000 kHz "measured" rate corresponds to a 35.226 kHz "true" rate.
    true_rate_hz = 35_226
    np.testing.assert_allclose(
        true_rate_hz * deadtime_factor(true_rate_hz), 35_000, rtol=1e-4
    )


def test_corrected_rate_saturates_at_one_over_tau():
    # Measured rate can never exceed 1/tau, no matter how high the true rate.
    np.testing.assert_allclose(
        1e15 * deadtime_factor(1e15), 1.0 / SWAPI_DEADTIME_S, rtol=1e-6
    )
