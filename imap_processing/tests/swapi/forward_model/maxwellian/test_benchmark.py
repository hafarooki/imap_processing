"""Claude: Timing benchmark of the Maxwellian count rates for a five-sweep chunk.

Not a pass/fail performance check (CI machines vary); it prints the time per
call so the effect of changes can be compared. Run it with output shown::

    poetry run pytest -s \
        imap_processing/tests/swapi/forward_model/maxwellian/test_benchmark.py
"""

import time

import numpy as np
import pytest

from imap_processing.swapi.constants import Species
from imap_processing.swapi.forward_model.look_angles import look_angles_to_velocity
from imap_processing.swapi.forward_model.maxwellian.integration import (
    maxwellian_count_rates,
)
from imap_processing.swapi.forward_model.response.measurement_response import (
    measurement_responses,
)

N_SWEEPS = 5
# Claude: ESA voltage of a 400 km/s proton at the center of the passband
PEAK_VOLTAGE = 444.0


def _sweep_voltages():
    """Claude: One 72-step sweep shaped like the flight sweep tables.

    Step 0 is the ramp-up (held at the top voltage), steps 1-62 the coarse sweep
    stepping down geometrically from 10240 V to 51.81 V, and steps 63-71 the
    fine sweep, placed around the proton peak as the onboard solver would.
    """
    coarse = 10240.0 * (51.81 / 10240.0) ** (np.arange(62) / 61)
    fine = PEAK_VOLTAGE * np.geomspace(0.94, 1.06, 9)
    return np.concatenate([[10240.0], coarse, fine])


@pytest.fixture
def chunk_responses(swapi_response, epoch):
    voltages = np.tile(_sweep_voltages(), N_SWEEPS)
    return measurement_responses(
        swapi_response, voltages, epoch, Species.PROTON, np.eye(3)
    )


def _time_per_call_us(function, n_calls=20, n_repeats=5):
    """Claude: Best and median microseconds per call over the repeats."""
    function()
    per_call = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        for _ in range(n_calls):
            function()
        per_call.append((time.perf_counter() - start) / n_calls * 1e6)
    return min(per_call), float(np.median(per_call))


@pytest.mark.parametrize("temperature", [2e4, 1e5, 5e5])
def test_benchmark_five_sweep_chunk(chunk_responses, temperature):
    velocity = np.asarray(look_angles_to_velocity(400.0, 2.0, 1.0), dtype=float)

    def evaluate():
        return maxwellian_count_rates(5.0, temperature, velocity, chunk_responses)

    rates, jacobian = evaluate()
    assert rates.shape == (N_SWEEPS * 72,)
    assert jacobian.shape == (N_SWEEPS * 72, 5)
    assert np.all(np.isfinite(rates))
    assert rates.max() > 0

    best, median = _time_per_call_us(evaluate)
    n_active = int(np.count_nonzero(rates))
    print(
        f"\n5x72 chunk, T={temperature:.0e} K ({n_active} steps with counts): "
        f"best {best:.0f} us/call, median {median:.0f} us/call"
    )
