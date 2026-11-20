import numpy as np
import pytest

from imap_processing.spice.time import str_yyyymmdd_to_ttj2000ns
from imap_processing.swapi.forward_model.response.measurement_times import (
    sweep_measurement_times,
)
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME


def test_sweep_measurement_times():
    sweep_start = int(str_yyyymmdd_to_ttj2000ns("20260101"))
    times = sweep_measurement_times(sweep_start + int(6e9))

    assert len(times) == 72
    step_spacing = 12 / 72
    ramp_up_s = step_spacing - SWAPI_LIVETIME
    livetime_center_offset = SWAPI_LIVETIME / 2
    for bin_index, time in enumerate(times):
        assert isinstance(time, int)

        # center of the livetime that follows the voltage ramp-up.
        seconds_into_sweep = (
            bin_index * step_spacing
            + ramp_up_s
            + livetime_center_offset
        )
        expected = seconds_into_sweep * 1e9
        assert time - sweep_start == pytest.approx(expected, rel=0, abs=1)

    # ensure evenly spaced with expected spacing
    np.testing.assert_allclose(np.diff(times), step_spacing * 1e9, atol=1)
