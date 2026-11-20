"""Timestamps of the individual ESA steps within a SWAPI sweep."""

import numpy as np

from imap_processing.swapi.constants import (
    NUM_ENERGY_STEPS,
    SWEEP_CADENCE_SECONDS,
)
from imap_processing.swapi.l2.swapi_l2 import SWAPI_LIVETIME


def sweep_measurement_times(sweep_epoch: int) -> list[int]:
    """
    Compute the measurement times for each ESA step within a sweep.

    To associate measurements with the spacecraft orientation at the time the
    measurement was taken, it is necessary to evaluate the exact time each
    measurement was taken. The start time for step ``i`` in ``[0, 71]`` is given
    by::

        t_start[i] = t_start + i * (12 / 72) s

    Counts accumulate during the 145 ms livetime at the end of each
    12 / 72 s ~= 166.67 ms step. The flight software ensures that the beginning
    and end of this window is consistent to within a few milliseconds. The
    measurement time ``t[i]`` of each ESA step is set to the center of that
    livetime::

        t[i] = t_start + (i + 1) * 166.67 ms - 145 ms / 2

    Parameters
    ----------
    sweep_epoch : int
        The center time of the sweep (6 s after sweep start) in TT J2000 ns.

    Returns
    -------
    measurement_times : list[int]
        A list of measurement times for the 72 steps in the sweep in TT J2000 ns.
    """
    t_sweep_start = sweep_epoch - round(SWEEP_CADENCE_SECONDS / 2 * 1e9)

    i = np.arange(NUM_ENERGY_STEPS)
    step_period_s = SWEEP_CADENCE_SECONDS / NUM_ENERGY_STEPS
    t_step = (i + 1) * step_period_s - SWAPI_LIVETIME / 2

    return [t_sweep_start + round(dt * 1e9) for dt in t_step]
