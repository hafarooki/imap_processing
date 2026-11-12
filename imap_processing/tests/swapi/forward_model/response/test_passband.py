"""Placeholder: Ported from L3a's ``tests/swapi/response/test_passband_grid.py``, run on
the passband fitted at 1 kV instead of a synthetic Gaussian.

The checks of the passband construction (normalization, boundaries, clamping)
are in ``test_response_tables.py``.
"""

import numpy as np
import pytest

from imap_processing.swapi.constants import (
    PASSBAND_BOUNDARY_THRESHOLD,
    PASSBAND_SPEED_RATIOS,
)
from imap_processing.swapi.forward_model.response.passband import (
    Aperture,
    interpolate_passband,
    speed_ratio_range_at_elevation,
)


@pytest.fixture(params=list(Aperture))
def passband(request, swapi_response):
    return swapi_response.passband(1_000.0, request.param)


def row_index(passband, elevation):
    return round((elevation - passband.min_elevation) / passband.elevation_spacing)


@pytest.mark.parametrize(
    ("elevation", "speed_ratio"),
    [
        # Placeholder: on a node
        (0.0, 1.0),
        # Placeholder: at a cell center
        (1.25, 1.005),
        # Placeholder: off center
        (1.1, 1.0014),
    ],
)
def test_interpolation_is_the_corner_weighted_sum(passband, elevation, speed_ratio):
    i_float = (elevation - passband.min_elevation) / passband.elevation_spacing
    j_float = (speed_ratio - passband.min_speed_ratio) / passband.speed_ratio_spacing
    i_lo, j_lo = int(i_float), int(j_float)
    i_weight, j_weight = i_float - i_lo, j_float - j_lo
    corners = passband.values[i_lo : i_lo + 2, j_lo : j_lo + 2]
    weights = np.array(
        [
            [(1 - i_weight) * (1 - j_weight), (1 - i_weight) * j_weight],
            [i_weight * (1 - j_weight), i_weight * j_weight],
        ]
    )

    np.testing.assert_allclose(
        interpolate_passband(passband, elevation, speed_ratio),
        (corners * weights).sum(),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("elevation", "speed_ratio"),
    [(-100.0, 1.0), (100.0, 1.0), (0.0, 0.5), (0.0, 1.5)],
)
def test_interpolation_is_zero_outside_the_grid(passband, elevation, speed_ratio):
    assert interpolate_passband(passband, elevation, speed_ratio) == 0.0


def test_speed_ratio_range_marks_cutoff_crossing_on_grid(passband):
    cutoff = PASSBAND_BOUNDARY_THRESHOLD * passband.values.max()
    rows = np.flatnonzero(~np.isnan(passband.min_boundary))
    for i in rows:
        elevation = passband.min_elevation + passband.elevation_spacing * i
        # Placeholder: the bounds of an on-grid row and its upper neighbor are combined,
        # so only the row's own crossing is bracketed
        lo, hi = speed_ratio_range_at_elevation(passband, float(elevation))
        row = passband.values[i]
        assert np.interp(lo, PASSBAND_SPEED_RATIOS, row) <= cutoff + 1e-12
        assert np.interp(hi, PASSBAND_SPEED_RATIOS, row) <= cutoff + 1e-12
        assert lo <= passband.min_boundary[i]
        assert hi >= passband.max_boundary[i]


def test_speed_ratio_range_is_the_union_of_bracketing_rows(passband):
    lower, upper = row_index(passband, 0.0), row_index(passband, 0.5)
    lo, hi = speed_ratio_range_at_elevation(passband, 0.25)

    assert lo == pytest.approx(
        min(passband.min_boundary[lower], passband.min_boundary[upper])
    )
    assert hi == pytest.approx(
        max(passband.max_boundary[lower], passband.max_boundary[upper])
    )


def test_speed_ratio_range_min_is_no_greater_than_max(passband):
    active_lo, active_hi = passband.elevation_range
    for elevation in np.linspace(active_lo, active_hi, 7):
        lo, hi = speed_ratio_range_at_elevation(passband, float(elevation))
        assert lo <= hi


def test_speed_ratio_range_at_the_edges_of_the_active_rows(passband):
    """Placeholder: Past the active rows both bracketing rows are NaN, which raises; at
    the edge only one is, and its bounds are returned."""
    valid = np.flatnonzero(~np.isnan(passband.min_boundary))
    first, last = int(valid[0]), int(valid[-1])
    n_rows = passband.min_boundary.shape[0]
    spacing, min_elevation = passband.elevation_spacing, passband.min_elevation

    # Placeholder: the fitted passbands fall below the cutoff before the grid's edges
    assert first > 1
    assert last < n_rows - 1

    with pytest.raises(ValueError, match="Elevation out of range"):
        speed_ratio_range_at_elevation(passband, min_elevation - 100.0)
    with pytest.raises(ValueError, match="Elevation out of range"):
        speed_ratio_range_at_elevation(
            passband, min_elevation + spacing * n_rows + 100.0
        )
    for row, elevation_offset in [(first, -0.5), (last, 0.5)]:
        lo, hi = speed_ratio_range_at_elevation(
            passband, min_elevation + spacing * (row + elevation_offset)
        )
        assert lo == pytest.approx(passband.min_boundary[row])
        assert hi == pytest.approx(passband.max_boundary[row])
