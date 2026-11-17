import pytest

from imap_processing.swapi.forward_model.response.passband import (
    interpolate_passband,
    speed_ratio_range_at_elevation,
)


@pytest.mark.parametrize("aperture", ["SG", "OA"])
def test_passband_normalization_and_bounds(passband_fit_table, aperture):
    passband = passband_fit_table.evaluate_passband(1000 / 1.89, aperture)
    cutoff = 1e-2 * passband.values.max()

    # normalized to 1 at the passband center
    assert interpolate_passband(passband, 0.0, 1.0) == pytest.approx(1.0)

    # zero far outside the elevation range
    assert interpolate_passband(passband, 45.0, 1.0) == 0.0
    assert interpolate_passband(passband, -45.0, 1.0) == 0.0

    for target_elevation in (-5.0, -2.5, 0.0, 2.5, 5.0):
        # row near target elevation
        row = round(
            (target_elevation - passband.min_elevation) / passband.elevation_spacing
        )
        # exact elevation of that row
        elevation = passband.min_elevation + row * passband.elevation_spacing
        lower, upper = passband.speed_ratio_spans[row]

        assert lower < upper

        # at cutoff at the edges
        assert interpolate_passband(passband, elevation, lower) == pytest.approx(cutoff)
        assert interpolate_passband(passband, elevation, upper) == pytest.approx(cutoff)

        # below cutoff outside the range
        step = passband.speed_ratio_spacing / 2
        assert interpolate_passband(passband, elevation, lower - step) < cutoff
        assert interpolate_passband(passband, elevation, upper + step) < cutoff

    min_elevation, max_elevation = passband.elevation_range
    assert min_elevation < 0.0 < max_elevation

    # integration limits don't exist AT the boundaries
    assert speed_ratio_range_at_elevation(passband, min_elevation) is not None
    assert speed_ratio_range_at_elevation(passband, max_elevation) is not None

    # they do exist WITHIN the boundaries
    assert speed_ratio_range_at_elevation(passband, min_elevation - 1e-3) is None
    assert speed_ratio_range_at_elevation(passband, max_elevation + 1e-3) is None
