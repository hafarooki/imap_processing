import pytest
from numpy.testing import assert_allclose

from imap_processing.swapi.forward_model.look_angles import (
    look_angles_to_velocity,
    velocity_to_look_angles,
)


def test_look_angles_to_velocity_points_into_boresight():
    assert_allclose(look_angles_to_velocity(400.0, 0.0, 0.0), [0.0, -400.0, 0.0])
    assert_allclose(
        look_angles_to_velocity(400.0, 90.0, 0.0), [-400.0, 0.0, 0.0], atol=1e-12
    )
    assert_allclose(
        look_angles_to_velocity(400.0, 0.0, 90.0), [0.0, 0.0, -400.0], atol=1e-12
    )


@pytest.mark.parametrize(
    ("azimuth", "elevation"),
    [(0.0, 0.0), (2.0, 1.0), (-30.0, 10.0), (120.0, -45.0), (-170.0, 80.0)],
)
def test_velocity_to_look_angles_inverts_look_angles_to_velocity(azimuth, elevation):
    vx, vy, vz = look_angles_to_velocity(450.0, azimuth, elevation)
    assert_allclose(velocity_to_look_angles(vx, vy, vz), (azimuth, elevation))
