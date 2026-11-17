import numpy as np
import pandas as pd
from pytest import approx

from imap_processing.swapi.forward_model.response.azimuthal_transmission import (
    AzimuthalTransmission,
    interpolate_azimuthal_transmission,
)


def test_interpolate_azimuthal_transmission_matches_expected(response_table_paths):
    path = response_table_paths["azimuthal_transmission_path"]
    target_azimuths = np.linspace(-720, 720)
    df = pd.read_csv(path).fillna(0)
    instance = AzimuthalTransmission.from_csv(path)
    actual = np.stack(
        [interpolate_azimuthal_transmission(instance, x) for x in target_azimuths]
    )
    expected = np.interp(
        x=np.abs((target_azimuths + 180) % 360 - 180),
        xp=df.abs_azimuth.values,
        fp=df.transmission.values,
    )
    assert actual == approx(expected)
