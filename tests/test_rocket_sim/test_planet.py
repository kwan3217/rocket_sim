"""
Describe purpose of this script here

Created: 1/22/25
"""
import numpy as np
import pytest
from atmosphere.atmosphere import SimpleEarthAtmosphere
from spiceypy import furnsh, kclear

from rocket_sim.planet import Earth
from voyager.voyager_depart_target import voyager_et0, init_spice


@pytest.fixture
def earth()->Earth:
    init_spice()
    yield Earth()
    kclear()


def test_wind(earth):
    print(earth)
    # Wind vector 1m along x axis from geocenter
    # is +y*w0
    wind=earth.wind(rj=np.array([1,0,0]))
    assert np.isclose(wind[1],earth.w0),"Wind vector came out wrong"


def test_launchpad(earth:Earth):
    pad41_lat= 28.583468 # deg, From Google Earth, so on WGS-84
    pad41_lon=-80.582876
    pad41_alt=0 # Hashtag Florida
    y0=earth.launchpad(lat=pad41_lat,lon=pad41_lon,alt=pad41_alt,deg=True,et=voyager_et0[1])
    print(y0)


def test_downrange(earth:Earth):
    lat= 0 # deg
    lon= 0
    alt=0
    rj=earth.lla2b(lat_deg=lat,lon_deg=lon,alt=alt)
    print(earth.downrange_frame(rj,azimuth=60,deg=True))


