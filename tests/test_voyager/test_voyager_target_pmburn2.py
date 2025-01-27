"""
Describe purpose of this script here

Created: 1/25/25
"""
import pytest
from spiceypy import kclear

from vehicle.voyager import init_spice
from vehicle.voyager_depart_target import target_pm


@pytest.fixture
def kernels():
    init_spice()
    yield None
    kclear()


@pytest.mark.parametrize(
    "vgr_id",
    [1,2]
)
def test_target_pm(kernels,vgr_id):
    target_pm(export=True, optimize=False,vgr_id=vgr_id)


