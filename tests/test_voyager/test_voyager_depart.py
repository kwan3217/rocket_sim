"""
Describe purpose of this script here

Created: 1/25/25
"""
import pytest
from spiceypy import kclear

from voyager.voyager_depart_target import init_spice, PMTargeter, Centaur2Targeter, best_pm_solution


@pytest.fixture
def kernels():
    init_spice()
    yield None
    kclear()


@pytest.mark.parametrize(
    "vgr_id",
    [1,2]
)
def test_target_pm(tmp_path,kernels,vgr_id):
    pm_targeter=PMTargeter(vgr_id=vgr_id,out_path=tmp_path)
    pm_targeter.target(optimize=False)
    pm_targeter.export()


@pytest.mark.parametrize(
    "vgr_id",
    [1,2]
)
def test_target_centaur2(tmp_path,kernels,vgr_id):
    pm_guess = best_pm_solution(vgr_id=vgr_id)
    centaur2_targeter = Centaur2Targeter(pm=pm_guess,out_path=tmp_path)
    centaur2_targeter.target(optimize=False)


