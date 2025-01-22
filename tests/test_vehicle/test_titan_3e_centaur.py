"""
Describe purpose of this script here

Created: 1/22/25
"""
import pytest

import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.universe import VerticalRange, TestStand
from rocket_sim.vehicle import Vehicle, g0
from vehicle.titan_3e_centaur import Titan3E

def plot_tlm(vehicle:Vehicle,tc_id:int):
    ts=np.array([tlm_point.t for tlm_point in vehicle.tlm_points])
    states=np.array([tlm_point.y0 for tlm_point in vehicle.tlm_points])
    masses=np.array([tlm_point.mass for tlm_point in vehicle.tlm_points])
    a_thr_mags=np.array([tlm_point.a_thr[2] for tlm_point in vehicle.tlm_points])
    plt.figure("Titan3E telemetry")
    plt.subplot(2,2,1)
    plt.title("Mass")
    plt.plot(ts,masses,label=f'mass {tc_id}')
    plt.legend()
    plt.subplot(2,2,2)
    plt.title("acc")
    plt.plot(ts,a_thr_mags/g0,label=f'acc {tc_id}')
    plt.legend()
    plt.subplot(2,2,3)
    plt.title("v")
    plt.plot(ts,states[:,5],label=f'vz {tc_id}')
    plt.legend()
    plt.subplot(2,2,4)
    plt.title("pos")
    plt.plot(ts,states[:,2],label='rz {tc_id}')
    plt.legend()
    plt.pause(0.1)

@pytest.mark.parametrize(
    "static",
    [True,False]
)
def test_titan3E(static:bool):
    for tc_id in (6,7):
        titan3E=Titan3E(tc_id=tc_id)
        print(titan3E.stages)
        print(titan3E.engines)
        if static:
            stand=TestStand(vehicles=[titan3E],fps=10)
            stand.runto(t1=130.0)
        else:
            range=VerticalRange(vehicles=[titan3E],fps=10,forces=[])
            range.runto(t1=130.0)
        plot_tlm(titan3E,tc_id=tc_id)
    plt.show()
