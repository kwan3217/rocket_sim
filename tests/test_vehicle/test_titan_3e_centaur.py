"""
Describe purpose of this script here

Created: 1/22/25
"""
from itertools import product

import pytest

import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.universe import VerticalRange, TestStand, ZeroGRange
from rocket_sim.vehicle import Vehicle, g0
from vehicle.titan_3e_centaur import Titan3E

def plot_tlm_vertical(vehicle:Vehicle,tc_id:int,mode:int):
    ts=np.array([tlm_point.t for tlm_point in vehicle.tlm_points])
    states=np.array([tlm_point.y0 for tlm_point in vehicle.tlm_points])
    masses=np.array([tlm_point.mass for tlm_point in vehicle.tlm_points])
    a_thr_mags=np.array([tlm_point.F_thr[2] for tlm_point in vehicle.tlm_points])/masses
    plt.figure(f"Titan3E Test Stand telemetry")
    if mode==1:
        plt.subplot(4,1,1)
        plt.ylabel("Mass/Mg")
        plt.plot(ts,masses/1000,label=f'mass {tc_id}')
        plt.legend()
        plt.subplot(4,1,2)
        plt.ylabel("acc/g")
        plt.plot(ts,a_thr_mags/g0,label=f'acc {tc_id}')
        plt.legend()
    else:
        plt.subplot(4,1,3)
        plt.ylabel("v/(m/s)")
        plt.plot(ts,states[:,5],label=f'vz {tc_id} {"zerog" if mode==2 else "grav"}')
        plt.legend()
        plt.subplot(4,1,4)
        plt.ylabel("alt/km")
        plt.xlabel("time/s")
        plt.plot(ts,states[:,2]/1000,label=f'rz {tc_id} {"zerog" if mode==2 else "grav"}')
        plt.legend()
    plt.pause(0.1)

@pytest.mark.parametrize(
    "mode,tc_id",
    list(product([1,2,3],[6,7]))
)
def test_titan3E(mode:int,tc_id:int):
    titan3E=Titan3E(tc_id=tc_id)
    print(titan3E.stages)
    print(titan3E.engines)
    if mode==1:
        stand=TestStand(vehicles=[titan3E],fps=10)
        stand.runto(t1=4400.0)
    elif mode==2:
        range=ZeroGRange(vehicles=[titan3E],fps=10)
        range.runto(t1=4400.0)
    elif mode==3:
        range=VerticalRange(vehicles=[titan3E],fps=10,forces=[])
        range.runto(t1=4400.0)
    plot_tlm_vertical(titan3E,tc_id=tc_id,mode=mode)
    plt.pause(0.1)


