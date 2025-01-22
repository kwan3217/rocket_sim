"""
Describe purpose of this script here

Created: 1/22/25
"""
import pytest

import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.universe import VerticalRange
from rocket_sim.vehicle import Vehicle
from titan_3e_centaur import Titan3E

def plot_tlm(vehicle:Vehicle,tc_id:int):
    ts=np.array([t for t,y,mass,thr_mag in vehicle.tlm])
    states=np.array([y for t,y,mass,thr_mag in vehicle.tlm])
    masses=np.array([mass for t,y,mass,thr_mag in vehicle.tlm])
    thr_mags=np.array([thr_mag for t,y,mass,thr_mag in vehicle.tlm])
    plt.figure("Titan3E telemetry")
    plt.subplot(2,2,1)
    plt.title("Mass")
    plt.plot(masses,label=f'mass {tc_id}')
    plt.legend()
    plt.subplot(2,2,2)
    plt.title("acc")
    plt.plot(thr_mags/masses/9.80665,label=f'acc {tc_id}')
    plt.legend()
    plt.subplot(2,2,3)
    plt.title("v")
    plt.plot(states[:,5],label=f'vz {tc_id}')
    plt.legend()
    plt.subplot(2,2,4)
    plt.title("pos")
    plt.plot(states[:,2],label='rz {tc_id}')
    plt.legend()
    plt.pause(0.1)


def test_titan3E():
    for tc_id in (6,7):
        titan3E=Titan3E(tc_id=tc_id)
        print(titan3E.stages)
        print(titan3E.engines)
        if False:
            stand=TestStand(vehicles=[titan3E],fps=10)
            stand.runto(t1=130.0)
        else:
            range=VerticalRange(vehicles=[titan3E],fps=10,forces=[])
            range.runto(t1=130.0)
        plot_tlm(titan3E,tc_id=tc_id)
    plt.show()
