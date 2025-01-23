from copy import copy

import numpy as np
import pytest
from matplotlib import pyplot as plt

from rocket_sim.vehicle import Stage, Engine, Vehicle, kg_per_lbm, g0
from rocket_sim.universe import ZeroGRange, TestStand

# From The Voyager Spacecraft, Gold Medal Lecture in Mech Eng, table 2 bottom line
Voyager1WetMass=825.4
Voyager1PropMass=103.4
Voyager1Stage=Stage(prop=Voyager1PropMass,total=Voyager1WetMass) #dry mass and RCS prop for Voyager
# Value from TC-7 Voyager 2 Flight Data Report, p10
MMPMTotalMass=4470*kg_per_lbm
PMWetMass=MMPMTotalMass-Voyager1WetMass
# Values from AIAA79-1334 Voyager Prop System, table 5
PMPropMass=1045.9
PMStage=Stage(prop=PMPropMass,total=PMWetMass)
BurnTime0=3722.2 # PM start from TC-6 timeline
BurnTime1=3767.3 # PM burnout
BurnTime=BurnTime1-BurnTime0
PMItot=2895392
PMve=PMItot/PMStage.prop_mass # Exhaust velocity, m/s
PMF=PMItot/BurnTime      # Mean thrust assuming rectangular thrust curve
PMEngine=Engine(PMF,PMve)


def plot_tlm(vehicle:Vehicle):
    ts=np.array([tlm_point.t for tlm_point in vehicle.tlm_points])
    states=np.array([tlm_point.y0 for tlm_point in vehicle.tlm_points])
    masses=np.array([tlm_point.mass for tlm_point in vehicle.tlm_points])
    a_thr_mags=np.array([tlm_point.a_thr[2] for tlm_point in vehicle.tlm_points])
    plt.figure("Time")
    plt.plot(ts,label='mass')
    plt.figure("Mass")
    plt.plot(masses,label='mass')
    plt.figure("acc")
    plt.plot(a_thr_mags/g0,label='acc')
    plt.figure("v")
    plt.plot(states[:,5],label='vz')
    plt.figure("pos")
    plt.plot(states[:,2],label='rz')
    plt.pause(0.1)


Voyager1=Vehicle(stages=[Voyager1Stage,PMStage],engines=[(PMEngine,1)])


@pytest.mark.parametrize("vehicle", [copy(Voyager1)])
def test_stand(vehicle:Vehicle):
    stand=TestStand(vehicles=[vehicle],fps=10)
    stand.runto(t1=45.1)
    plot_tlm(vehicle)


@pytest.mark.parametrize("vehicle", [copy(Voyager1)])
def test_zero_g_range(vehicle:Vehicle):
    stand=ZeroGRange(vehicles=[vehicle],fps=10)
    stand.runto(t1=45.1)
    plot_tlm(vehicle)


@pytest.mark.parametrize("vehicle", [copy(Voyager1)])
def test_time_reverse(vehicle:Vehicle):
    stand=ZeroGRange(vehicles=[vehicle],fps=10)
    stand.runto(t1=45.1)
    stand.runto(t1=0.0)
    plot_tlm(vehicle)
