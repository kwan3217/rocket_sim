from collections import namedtuple

import numpy as np
import pytest
from atmosphere.atmosphere import SimpleEarthAtmosphere, Atmosphere
from kwanmath.vector import vlength
from matplotlib import pyplot as plt

from rocket_sim.drag import f_drag, mach_drag, INCHES
from rocket_sim.planet import Planet
from rocket_sim.universe import VerticalRange
from rocket_sim.vehicle import Vehicle, g0
from vehicle.titan_3e_centaur import Titan3E


def plot_tlm(vehicle:Vehicle,drag_enabled:bool):
    ts=np.array([tlm_point.t for tlm_point in vehicle.tlm_points])
    states=np.array([tlm_point.y0 for tlm_point in vehicle.tlm_points])
    masses=np.array([tlm_point.mass for tlm_point in vehicle.tlm_points])
    a_thr_mags=np.array([vlength(tlm_point.F_thr) for tlm_point in vehicle.tlm_points])/masses
    drag_mags=np.array([tlm_point.Fs[0][2] if len(tlm_point.Fs)>0 else 0.0 for tlm_point in vehicle.tlm_points])
    a_drag_mags=drag_mags/masses
    a_grav_mags=np.array([tlm_point.accs[0][2] for tlm_point in vehicle.tlm_points])
    a_mags=a_thr_mags+a_drag_mags+a_grav_mags
    plt.figure("Vehicle telemetry")
    plt.subplot(4,1,1)
    plt.ylabel("Mass/kg")
    plt.plot(ts,masses,label=f'mass {drag_enabled}')
    plt.legend()
    plt.subplot(4,1,2)
    plt.ylabel("acc/g")
    if drag_enabled:
        plt.plot(ts,a_thr_mags/g0,label=f'thr {drag_enabled}')
        plt.plot(ts,a_drag_mags/g0,label=f'drag {drag_enabled}')
        plt.plot(ts,a_grav_mags/g0,label=f'grav {drag_enabled}')
        plt.plot(ts,a_mags/g0,label=f'tot {drag_enabled}')
    plt.legend()
    plt.subplot(4,1,3)
    plt.ylabel("v/(m/s)")
    plt.plot(ts,states[:,5],label=f'vz {drag_enabled}')
    plt.legend()
    plt.subplot(4,1,4)
    plt.ylabel("pos/m")
    plt.xlabel("time/s")
    plt.plot(ts,states[:,2],label=f'rz {drag_enabled}')
    plt.legend()
    plt.pause(0.1)


class FlatWorld(Planet):
    def __init__(self,*,atm:Atmosphere):
        super().__init__(atm=atm,w0=0.0,mu=1.0,re=1000.0,f=0.0)
    def b2lla(self,rb:np.array,centric:bool=False,deg:bool=False):
        return namedtuple('xyz2lla', ['lat', 'lon', 'alt'])(0,0,rb[2])


@pytest.mark.parametrize(
    "drag_enabled",
    [True,False]
)
def test_drag(drag_enabled):
    tc_id=6
    titan3E=Titan3E(tc_id=tc_id)
    print(titan3E.stages)
    print(titan3E.engines)
    if drag_enabled:
        range=VerticalRange(vehicles=[titan3E],fps=10,forces=[])
        range.runto(t1=130.0)
    else:
        world=FlatWorld(atm=SimpleEarthAtmosphere())
        range=VerticalRange(vehicles=[titan3E],fps=10,forces=[
            f_drag(planet=world,clcd=mach_drag(),Sref=np.pi*(60*INCHES)**2)
        ])
        range.runto(t1=130.0)
    plot_tlm(titan3E,drag_enabled)
