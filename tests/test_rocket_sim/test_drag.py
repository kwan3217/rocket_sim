from collections import namedtuple

import numpy as np
import pytest
from atmosphere.atmosphere import SimpleEarthAtmosphere, Atmosphere
from matplotlib import pyplot as plt

from rocket_sim.drag import f_drag, mach_drag, INCHES
from rocket_sim.planet import Planet
from rocket_sim.universe import VerticalRange
from rocket_sim.vehicle import Vehicle
from vehicle.titan_3e_centaur import Titan3E


def plot_tlm(vehicle:Vehicle,tc_id:int):
    ts=np.array([t for t,y,mass,thr_mag in vehicle.tlm])
    states=np.array([y for t,y,mass,thr_mag in vehicle.tlm])
    masses=np.array([mass for t,y,mass,thr_mag in vehicle.tlm])
    thr_mags=np.array([thr_mag for t,y,mass,thr_mag in vehicle.tlm])
    plt.figure("Vehicle telemetry")
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
    plt.plot(states[:,2],label=f'rz {tc_id}')
    plt.legend()
    plt.pause(0.1)


class FlatWorld(Planet):
    def __init__(self,*,atm:Atmosphere):
        super().__init__(M0_rb=None,atm=atm,w0=0.0,mu=1.0,re=1000.0,f=0.0)
    def b2lla(self,rb:np.array,centric:bool=False,deg:bool=False):
        return namedtuple('xyz2lla', ['lat', 'lon', 'alt'])(0,0,rb[3])



def test_drag():
    tc_id=6
    for drag_enabled in (False,True):
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
        plot_tlm(titan3E,tc_id=1 if drag_enabled else 0)
    plt.show()
