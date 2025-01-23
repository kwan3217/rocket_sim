"""
Pitch program guidance

Created: 1/23/25
"""
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.planet import SpicePlanet
from rocket_sim.vehicle import Vehicle


def pitch_program(*, planet: SpicePlanet, y0:np.ndarray, azimuth: float, deg: bool, t0:float, pitch0: float, tdpitch: list[tuple[float,float]]) ->Callable[...,np.ndarray]:
    Mjd=planet.downrange_frame(y0[:3],azimuth=azimuth,deg=deg)
    this_t=t0
    this_p=pitch0
    tpitch=[(this_t,this_p)]
    for ttick,dptick in tdpitch:
        deltat=ttick-this_t
        this_t=ttick
        this_p+=dptick*deltat
        tpitch.append((this_t,this_p))
    tpitch=np.array(tpitch)
    ts=tpitch[:,0]
    pitches=tpitch[:,1]
    if deg:
        pitches=np.deg2rad(pitches)
        pitch0=np.deg2rad(pitch0)
    tpitch=lambda t:np.interp(t,np.array(ts),np.array(pitches),left=pitch0,right=pitches[-1])
    plt.figure("Pitch program")
    plt.plot(np.arange(0,140,0.1),np.rad2deg(tpitch(np.arange(0,140,0.1))))
    plt.pause(0.1)
    def inner(*, t: float, y: np.ndarray, dt: float, major_step: bool, vehicle: Vehicle):
        pitch=tpitch(t)
        vhatd=np.array([[np.cos(pitch)],
                        [0.0],
                        [np.sin(pitch)]])
        vhatj=Mjd @ vhatd
        if major_step:
            vehicle.tlm_point.pitch_program_pitch=pitch
            vehicle.tlm_point.pitch_program_vhatd=vhatd
            vehicle.tlm_point.pitch_program_vhatd=vhatj
        return vhatj.reshape(-1)
    return inner

