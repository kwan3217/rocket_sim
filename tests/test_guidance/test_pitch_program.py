"""
Describe purpose of this script here

Created: 1/23/25
"""
import pytest

import numpy as np
from kwanmath.vector import vangle, vlength
from matplotlib import pyplot as plt
from spiceypy import pxform, kclear

from guidance.pitch_program import pitch_program
from rocket_sim.drag import INCHES, f_drag, mach_drag
from rocket_sim.gravity import SpiceTwoBody, SpiceJ2, SpiceThirdBody

from rocket_sim.planet import SpicePlanet, Earth
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Vehicle
from vehicle.titan_3e_centaur import Titan3E
from vehicle.voyager import voyager_et0, init_spice


def plot_tlm(vehicle:Vehicle,tc_id:int,earth:SpicePlanet,
             launch_lat:float, launch_lon:float, deg:bool,
             launch_alt:float, launch_et0:float,drag_enabled:bool):
    ts=np.array([tlm_point.t for tlm_point in vehicle.tlm_points])
    states=np.array([tlm_point.y0 for tlm_point in vehicle.tlm_points]).T
    masses=np.array([tlm_point.mass for tlm_point in vehicle.tlm_points])
    Fs=vlength(np.array([tlm_point.F_thr for tlm_point in vehicle.tlm_points]).T)
    alts=earth.b2lla(states[0:3,:]).alt
    # Position of the launchpad at each t, for calculating downrange
    y0b=earth.lla2b(lat_deg=launch_lat,lon_deg=launch_lon,alt=launch_alt)
    # Make a stack of matrices of shape (N,3,3). Each Mjbs[i,:,:] is
    # the (3,3) matrix for ts[i].
    Mjbs=np.array([pxform(earth.bf_frame,'J2000',t+launch_et0) for t in ts])
    # Transform y0b by each matrix in the stack above. This produces
    # a result of shape (3,N), one column vector for each time point.
    y0js=(Mjbs @ y0b)[:,:,0].T #Should be shape (3,N), one column vector for each time point
    downranges=vangle(y0js,states[0:3,:])*earth.re
    plt.figure("Vehicle telemetry")
    plt.subplot(2,2,1)
    plt.ylabel("Alt/km")
    plt.xlabel("simt/s")
    plt.plot(ts,alts/1000,label=f'alt {"with drag" if drag_enabled else "no drag"}')
    plt.legend()
    plt.subplot(2,2,2)
    plt.ylabel("Downrange/km")
    plt.xlabel("simt/s")
    plt.plot(ts,downranges/1000,label=f'alt {"with drag" if drag_enabled else "no drag"}')
    plt.legend()
    plt.subplot(2,2,3)
    plt.ylabel("Alt/km")
    plt.xlabel("Downrange/km")
    plt.plot(downranges/1000,alts/1000,label=f'traj {"with drag" if drag_enabled else "no drag"}')
    plt.axis('equal')
    plt.legend()
    plt.subplot(2,2,4)
    plt.ylabel("F/N")
    plt.xlabel("simt/s")
    plt.plot(ts,Fs,label=f'Thrust {"with drag" if drag_enabled else "no drag"}')
    plt.legend()
    plt.pause(0.1)


@pytest.fixture
def kernels():
    init_spice()
    yield None
    kclear()


@pytest.mark.parametrize(
   "vgr_id,drag_enabled",
    [(1,True),(1,False)]
)
def test_titan3E_pitch_program_spheroid_drag(kernels,vgr_id:int,drag_enabled:bool):
    earth=Earth()
    pad41_lat= 28.583468 # deg, From Google Earth, so on WGS-84
    pad41_lon=-80.582876
    pad41_alt=0 # Hashtag Florida
    et0=voyager_et0[vgr_id]
    y0=earth.launchpad(lat=pad41_lat,lon=pad41_lon,alt=pad41_alt,deg=True,et=et0)
    titan3E = Titan3E(tc_id=vgr_id+5)
    # Flight azimuth from TC-6 report section IV
    titan3E.guide = pitch_program(planet=earth, y0=y0, azimuth=90.0, deg=True,
                                  t0=0.0, pitch0=92.0,
                                  tdpitch=[( 10.0, -1.17),
                                           ( 20.0, -0.53),
                                           ( 30.0, -0.73),
                                           ( 62.0, -0.63),
                                           ( 75.0, -0.52),
                                           ( 95.0, -0.38),
                                           (114.0,  0.00),
                                           (130.0, -0.75),
                                           (140.0, -0.08)])
    print(titan3E.stages)
    print(titan3E.engines)
    earth_twobody = SpiceTwoBody(spiceid=399)
    earth_j2 = SpiceJ2(spiceid=399)
    moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=voyager_et0[vgr_id])
    sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=voyager_et0[vgr_id])
    Sref=np.pi*(7*12*INCHES)**2+2*np.pi*(60*INCHES)**2
    drag=f_drag(planet=earth, clcd=mach_drag(), Sref=Sref)
    sim = Universe(vehicles=[titan3E],
                   accs=[earth_twobody, earth_j2, moon, sun],
                   forces=[drag] if drag_enabled else [],
                   t0=0, y0s=[y0], fps=10)
    sim.runto(t1=3800)
    plot_tlm(titan3E, tc_id=titan3E.tc_id,earth=earth,
             launch_lat=pad41_lat,launch_lon=pad41_lon,deg=True,
             launch_alt=pad41_alt,launch_et0=et0,drag_enabled=drag_enabled)
