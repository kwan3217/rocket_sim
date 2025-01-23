"""
Guidance programs intended to mostly be used in orbit

Created: 1/22/25
"""
from typing import Callable

import numpy as np
from kwanmath.geodesy import llr2xyz
from kwanmath.vector import vnormalize, vcross, vdot

from rocket_sim.vehicle import Vehicle


def prograde_guide(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
    # Prograde
    return y[3:]/np.linalg.norm(y[3:])


def inertial_guide(*,lon:float|None=None,lat:float|None=None,v:float|None=None):
    if v is None:
        lon=np.deg2rad(lon)
        lat=np.deg2rad(lat)
        v=np.array([np.cos(lon)*np.cos(lat),np.sin(lon)*np.cos(lat),np.sin(lat)])
    def inner(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
        return v
    return inner


def yaw_rate_guide(*,r0:np.ndarray,v0:np.ndarray,dpitch:float,dyaw:float,yawrate:float,t0:float)->Callable[...,np.ndarray]:
    # Pre-calculate as much as we can
    # We are going to define our variation from prograde in the VNC frame:
    #  (V)elocity vector is along first axis
    #  (N)ormal vector is perpendicular to orbit plane and therefore parallel to r x v
    #  (C)o-normal is perpendicular to the other two and as close to parallel to r as is reasonable.
    # Figure these basis vectors as the appear in ICRF
    rbar = vnormalize(r0.reshape(-1, 1))  # Just for legibility purposes -- rbar is not a basis vector.
    vbar = vnormalize(v0.reshape(-1, 1))
    nbar = vnormalize(vcross(rbar, vbar))
    cbar = vnormalize(vcross(vbar, nbar))
    assert vdot(rbar, cbar) > 0, 'cbar is the wrong direction'
    # So each column i of a matrix is where basis vector i of the
    # from system lands in the to system. v is VNC frame, j is ICRF/J2000 frame
    # This way, the vector <1,0,0> in VNC is prograde, the xy(vn) plane is equatorial
    # and a "lon/lat" vector in this frame is a yaw=lon and pitch=lat deviation
    # from prograde.
    Mjv = np.hstack((vbar, nbar, cbar))

    def inner(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
        aim_v = llr2xyz(lon=dyaw+yawrate*(t-t0), lat=dpitch, r=1, deg=True)
        aim_j = Mjv @ aim_v
        return aim_j.reshape(-1)
    return inner


def dprograde_guide(*,dpitch:float,dyaw:float,pitchrate:float,t0:float)->Callable[...,np.ndarray]:
    """
    Guidance program that applies a pitch and yaw to prograde guidance
    :param dpitch:
    :param dyaw:
    :return: guidance routine which has the given pitch and yaw offset from
             prograde in the instantaneous VNC frame
    """
    def inner(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
        # Can't pre-calculate because we want to use instantaneous prograde direction
        # We are going to define our variation from prograde in the VNC frame:
        # Figure these basis vectors as the appear in ICRF
        rbar = vnormalize(y[:3].reshape(-1, 1))  # Just for checking purposes -- rbar is not a basis vector.
        vbar = vnormalize(y[3:].reshape(-1, 1))
        nbar = vnormalize(vcross(rbar, vbar))
        cbar = vnormalize(vcross(vbar, nbar))
        # Verify that cbar is roughly up
        assert vdot(rbar, cbar) > 0, 'cbar is the wrong direction'
        # So each column i of a matrix is where basis vector i of the
        # from system lands in the to system. v is VNC frame, j is ICRF/J2000 frame
        # This way, the vector <1,0,0> in VNC is prograde, the xy(vn) plane is equatorial
        # and a "lon/lat" vector in this frame is a yaw=lon and pitch=lat deviation
        # from prograde.
        Mjv = np.hstack((vbar, nbar, cbar))
        aim_v = llr2xyz(lon=dyaw, lat=dpitch+pitchrate*(t-t0), r=1, deg=True)
        aim_j = Mjv @ aim_v
        return aim_j.reshape(-1)
    return inner


def seq_guide(guides:dict[float,Callable[...,np.ndarray]]):
    """
    Create a guidance program that sequences a bunch of separate guidance programs
    :param guides: The key is the time that the attached program switches off, the value
                   is the guidance program active before this time. The last one might
                   as well be active until float('inf').
    :return:
    """
    t1s=[k for k,v in guides.items()]
    t0s=[-float('inf')]+t1s[:-1]
    guides=[v for k,v in guides.items()]
    def inner(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
        for i,(t0,t1) in enumerate(zip(t0s,t1s)):
            if t0<=t<t1:
                return guides[i](t=t,y=y,dt=dt,major_step=major_step,vehicle=vehicle)
        return np.array([0,0,1])
    return inner
