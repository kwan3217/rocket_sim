"""
Gravitational forces

Created: 1/16/25
"""
from typing import Callable

import numpy as np
from spiceypy import spkezr, gdpool


def aTwoBody(*,gm:float)->Callable[...,np.ndarray]:
    def inner(*,t:float,y:np.ndarray,dt:float)->np.ndarray:
        """
        Two-body gravity acceleration
        :param rv: Position vector in an inertial frame
        :return: Two-body acceleration in (distance units implied by rv)/s**2
        """
        return -gm*y[:3]/np.linalg.norm(y[:3])**3
    return inner


def SpiceTwoBody(spiceid:int)->Callable[...,np.ndarray]:
    """
    Return a two-body gravity field based on Spice kernels
    :param spiceid: Integer spice ID of body in question. Must have kernels
                    loaded that supply BODYxxx_GM kernel constant, typically
                    a .tpc that comes with the solar system DE you are using.
    :return: A function
    """
    return aTwoBody(gm=gdpool(f"BODY{spiceid}_GM",0,1)[0]*(1000**3)) # Stored value is in km**3/s**2,
                                                                 # convert to SI m**3/s**2


def aJ2(*,j2:float,gm:float,re:float):
    """
    Make a J2 function compatible with the Universe gravity model as an acc
    :param j2: J2 gravity coefficient
    :param gm: Two-body gravitational constant
    :param re: Effective equatorial radius, must correspond with the J2 value provided
    :return: A function which calculates J2 acceleration given position
    """
    def inner(*,t:float,y:np.ndarray,dt:float):
        """
        J2 gravity acceleration

        :param t: Simulation time in s, not used here but needed for interface
        :param y: Simulation state in m and s. Only position part y[:3] is used.
        :param dt: Simulation time step size, not used here but needed for interface
        """
        r=np.linalg.norm(y[:3])
        coef=-3*j2*gm*re**2/(2*r**5)
        rx,ry,rz=y[0:3]
        j2x=rx*(1-5*rz**2/r**2)
        j2y=ry*(1-5*rz**2/r**2)
        j2z=rz*(3-5*rz**2/r**2)
        return coef*np.array((j2x,j2y,j2z))
    return inner


def SpiceJ2(spiceid:int)->Callable[...,np.ndarray]:
    """
    Return a two-body gravity field based on Spice kernels
    :param spiceid: Integer spice ID of body in question. Must have kernels
                    loaded that supply BODYxxx_GM kernel constant, typically
                    a .tpc that comes with the solar system DE you are using.
                    Must also include
    :return: A function
    """
    return aJ2(gm=gdpool(f"BODY{spiceid}_GM",0,1)[0]*(1000**3), # Stored value is km**3/s**2, convert to SI m**3/s**2
               j2=gdpool(f"BODY{spiceid}_J2",0,1)[0],
               re=gdpool(f"BODY{spiceid}_RADII",0,3)[0]*1000)   # Stored value is km, convert to SI m


def SpiceThirdBody(*,spice_id_center:int=399,
                     spice_id_body:int,
                     spice_id_frame:str='J2000',
                     et0:float
                  )->Callable[...,np.ndarray]:
    """
    Return a function that calculates the differential acceleration
    between the spacecraft in question and the central body, due
    to a distant body exerting two-body gravity.

    :param spice_id_center: Spice ID of current dominant gravitational body (Earth
                            by default). The local coordinate frame is centered at
                            the center of mass of this body.
    :param spice_id_body: Spice ID of body causing perturbation
    :param spice_id_frame: Local coordinate frame is non-rotating and parallel
                           to this inertial frame
    :param et0: Spice ephemeris time of simulator t0.
    :return: A function that computes the pertutbation of the spacecraft due to the specified body.
             This function expects parameters: t (simulator time in seconds), y (spacecraft state vector),
             dt (time step, unused), and major_step (boolean flag, unused).
    """
    spice_id_center=str(spice_id_center)
    spice_id_body=str(spice_id_body)
    body_gm=gdpool(f"BODY{spice_id_body}_GM",0,1)[0]*(1000**3) # convert from stored km**3/s**2 to m**3/s**2
    def inner(*,t:float,y:np.ndarray,dt:float)->np.ndarray:
        """
        Compute the perturbation due to the third body on the spacecraft.

        This function calculates the difference in gravitational acceleration
        between the spacecraft and the central body towards the third body,
        effectively giving the tidal force experienced by the spacecraft in the
        frame of the central body.

        :param t: Simulator time, must be in seconds
        :param y: Spacecraft position and velocity relative to center of spice_id_center
        :param dt: Time step size, not used but needed for interface consistency
        :return: Tidal acceleration vector in the same frame as y[:3]
        """
        # State vector of third body relative to center body of local frame
        body_state=spkezr(spice_id_body,et0+t,spice_id_frame,'LT',spice_id_center)[0]*1000 # convert from km and km/s to m and m/s
        # direction of force, towards the other body
        body_r=body_state[:3]
        center_a=body_r*body_gm/np.linalg.norm(body_r)**3
        # Imagine a 1D universe where the spacecraft is 1 unit to the right and the third
        # body is 4 units. body_r would be +4, and the body would be 3 units to the right
        # of the spacecraft. Therefore the vector from the spacecraft to the third body
        # is body_state-sc_state.
        sc_r=body_state[:3]-y[:3]
        sc_a=sc_r*body_gm/np.linalg.norm(sc_r)**3
        a=sc_a-center_a
        return a
    return inner