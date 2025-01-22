"""
Simulate a rocket in 3DoF. This means that the rocket has a definite position and velocity, but no attitude.

In fact, since the drag model is attitude dependent, we are going to include a limited amount of attitude,
but we are not going to simulate attitude kinematics or dynamics -- just something like "the rocket is
pointed exactly into the wind" or "the rocket is pointing exactly the direction guidance wants it to point"
"""
import numpy as np
from atmosphere.atmosphere import Atmosphere, SimpleEarthAtmosphere
from kwanmath.geodesy import aTwoBody, aJ2, xyz2lla, lla2xyz
from kwanmath.matrix import rot_z
from kwanmath.vector import vnormalize, vcross, vdot
from numpy.linalg import inv
from spiceypy import pxform, str2et, furnsh, kclear


class Planet:
    def __init__(self,*,M0_rb:np.array,atm:Atmosphere,w0:float,mu:float,re:float,f:float):
        """

        :param M0_rb: Matrix which transforms from planet body-fixed frame to global inertial frame. Can also be used
                      to transform between global inertial frame and an inertial frame frozen to match the reference
                      frame at particular time (good for doing gravity field stuff)
        :param atm: Atmosphere model which takes altitude in m
        :param w0: Rotation rate of central body in rad/s
        :param mu: Gravitational constant in m**3/s**2
        :param re: Equatorial radius in m
        :param f:  Flattening ratio
        """
        self.M0_rb=M0_rb
        self.atm=atm
        self.w0=w0
        self.mu=mu
        self.re=re
        self.f=f
        self.rp=self.re*(1-f)
        e=np.sqrt(2*f-f**2)
        ep=e/np.sqrt(1-e**2)
        twoq0=(1+3/ep**2)*np.arctan(ep)-3/ep
        cbar20=((4*w0**2*re**3*e**3)/(15*mu*twoq0)-e**2)/(3*np.sqrt(5))
        self.j2=-cbar20*np.sqrt(5)
    def M_rb(self,*,t_micros:int):
        M_fb=rot_z(t_micros*self.w0/1_000_000)
        return self.M0_rb @ M_fb
    def b2lla(self,rb:np.array,centric:bool=False,deg:bool=False):
        return xyz2lla(centric=centric,deg=deg,xyz=rb,re=self.re,rp=self.rp,east=True)
    def lla2b(self,lat_deg:float=None,lat_rad:float=None,
                     lon_deg:float=None,lon_rad:float=None,
                     alt:float=None,
                     centric:bool=False):
        return lla2xyz(centric=centric,lat_deg=lat_deg,lat_rad=lat_rad,
                                       lon_deg=lon_deg,lon_rad=lon_rad,alt=alt,re=self.re,rp=self.rp,east=True)
    def gravity(self,*,t_micros:int,x:np.array)->np.array:
        """
        Calculate gravitational acceleration at given state vector caused by this planet

        :param t_micros:
        :param x:
        :return: acceleration in m/s
        """
        M_fr=self.M_rb(t_micros=t_micros).T
        xf=np.vstack((M_fr@x[:3,0],M_fr@x[:]))
        return aTwoBody(xf,gm=self.mu)+aJ2(xf,j2=self.j2,gm=self.mu,re=self.re)
    def Ms_rb(self,*,t_micros:int)->np.array:
        """
        Return a matrix which transforms a state from body-fixed to reference,
        including velocity (IE it handles the "wind")

        :param t_micros:
        :param x:
        :return: 6x6 matrix that transforms position and velocity from body-fixed
                 frame to inertial frame, including boost from rotation of planet

        Note: You can use the inverse of this to do the inverse transformation, but
              this matrix is NOT orthonormal, so you have to do a proper inverse, not
              just a transpose.
        """
        axis=np.array([[0.0],[0.0],[self.w0]])
        M_rb=self.M_rb(t_micros=t_micros)
        zcross=np.array([[ 0.0,-1.0, 0.0],
                         [ 1.0, 0.0, 0.0],
                         [ 0.0, 0.0, 0.0]])
        N_rb=self.w0*zcross@M_rb
        return np.vstack((np.hstack((M_rb,np.zeros((3,3)))),
                          np.hstack((N_rb,M_rb))))


class Earth(Planet):
    @staticmethod
    def gha0():
        """
        Calculate the greenwich hour angle at one specific moment
        near the Voyager launch. We treat this as self-contained
        and only use Spice here.
        :return:
        """
        furnsh('/spice/leapseconds/naif0012.tls')
        #At 1977-09-21 00:00:00.0 UT1, the Greenwich Mean Sidereal Time was
        #23:59:01.2517, meaning that the origin point was almost directly
        #over the prime meridian. Let's see what we can make of this ourselves.
        # The official current definition of Earth Rotation Angle is:
        # theta(t_u)=360deg*(0.779_057_273_264+1.002_737_811_911_354_480*t_u)
        #where t_u is the difference in days between the time in question
        #and 2000-01-01 12:00:00 UT1 (Wiki documents the epoch as being at noon TDT
        #but that doesn't make sense.
        day0=2451545.0
        #We will use UTC as a stand-in for UT1. In early September 1977, DUT1 was about +0.26s,
        #so the error in using UTC is a fraction of an arcsecond.
        day1=2443407.5 #1977-09-21
        gmst_ref=(23+59/60+ 1.2517/3600)*15
        t_u=day1-day0
        theta=360*((0.779_057_273_264+1.002_737_811_911_354_480*t_u)%1)
        print(f"{gmst_ref=},{theta=},{gmst_ref-theta=}")
        kclear()
    def __init__(self,et0:float=None):
        super().__init__(M0_rb=pxform("IAU_EARTH","J2000",et0) if et0 is not None else np.identity(3),
                         atm=SimpleEarthAtmosphere(),
                         w0=72.92115e-6,mu=3.986004418e14,re=6378137.0,f=1.0/298.257223563)


