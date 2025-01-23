"""
Simulate a rocket in 3DoF. This means that the rocket has a definite position and velocity, but no attitude.

In fact, since the drag model is attitude dependent, we are going to include a limited amount of attitude,
but we are not going to simulate attitude kinematics or dynamics -- just something like "the rocket is
pointed exactly into the wind" or "the rocket is pointing exactly the direction guidance wants it to point"
"""
from dataclasses import dataclass

import numpy as np
from atmosphere.atmosphere import Atmosphere, SimpleEarthAtmosphere
from kwanmath.geodesy import xyz2lla, lla2xyz
from kwanmath.vector import vcross, vnormalize, vdot
from spiceypy import furnsh, kclear, gdpool, pxform


@dataclass
class Planet:
    atm:Atmosphere
    w0:float
    mu:float
    re:float
    f:float
    rp:float
    j2:float
    def __init__(self,*,atm:Atmosphere,w0:float,mu:float,re:float,f:float):
        """

        :param atm: Atmosphere model which takes altitude in m
        :param w0: Rotation rate of central body in rad/s
        :param mu: Gravitational constant in m**3/s**2
        :param re: Equatorial radius in m
        :param f:  Flattening ratio
        j2 is calculated from geometric flattening and rotation rate
        """
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
    def b2lla(self,rb:np.array,centric:bool=False,deg:bool=False):
        return xyz2lla(centric=centric,deg=deg,xyz=rb,re=self.re,rp=self.rp,east=True)
    def lla2b(self,lat_deg:float=None,lat_rad:float=None,
                     lon_deg:float=None,lon_rad:float=None,
                     alt:float=None,
                     centric:bool=False):
        return lla2xyz(centric=centric,lat_deg=lat_deg,lat_rad=lat_rad,
                                       lon_deg=lon_deg,lon_rad=lon_rad,alt=alt,re=self.re,rp=self.rp,east=True)
    def wind(self,rj:np.array):
        """
        Calculate the motion of the body-fixed frame relative to the inertial
        frame at a given point in the inertial frame. The atmosphere is
        relatively fixed compared to the ground (perfectly so where the wind
        speed is zero) so the velocity of the air relative to the reference
        frame is a vector in a plane parallel to the equator (so zero polar
        component) and a magnitude of hundreds or thousands of meters per
        second near the equator depending on the planet.

        :param rj: Position in the inertial reference frame
        :return: Inertial wind speed at this location

        For now, we consider the body to have its center at zero and its polar
        axis aligned with the Z axis. This is a small ~0.25deg error between
        the actual rotation axis in 1977 and J2000/ICRF.
        """
        return vcross(np.array([0,0,self.w0]),rj.reshape(-1))


class SpicePlanet(Planet):
    def __init__(self,*,spice_id:int,atm:Atmosphere,bf_frame:str=None):
        """
        Create a planet from its Spice information, along with some extra stuff.
        :param spice_id: Spice ID of planet. PCK files with this body radii,
                         rotation model, GM, and J2 must be loaded first.
        :param atm: Atmosphere model to use
        """
        re,_,rp=gdpool(f"BODY{spice_id}_RADII",0,3)*1000
        f=1-(rp/re)
        mu=gdpool(f"BODY{spice_id}_GM",0,1)[0]*1000**3
        atm=atm
        # First - I know that IAU_EARTH is not recommended. This
        # is a low-precision project, and I don't need the binary
        # Earth orientation kernels. Second, what we are concerned
        # with here is the rotation rate. The kernel constant array
        # BODYnnn_PM is the position of the prime meridian as a polynomial
        # with constant, linear, quadratic, etc coefficient (most
        # planets have zero for quadratic and above). The input is
        # time in days of 86,400 TDB seconds from the J2000 epoch,
        # 2000-01-01T12:00:00.000 TDB. self.w0 (omega-0) is in
        # radians per second, so we convert here.
        w0=np.deg2rad(gdpool(f"BODY{spice_id}_PM",0,3)[1])/86400.0
        j2=gdpool(f"BODY{spice_id}_J2",0,1)[0]
        super().__init__(atm=atm,w0=w0,mu=mu,re=re,f=f)
        self.j2=j2
        self.bf_frame=bf_frame
    def launchpad(self,*,lat:float,lon:float,alt:float,deg:bool,et:float):
        """
        Calculate the inertial state of a launch site, given its body-fixed
        geodetic coordinates
        :param lat: Latitude of launch site
        :param lon: Longitude of launch site
        :param alt: Altitude of launch site above ellipsoid
        :param deg: If True, lat and lon are in degrees, if False then radians
        :param et: Spice ET of launch, used to calculate the rotation from designated
                   body frame to J2000/ICRF, roughly mean equator and equinox at J2000
                   epoch, 2000-01-01T12:00:00 TDB.
        :return: State vector of launch site in inertial J2000/ICRF
        We perform the following inaccurate steps to bridge the fact that the equator
        precessed from 1977 to 2000:
        1. Transform LLA to XYZ in body frame
        2. Calculate matrix from body frame to J2000
        3. Rotate body frame vector to J2000
        4. Calculate wind vector in J2000, assuming pole is perfectly aligned to z axis
        """
        if deg:
            lat=np.deg2rad(lat)
            lon=np.deg2rad(lon)
        rb=lla2xyz(lat_rad=lat,lon_rad=lon,alt=alt,re=self.re,rp=self.rp,centric=False).reshape(-1,1)
        Mjb=pxform(self.bf_frame,'J2000',et)
        rj=(Mjb @ rb).reshape(-1)
        vj=self.wind(rj).reshape(-1)
        return np.hstack((rj,vj))
    def downrange_frame(self,rj:np.ndarray,azimuth:float,deg:bool=True):
        """
        Return a matrix which transforms a vector in the downrange frame to
        one in the inertial frame
        :param rj: Location in the inertial frame
        :param azimuth: range azimuth east of true north
        :param deg: If true, azimuth is in degrees, otherwise radians
        :return: Mjd -- A matrix which transforms the downrange frame to the inertial frame

        Downrange frame has:
          * qbar - First basis vector points downrange in the local horizon plane
          * hbar - Second basis vector points crossrange to the left, in the local
                   horizon plane. This is the north side of an easterly launch
          * rbar - Third basis is vertical

        This uses a planetocentric vertical, rather than the more accurate planetodetic
        vertical. It's pretty straightforward to do planetodetic but is quite a bit
        more computationally intensive.
        """
        if deg:
            azimuth=np.deg2rad(azimuth)
        rbar=vnormalize(rj.reshape(-1,1))
        zbar=np.array([[0],[0],[1]])
        ebar=vnormalize(vcross(zbar,rj))
        nbar=vnormalize(vcross(rbar,ebar))
        assert vdot(nbar,zbar)>0,"nbar screwed up"
        qbar= nbar*np.cos(azimuth)+ebar*np.sin(azimuth)
        hbar= nbar*np.sin(azimuth)-ebar*np.cos(azimuth)
        assert np.isclose(vdot(qbar,hbar),0),"qbar and hbar not perpendicular"
        assert np.isclose(vdot(qbar,rbar),0),"qbar and rbar not perpendicular"
        assert np.isclose(vdot(hbar,rbar),0),"hbar and rbar not perpendicular"
        Mjd=np.hstack((qbar,hbar,rbar))
        return Mjd


class Earth(SpicePlanet):
    def __init__(self,atm:Atmosphere=None,bf_frame:str='IAU_EARTH'):
        if atm is None:
            atm=SimpleEarthAtmosphere()
        super().__init__(spice_id=399,atm=atm,bf_frame=bf_frame)
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


