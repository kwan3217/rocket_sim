"""
Calculate drag force on a vehicle

Created: 1/22/25
"""
from typing import Callable

import numpy as np
from kwanmath.vector import vlength
from scipy.interpolate import interp1d, RectBivariateSpline

from rocket_sim.planet import Planet
from rocket_sim.vehicle import Vehicle

# Table 5-1. Coefficient of the Q-dependent component of total vehicle axial force [C_A]
#Mach number data points for axial coefficients

#axial coefficient, from table 3-1 **C_A of total vehicle
# Make the 1D table into a function f(mach)
CaMach=np.array([
   0.00, 0.25, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 1.00, 1.05, 1.10, 1.15, 1.25, 1.40, 1.50, 1.75, 2.00, 2.50, 3.50, 4.50, 6.00, 8.00,10.00
])
atlas_booster_Ca0=interp1d(CaMach,np.array([
  0.373,0.347,0.345,0.350,0.365,0.391,0.425,0.481,0.565,0.610,0.725,0.760,0.773,0.770,0.740,0.665,0.622,0.530,0.459,0.374,0.303,0.273,0.259,0.267,0.289
]),kind='linear')


# Table 5-2. Components of total vehicle normal force coefficient
#Mach number data points for normal coefficients
CnMach=np.array([
  0.00, 0.20, 0.50, 0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.15, 1.20, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.00, 5.00, 7.00, 10.00
])
# Cn0 is the normal force at zero angle of attack, due to vehicle asymmetry.
# Other, smoother, vehicles might want to just use 0 for this.
# Make the 1D table into a function f(mach)
atlas_booster_Cn0=interp1d(CnMach,np.array([
  0.0052,0.0052,0.0052,0.0100,0.0085,0.0079,0.0062,0.0057,0.0053,0.0051,0.0050,0.0050,0.0055,0.0072,0.0063,0.0059,0.0060,0.0062,0.0064,0.0060,0.0050,0.0036,0.0030
]),kind='linear')


#Alpha data points for angle-of-attack dependence
CnAlpha=np.array([
  0,	2,	4,	6,	8,	60,	90
])
#Derivative of normal force with respect to angle of attack in degrees. List of lists, first index is alpha index, second is mach.
# Reformat into a scipy interpolator CnStarOverAlpha(alpha,mach)
atlas_booster_CnStarOverAlpha=RectBivariateSpline(CnAlpha, CnMach, np.array([
  [0.0556,0.0556,0.0557,0.0564,0.0586,0.0650,0.0692,0.0743,0.0731,0.0677,0.0678,0.0672,0.0690,0.0733,0.0732,0.0720,0.0736,0.0769,0.0789,0.0795,0.0803,0.0797,0.0743],
  [0.0556,0.0556,0.0557,0.0566,0.0593,0.0660,0.0709,0.0754,0.0741,0.0699,0.0699,0.0691,0.0706,0.0741,0.0741,0.0790,0.0746,0.0787,0.0819,0.0803,0.0816,0.0811,0.0756],
  [0.0556,0.0556,0.0559,0.0606,0.0643,0.0702,0.0770,0.0795,0.0781,0.0754,0.0743,0.0740,0.0744,0.0779,0.0785,0.0784,0.0796,0.0823,0.0843,0.0851,0.0869,0.0865,0.0810],
  [0.0556,0.0556,0.0581,0.0658,0.0706,0.0760,0.0812,0.0846,0.0838,0.0817,0.0810,0.0806,0.0811,0.0843,0.0859,0.0861,0.0869,0.0909,0.0932,0.0942,0.0955,0.0950,0.0894],
  [0.0556,0.0556,0.0625,0.0718,0.0774,0.0817,0.0899,0.0909,0.0897,0.0882,0.0874,0.0872,0.0894,0.0947,0.0970,0.0979,0.1002,0.1047,0.1086,0.1101,0.1120,0.1110,0.1049],
  [0.1348,0.1348,0.1486,0.1866,0.2179,0.2506,0.2699,0.2878,0.3008,0.3057,0.2990,0.2872,0.2676,0.2605,0.2549,0.2517,0.2461,0.2413,0.2391,0.2358,0.2348,0.2308,0.2261],
  [0.1198,0.1198,0.1320,0.1658,0.1938,0.2241,0.2423,0.2595,0.2704,0.2714,0.2648,0.2551,0.2374,0.2312,0.2265,0.2229,0.2183,0.2131,0.2105,0.2095,0.2089,0.2077,0.2065]
]), kx=1,ky=1)

atlas_sustainer_CnStarOverAlpha=interp1d(CnAlpha, np.array([
  0.0743,0.0756,0.0810,0.0894,0.1049,0.2261,0.2065
]), kind='linear')


def atlas_Cn0(*,M:float,booster_attached:bool,sustainer_attached:bool)->float:
    if booster_attached:
        return atlas_booster_Cn0(M)
    elif sustainer_attached:
        # Sustainer phase, after booster drop. Valid at any mach above M=8.
        return 0.0030
    else:
        # Centaur phase, no lift
        return 0


def atlas_CnStarOverAlpha(*,M:float,beta_deg:float,booster_attached:bool,sustainer_attached:bool)->float:
    if booster_attached:
        return atlas_booster_CnStarOverAlpha(beta_deg,M)
    elif sustainer_attached:
        return atlas_sustainer_CnStarOverAlpha(beta_deg)
    else:
        # Centaur phase, no lift
        return 0


def atlas_Ca0(*,M:float,booster_attached:bool,sustainer_attached:bool)->float:
    if booster_attached:
        return atlas_booster_Ca0(M)
    elif sustainer_attached:
        # p45 of report
        # By the time the vehicle drops the booster, it's going fast enough that coefficients are not functions of mach
        return 0.272
    else:
        # By the time the sustainer drops, we should be out of the atmosphere and this should hardly matter
        return 2.200


def atlas_drag(*,Cn0:Callable[...,float]=atlas_Cn0,
                 CnStarOverAlpha:Callable[...,float]=atlas_CnStarOverAlpha,
                 Ca0:Callable[...,float]=atlas_Ca0)->Callable[...,tuple[float,float]]:
    """
    Return a function that calculates the lift and drag coefficients
    given the mach and angle of attack, in the form that the Atlas-Centaur
    SLV-3D drag coefficients are calculated
    :param Cn0: Function which calculates the normal coefficient at zero angle of attack.
                Atlas had lots of protruberances such that it did have lift at zero angle
                of attack. Titan is much smoother so the default is zero. Function should
                have prototype like this:
                def Cn0(*,M:float,**kwargs)->float
                    ...
                    return cn0
    :param CnStarOverAlpha: Function which calculates the derivative of the normal coefficient
                            with respect to angle of attack. Function should have a prototype like this:
                def CnStarOverAlpha(*,beta_deg:float,**kwargs)->float
                    ...
                    return cnStarOverAlpha
    :param Ca0: Function which calculates the axial force given the Mach number. Function should have
                a prototype like this:
                def Ca0(*,M:float,**kwargs)->float
                    ...
                    return ca0
    :return: Function which calculates lift and drag coefficient given mach and angle of attack
             as if it was declared thusly:
             def f(*,M:float,beta_rad:float,**kwargs)->tuple[float,float]:
                 ...
                 return cl,cd
    """
    def inner(*,M:float,beta_rad:float,vehicle:Vehicle)->tuple[float,float]:
        booster_attached=True # This should be pulled from the right vehicle.stages[].attached
        sustainer_attached=True # ibid
        beta_deg=np.rad2deg(beta_rad)
        cn0=Cn0(M=M,booster_attached=booster_attached,sustainer_attached=sustainer_attached)
        cnStarOverAlpha=CnStarOverAlpha(beta_deg=beta_deg,M=M,booster_attached=booster_attached,sustainer_attached=sustainer_attached)
        Cn=cn0+beta_deg*cnStarOverAlpha
        Ca=Ca0(M=M,booster_attached=booster_attached,sustainer_attached=sustainer_attached)
        cl = Cn*np.cos(beta_rad)-Ca*np.sin(beta_rad)
        cd = Cn*np.sin(beta_rad)+Ca*np.cos(beta_rad)
        return cl,cd
    return inner


def mach_drag(*,Ca0:Callable[[float],float]=atlas_booster_Ca0)->Callable[...,tuple[float,float]]:
    """
    Generate a pure mach-driven drag model. This model generates no lift, even in nonzero angle of attack.
    :param Ca0: Function which calculates the axial coefficient as a function of mach. It doesn't
                get any other arguments, so scipy interp1d is usable here.
    :return:
    """
    def inner(*,M:float,beta_rad:float,vehicle:Vehicle)->tuple[float,float]:
        return 0,Ca0(M)
    return inner


def f_drag(*,planet:Planet,clcd:Callable[...,tuple[float,float]],Sref:float)->Callable[...,np.ndarray]:
    """
    :param Sref: Reference area. Typical values are the cross section area of a cylindrical rocket.
    :param clcd: Callable that returns a tuple of lift and drag coefficients
    :param planet: Planet object that holds the ellipsoid model for getting altitude, and the atmosphere model
                   for getting density.
    :return: A function as if it had the following prototype, suitable for being an element of forces=[] for
             a Universe
             def force(*,t:float,dt:float,y:np.ndarray,vehicle:Vehicle)->np.ndarray:
                 ...
    """
    def inner(*,t:float,dt:float,y:np.ndarray,vehicle:Vehicle)->np.ndarray:
        alt=planet.b2lla(rb=y[:3]).alt
        air_props=planet.atm.calc_props(alt)
        rho=air_props.Density
        wind=planet.wind(y[:3]).reshape(-1)
        vrel=y[3:]-wind
        vrel_mag=vlength(vrel)
        if vrel_mag==0:
            # Early exit, plus avoid divide by zero when computing vrel direction on a zero vrel
            return np.zeros(3)
        M=vrel_mag/air_props.VSound
        qbar=rho*vrel_mag**2/2
        cl,cd=clcd(M=M,beta_rad=0.0,vehicle=vehicle)
        d=-vrel/vrel_mag*qbar*Sref*cd
        return d
    return inner


# Other coefficients that I carefully typed up but haven't used yet:
# Aerodynamics tables





#Cm0 is the moment at zero angle of attack
Cm0=np.array([
  -0.0060,-0.0060,-0.0068,-0.0088,-0.0074,-0.0060,-0.0060,-0.0062,-0.0065,-0.0070,-0.0065,-0.0058,-0.0028,-0.0083,-0.0060,-0.0036,-0.0032,-0.0040,-0.0043,-0.0040,-0.0030,-0.0020,-0.0020
])


# Sustainer phase, after booster drop. Valid at any mach above M=8.
Cn0Sust=0.0030

# Coefficient for pitching moment
XcpLref=[
  [0.1492,0.1493,0.1498,0.1632,0.1917,0.2114,0.2174,0.2169,0.1956,0.1084,0.1063,0.1089,0.1416,0.1540,0.1756,0.1800,0.1804,0.1847,0.1983,0.2231,0.2586,0.2830,0.2941],
  [0.1718,0.1719,0.1724,0.2125,0.2411,0.2401,0.2382,0.2477,0.2470,0.2227,0.1948,0.1881,0.1810,0.1703,0.1628,0.1637,0.1685,0.1801,0.2114,0.2154,0.2402,0.2632,0.2741],
  [0.1718,0.1689,0.1784,0.2235,0.2381,0.2485,0.2546,0.2623,0.2600,0.2395,0.2257,0.2203,0.2094,0.1997,0.1904,0.1839,0.1861,0.1969,0.2101,0.2256,0.2504,0.2735,0.2845],
  [0.1718,0.1719,0.1987,0.2433,0.2574,0.2661,0.2685,0.2686,0.2675,0.2465,0.2389,0.2337,0.2265,0.2203,0.2132,0.2053,0.2043,0.1942,0.2326,0.2534,0.2766,0.2975,0.3056],
  [0.1718,0.1719,0.2158,0.2552,0.2678,0.2749,0.2773,0.2756,0.2724,0.2534,0.2468,0.2421,0.2439,0.2422,0.2357,0.2338,0.2364,0.2481,0.2720,0.2912,0.3130,0.3308,0.3363],
  [0.3846,0.3846,0.3829,0.3794,0.3775,0.3764,0.3761,0.3762,0.3772,0.3822,0.3830,0.3841,0.3843,0.3838,0.3839,0.3838,0.3840,0.3838,0.3836,0.3834,0.3836,0.3837,0.3826],
  [0.4038,0.4038,0.4021,0.3989,0.3977,0.3981,0.4003,0.4022,0.4022,0.4022,0.4024,0.4027,0.4027,0.4027,0.4027,0.4027,0.4028,0.4028,0.4028,0.4028,0.4027,0.4024,0.4015]
]

INCHES=0.0254 # convert measurements in inches into meters
Sref=(12*12*78.5)*INCHES*INCHES #Reference area, roughly equal to 10' diameter circle but exactly as specified in the document on p44
Lref=1500*INCHES # Reference length, roughly equal to vehicle total length but exactly as specified on p
