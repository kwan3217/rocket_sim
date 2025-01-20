"""
Simulate the Propulsion Module burn of the Voyager 1 spacecraft.
The state after the burn is known much better than that before the burn,
since there is Horizons data seconds after (and even during, but I rate
that as unreliable) the burn. I don't have direct access to that kernel
but I do have a set of vectors from the first data through the next hour
at 1s intervals, and from first data through the end of the day at
60s intervals.

Voyager is documented to be commanded to hold a fixed inertial attitude
during the burn [[citation needed, but I have seen this]]. So to start with
we will aim the burn in the prograde direction from the post-burn state.

Created: 1/17/25
"""
from dataclasses import dataclass
from math import isclose
from typing import Callable

import numpy as np
from bmw import elorb, Elorb
from kwanmath.geodesy import xyz2llr, llr2xyz
from kwanmath.vector import vlength, vcross, vnormalize, vdot
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from spiceypy import str2et, furnsh, etcal, gdpool, sxform, pxform
from kwanspice.mkspk import mkspk

from rocket_sim.gravity import SpiceTwoBody, SpiceJ2, SpiceThirdBody
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Stage, Engine, Vehicle

# will be filled in by main() since this takes actual work IE loading a kernel
EarthGM=None
EarthRe=None
horizons_et1=None
voyager1_et0=None

# Take first reliable position as 60s after first Horizons data point.
# This line is that data, copied from data/v1_horizons_vectors_1s.txt line 146
horizons_data1 = [2443392.083615544,  # JDTDB
                  "A.D. 1977-Sep-05 14:00:24.3830",  # Gregorian TDB
                  48.182579,  # DeltaT -- TDB-UT
                  7.827005697472316E+03,  # X position, km, in Earth-centered frame parallel to ICRF (Spice J2000)
                 -4.769757853525854E+02,  # Y position, km
                 -5.362302110171012E+02,  # Z position, km
                  7.660453026119973E+00,  # X velocity, km/s
                  1.080867154837185E+01,  # Y velocity, km/s
                  5.581379825305540E+00,  # Z velocity, km/s
                  2.621760038208511E-02,  # Light time to geocenter, s
                  7.859838861407035E+03,  # Range from center, km
                  6.591742058881379E+00]  # Range rate from center, km/s
# Following is from TC-6 flight data report table 4-3, Guidance Telemetry, so what Centaur
# thought it hit.
sim_t0 = 3600.03 # Time of post-MECO2 (after Centaur burn but before PM burn) target
target_a = -4220.43 * 1852  # Initial coefficient is nautical miles, convert to meters
target_e = 1.84171
target_i = 28.5165 # degrees
target_lan = 170.237 # degrees
# Taken from data/v2_horizons_vectors.txt line 92. This one is on a 60s cadence so it's just the second record.
horizons_data2 = [2443376.148289155,      #JDTDB
                  "A.D. 1977-Aug-20 15:33:32.1830", # Gregorian TDB
                  48.182846,              #TDB-UT
                  7.513438354530380E+03,  # rx km
                 -1.311047210468180E+03,  # ry km
                  2.089389319809422E+03,  # rz km
                  5.920525058736983E+00,  # vx km/s
                  8.878861450139366E+00,  # vy km/s
                  9.461766135385785E+00,  # vz km/s
                  2.637818209251871E-02,  # LT s
                  7.907980047087768E+03,  # R  km
                  6.653052526114903E+00]  # RR km/s


class Voyager(Vehicle):
    def __init__(self,*,vgr_id:int=1):
        self.vgr_id=vgr_id
        self.spice_id=-30-vgr_id
        lb_kg_conv = 0.45359237  # this many kg in 1 lb
        g0 = 9.80665  # Used to convert kgf to N
        lbf_N_conv = lb_kg_conv * g0  # This many N in 1 lbf

        # From The Voyager Spacecraft, Gold Medal Lecture in Mech Eng, table 2 bottom line
        # mm is the mission module, what would be known as the "spacecraft" after Earth departure.
        self.mm_mtot = 825.4
        self.mm_mprop = 103.4
        mm = Stage(prop=self.mm_mprop, total=self.mm_mtot)  # dry mass and RCS prop for Voyager
        # Value from TC-7 Voyager 2 Flight Data Report, p10
        self.mmpm_mtot = 4470 * lb_kg_conv
        # pm is the propulsion module
        self.pm_mtot = self.mmpm_mtot - self.mm_mtot
        # Values from AIAA79-1334 Voyager Prop System, table 5
        self.pm_mprop = [None,1045.9,1046.0][vgr_id]   # kg, SRM expelled mass
        pm = Stage(prop=self.pm_mprop, total=self.pm_mtot)
        self.t_pm0 = [None,3722.2,3673.7][vgr_id]  # PM start from TC-6 timeline
        self.t_pm1 = [None,3767.3,3715.7][vgr_id]  # PM burnout
        dt_pm1 = self.t_pm1 - self.t_pm0
        self.pm_Itot = [None,2895392,2897042][vgr_id]  # Table 5, total impulse calculated from tracking data, N*s
        self.pm_ve = self.pm_Itot / self.pm_mprop  # Exhaust velocity, m/s
        self.pm_F = self.pm_Itot / dt_pm1  # Mean thrust assuming rectangular thrust curve
        pm_engine = Engine(self.pm_F, self.pm_ve)
        # All centaur stuff is for engine cn, burn bn
        # Voyager 1 TC-6 from Table 8-4, p87 of flight data report
        self.thrust_eb={}
        self.thrust_eb[(1,1)]=[None,14807,15033][vgr_id]*lbf_N_conv
        self.thrust_eb[(1,2)]=[None,14883,15166][vgr_id]*lbf_N_conv
        self.thrust_eb[(2,1)]=[None,15073,15200][vgr_id]*lbf_N_conv
        self.thrust_eb[(2,2)]=[None,15242,15460][vgr_id]*lbf_N_conv
        self.ve_eb={}
        self.ve_eb[(1,1)]=[None,441.5,441.8][vgr_id]*g0
        self.ve_eb[(1,2)]=[None,441.7,441.5][vgr_id]*g0
        self.ve_eb[(2,1)]=[None,441.1,442.0][vgr_id]*g0
        self.ve_eb[(2,2)]=[None,441.3,441.4][vgr_id]*g0
        # Mass mixture ratio - this many kg of oxidizer (LOX) is used for each kg of fuel (LH2)
        self.mr_eb={}
        self.mr_eb[(1,1)]=[None,4.90,5.08][vgr_id]
        self.mr_eb[(1,2)]=[None,4.86,5.00][vgr_id]
        self.mr_eb[(2,1)]=[None,5.03,4.98][vgr_id]
        self.mr_eb[(2,2)]=[None,4.97,5.06][vgr_id]
        # Centaur burn timing for centaur burn n start and end times (cbn[0|1])
        self.t_cb={}
        self.t_cb[(1,0)]=[None, 484.6, 478.7][vgr_id]
        self.t_cb[(1,1)]=[None, 594.0, 580.6][vgr_id]
        self.t_cb[(2,0)]=[None,3199.8,3148.5][vgr_id]
        self.t_cb[(2,1)]=[None,3535.3,3488.0][vgr_id]
        # Separation time of PM
        self.tsep_pm=[None,3705.2,3657.7][vgr_id]
        # Centaur residuals after burn 2, flight data report section p124 (V1) and p117 (V2)
        self.c_lox_resid=[None,276,374][vgr_id]*lb_kg_conv
        self.c_lh2_resid=[None, 36, 47][vgr_id]*lb_kg_conv
        # Calculated results below
        self.dt_cb={}
        self.mdot_eb={}
        self.mdotlox_eb={}
        self.mdotlh2_eb={}
        self.mprop_eb={}
        self.mlox_eb={}
        self.mlh2_eb={}
        self.mdot_b={}
        self.mdotlox_b={}
        self.mdotlh2_b={}
        self.mprop_b={}
        self.mlox_b={}
        self.mlh2_b={}
        for i_burn in (1,2):
            # Calculate each burn
            self.dt_cb[i_burn]=self.t_cb[(i_burn,1)]-self.t_cb[(i_burn,0)] # Burn time for i_burn
            for i_e in (1,2):
                # Calculate each engine in each burn
                eb=(i_e,i_burn)
                self.mdot_eb[eb]=self.thrust_eb[eb]/self.ve_eb[eb]  # Prop mass flow rate
                self.mdotlox_eb[eb]=self.mdot_eb[eb]*self.mr_eb[eb]/(1+self.mr_eb[eb]) # oxidizer flow rate
                self.mdotlh2_eb[eb]=self.mdot_eb[eb]*1             /(1+self.mr_eb[eb]) # fuel flow rate
                assert isclose(self.mdotlox_eb[eb]/self.mdotlh2_eb[eb],self.mr_eb[eb]),f"Mixture ratio for C-{i_e}, burn {i_burn} doesn't check out"
                self.mprop_eb[eb]=self.mdot_eb[eb]*self.dt_cb[i_burn]     # Propellant used
                self.mlox_eb[eb]=self.mdotlox_eb[eb]*self.dt_cb[i_burn]   # oxidizer used
                self.mlh2_eb[eb]=self.mdotlh2_eb[eb]*self.dt_cb[i_burn]   # fuel used
                assert isclose(self.mprop_eb[eb],self.mlox_eb[eb]+self.mlh2_eb[eb]),"Mixture ratio for c-1 doesn't check out"
                assert isclose(self.mlox_eb[eb]/self.mlh2_eb[eb],self.mr_eb[eb]),"Mixture ratio for c-1 doesn't check out"
            #   totals for burn
            self.mdot_b[i_burn]=self.mdot_eb[(1,i_burn)]+self.mdot_eb[(2,i_burn)]
            self.mdotlox_b[i_burn]=self.mdotlox_eb[(1,i_burn)]+self.mdotlox_eb[(1,i_burn)]
            self.mdotlh2_b[i_burn]=self.mdotlh2_eb[(1,i_burn)]+self.mdotlh2_eb[(1,i_burn)]
            self.mprop_b[i_burn] = self.mprop_eb[(1, i_burn)] + self.mprop_eb[(2, i_burn)]
            self.mlox_b[i_burn] = self.mlox_eb[(1, i_burn)] + self.mlox_eb[(1, i_burn)]
            self.mlh2_b[i_burn] = self.mlh2_eb[(1, i_burn)] + self.mlh2_eb[(1, i_burn)]
        # Total propellant in stage
        self.c_mlox=self.mlox_b[1]+self.mlox_b[2]+self.c_lox_resid
        self.c_mlh2=self.mlh2_b[1]+self.mlh2_b[2]+self.c_lh2_resid
        self.c_mprop=self.c_mlox+self.c_mlh2
        # Predict time to depletion of each component
        self.lox_t_depl=self.c_lox_resid/self.mdotlox_b[2]
        self.lh2_t_depl=self.c_lh2_resid/self.mdotlh2_b[2]
        # Check which is the limiting factor, and how much of the other would be left
        if self.lox_t_depl<self.lh2_t_depl:
            print("LOX is limiting factor")
            self.lox_depl_resid=0
            self.lh2_depl_resid=(self.lh2_t_depl-self.lox_t_depl)*self.mdotlh2_b[2]
            self.t_depl=self.lox_t_depl
        else:
            print("LH2 is limiting factor")
            self.lh2_depl_resid = 0
            self.lox_depl_resid = (self.lox_t_depl - self.lh2_t_depl) * self.mdotlox_b[2]
            self.t_depl = self.lh2_t_depl
        # Now build the engines and stage. We actually give it 4 engines, since we have
        # different stats for each burn.
        self.centaur=Stage(dry=4400*lb_kg_conv,prop=self.c_mprop)
        self.eb={eb:Engine(thrust10=thr,ve0=self.ve_eb[eb]) for eb,thr in self.thrust_eb.items()}
        super().__init__(stages=[self.centaur,pm,mm],
                         engines=[(pm_engine,1)]+[(engine,0) for eb,engine in self.eb.items()],
                         extras=[tlm])
        self.i_epm=0
        self.i_eb={1:(1,1),2:(1,2),3:(2,1),4:(2,2)}
    def sequence(self, *, t: float, y: np.ndarray, dt: float):
        # Sequence the PM engine
        self.engines[self.i_epm].throttle = 1 if self.t_pm0 <= t < self.t_pm1 else 0
        # Sequence the centaur engines
        for i_engine,(e,b) in self.i_eb.items():
            self.engines[i_engine]=1 if self.t_cb[b,0]<=t<self.t_cb[b,1] else 0
        self.stages[0].attached=(t<self.tsep_pm)


@dataclass
class TlmPoint:
    t:float
    y:np.ndarray
    mass:float
    thrust:float
    dir:np.ndarray
    elorb:Elorb


def tlm(*, t: float, dt: float, y: np.ndarray, major_step: bool, vehicle: Vehicle):
    if major_step:
        vehicle.tlm.append(TlmPoint(t=t,
                                    y=y.copy(),
                                    mass=vehicle.mass(),
                                    thrust=vehicle.thrust_mag(t=t, dt=dt, y=y, major_step=False),
                                    dir=vehicle.thrust_dir(t=t,dt=dt,y=y, major_step=False),
                                    elorb=elorb(y[:3].reshape(-1,1),y[3:].reshape(-1,1),l_DU=EarthRe,mu=EarthGM,t0=t)))


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
        aim_v = llr2xyz(lon=dyaw+yawrate*(t-t0), lat=dpitch, r=1)
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
                guides[i](t=t,y=y,dt=dt,major_step=major_step,vehicle=vehicle)
    return inner



def sim_pm_centaur(*,dpitch:float=0.0, dthr:float=0.0, dyaw:float=0.0, yawrate:float=0.0, fps:int=100, verbose:bool=True):
    horizons_t1 = horizons_et1 - voyager1_et0
    y1 = np.array(horizons_data1[3:9]) * 1000.0  # Convert km to m and km/s to m/s
    vgr1 = Voyager()
    vgr1.guide = seq_guide({vgr1.tsep_pm:prograde_guide,
                            float('inf'):yaw_rate_guide(r0=y1[:3], v0=y1[3:],
                                                        dpitch=dpitch, dyaw=dyaw, yawrate=yawrate,
                                                        t0=vgr1.t_pm0)})
    # Tweak engine efficiency and max thrust to hit same mdot
    vgr1.engines[0].ve0 *= 1.0 + dthr
    vgr1.engines[0].thrust10 *= 1.0 + dthr
    earth_twobody = SpiceTwoBody(spiceid=399)
    earth_j2 = SpiceJ2(spiceid=399)
    moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=voyager1_et0)
    sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=voyager1_et0)
    sim = Universe(vehicles=[vgr1], accs=[earth_twobody, earth_j2, moon, sun], t0=horizons_t1, y0s=[y1], fps=fps)
    # Propellant tank "starts" out empty and fills up as time runs backwards
    vgr1.stages[0].prop_mass = 0
    sim.runto(t1=sim_t0)
    if verbose:
        ts = np.array([x.t for x in vgr1.tlm])
        states = np.array([x.y for x in vgr1.tlm])
        masses = np.array([x.mass for x in vgr1.tlm])
        thrusts = np.array([x.thrust for x in vgr1.tlm])
        accs = thrusts / masses
        elorbs = [x.elorb for x in vgr1.tlm]
        eccs = np.array([elorb.e for elorb in elorbs])
        incs = np.array([np.rad2deg(elorb.i) for elorb in elorbs])
        smis = np.array([elorb.a for elorb in elorbs]) / 1852  # Display in nmi to match document
        c3s = -(EarthGM / (1000 ** 3)) / (np.array([elorb.a for elorb in elorbs]) / 1000)  # work directly in km
        plt.figure("spd")
        plt.plot(ts, np.linalg.norm(states[:, 3:6], axis=1), label='spd')
        plt.figure("ecc")
        plt.plot(ts, eccs, label='e')
        plt.figure("inc")
        plt.plot(ts, incs, label='i')
        plt.figure("a")
        plt.plot(ts, smis, label='a')
        plt.ylabel('semi-major axis/nmi')
        plt.figure("c3")
        plt.plot(ts, c3s, label='c3')
        plt.ylabel('$C_3$/(km**2/s**2)')
        plt.figure("mass")
        plt.plot(ts, masses, label='i')
        plt.show()
    return vgr1


def opt_interface_pm_burn(target:np.ndarray=None,verbose:bool=False, fps:int=100)->float:
    """
    Calculate the "cost" of a set of targeting parameters by
    walking back through the PM maneuver from a known state.
    Cost is related to difference of a, e, i from documented
    pre-PM state.
    :param target: Array containing difference in degrees between latitude
                   (index 0) and longitude (index 1) of target inertial
                   position away from prograde at Horizons data point.
                   (0,0) is a valid initial guess that results in prograde
                   at Horizons.
    :return: squared length of vector whose length is weighted error in a, e, and i.
             This is zero if the target is hit, and greater than zero for any miss.
    """
    dpitch, dyaw, dthr,yawrate = target
    vgr1 = sim_pm(dpitch=dpitch, dthr=dthr, dyaw=dyaw, fps=fps, verbose=verbose, yawrate=yawrate)
    # Initial Pre-PM orbit does not have complete orbit elements. It is reasonable
    # to believe that tracking was done in a frame aligned to the equator of date,
    # not J2000. We can't transform an incomplete orbit element set to J2000, so
    # we convert J2000 to IAU_EARTH. I have heard the warnings about the low precision
    # of IAU_EARTH but I don't have the tracking frame precise enough for the
    # precision to make a difference.
    # We use pxform() and not sxform() because we are transforming from the inertial
    # J2000 frame to an inertial frame parallel to IAU_EARTH at the instant in
    # question. We call this the "frozen frame" because it represents the rotating
    # frame but frozen in time so it no longer rotates. Function sxform() would
    # transform the velocity to within the rotating IAU_EARTH frame which would
    # screw up the orbit elements stuff which needs inertial velocity. Matrix is
    # Mej going to the IAU_EARTH (e) frozen frame from the J2000/ICRF frame (j).
    y0_j = vgr1.y.copy()
    Mej = pxform('J2000', 'IAU_EARTH', sim_t0 + voyager1_et0)
    r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
    v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
    r0_e = Mej @ r0_j
    v0_e = Mej @ v0_j
    # In this frame, the reported ascending node is Longitude of ascending node, not
    # right ascension. It is relative to the Earth-fixed coordinates at this instant,
    # not the sky coordinates.
    elorb0 = elorb(r0_e, v0_e, l_DU=EarthRe, mu=EarthGM, t0=sim_t0, deg=True)
    da = (target_a - elorb0.a) / EarthRe  # Use Earth radius to weight da
    de = (target_e - elorb0.e)
    di = np.deg2rad(target_i - elorb0.i)
    dlan = np.deg2rad(target_lan - elorb0.an)
    cost = (da ** 2 + de ** 2 + di ** 2 + 0 * dlan ** 2) * 1e6
    print(f"{dyaw=} deg, {yawrate=} deg/s, {dpitch=} deg, {dthr=}")
    print(f"   {da=} Earth radii")
    print(f"      ({da * EarthRe} m)")
    print(f"   {de=}")
    print(f"   {di=} rad")
    print(f"      ({np.rad2deg(di)} deg)")
    print(f"   {dlan=} rad")
    print(f"      ({np.rad2deg(dlan)} deg)")
    print(f"Cost: {cost}")
    return cost


def init():
    # Furnish the kernels
    furnsh("data/naif0012.tls")
    furnsh("data/pck00011.tpc")  # Sizes and orientations of all planets including Earth
    furnsh("data/gm_de440.tpc")  # Masses of all planets and many satellites, including Earth
    furnsh("data/gravity_EGM2008_J2.tpc") # Cover up Earth mass from gm_de440.tpc and size from pck00011.tpc
    furnsh("data/de440.bsp")     # Solar system ephemeris
    global EarthRe, EarthGM
    EarthGM=gdpool("BODY399_GM",0,1)[0]*1000**3
    EarthRe=gdpool("BODY399_RADII",0,3)[0]*1000
    # Time of Horizons vector, considered to be t1. Calculate
    # here because kernels aren't available up there
    global horizons_et1, voyager1_et0
    horizons_et1 = str2et(f"{horizons_data1[1]} TDB")
    # Launch timeline T0
    voyager1_et0 = str2et(f"1977-09-05T12:56:00.958Z")


def target():
    # Parameters are:
    #   0 - dpitch, pitch angle between pure prograde and target in degrees
    #   1 - dyaw, yaw angle between pure in-plane and target in degrees
    #   2 - dthr, fractional difference in engine efficiency. 0 is 100%,
    #       +0.1 is 110%, -0.1 is 90%, etc. This value changes thrust10 and ve
    #       simultaneously so as to not affect mdot, so the propellant
    #       will drain exactly as fast as before.
    #   3 - yawrate - change in yaw vs time in deg/s.
    #initial_guess=np.zeros(4)
    #initial_guess = [-1.4501264557665727,
    #                  0.0076685577917476625,
    #                 -0.001724033782731768,
    #                 0]  # Best known three-parameter fit for four targets
    #initial_guess=[-1.4501304910881003,
    #               -0.13110923799828175,
    #               -0.0017211606889325382,
    #                0.0] # Best known three-parameter fit for three targets
    initial_guess=[-1.4486738774680445,
                   -0.13109880981879152,
                   -0.0027462922256387273,
                   0.0] # Best known three-parameter fit at 100Hz
    bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(0,0)]  # Freeze yaw rate at 0
    #initial_guess = [-1.438, 13.801, +0.00599, -0.549]  # From previous four-parameter form
    #bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(-1,1)]  # Bounds on
    if False:
        result = minimize(opt_interface_pm_burn, initial_guess, method='L-BFGS-B', options={'ftol':1e-12,'gtol':1e-12,'disp':True}, bounds=bounds)
        print("Achieved cost:", result.fun)
        final_guess=result.x
    else:
        final_guess=initial_guess
    print("Optimal parameters:", final_guess)
    print("Optimal run: ")
    vgr1=sim_pm(**{k:v for k,v in zip(('dpitch','dyaw','dthr','yawrate'),final_guess)},verbose=False)
    states=sorted([np.hstack((np.array(x.t),x.y)) for x in vgr1.tlm],key=lambda x:x[0])
    decimated_states=[]
    i=0
    di=1
    done=False
    while not done:
        states[i][0]+=voyager1_et0
        decimated_states.append(states[i])
        try:
            if vgr1.t_pm0<states[i+di][0]<vgr1.t_pm1:
                di=1
            else:
                di=100
        except IndexError:
            break
        i+=di
    mkspk(oufn=f'data/vgr{vgr1.vgr_id}_pm.bsp',
          fmt=['f', '.3f', '.3f', '.3f', '.6f', '.6f', '.6f'],
          data=decimated_states,
          input_data_type='STATES',
          output_spk_type=5,
          object_id=vgr1.spice_id,
          object_name=f'VOYAGER {vgr1.vgr_id}',
          center_id=399,
          center_name='EARTH',
          ref_frame_name='J2000',
          producer_id='https://github.com/kwan3217/rocket_sim',
          data_order='EPOCH X Y Z VX VY VZ',
          input_data_units=('ANGLES=DEGREES', 'DISTANCES=m'),
          data_delimiter=' ',
          leapseconds_file='data/naif0012.tls',
          pck_file='data/gravity_EGM2008_J2.tpc',
          segment_id=f'VGR{vgr1.vgr_id}_PM',
          time_wrapper='# ETSECONDS',
          comment="""
           Best-estimate of trajectory through Propulsion Module burn,
           hitting historical post-MECO2 orbit elements which are
           available and matching Horizons data."""
          )


def export():
    with open("data/vgr1/v1_horizons_vectors_1s.txt","rt") as inf:
        states=[]
        for line in inf:
            line=line.strip()
            if line=="$$SOE":
                break
        for line in inf:
            line=line.strip()
            if line=="$$EOE":
                break
            jdtdb,gregoriantdb,deltat,rx,ry,rz,vx,vy,vz,lt,r,rr,_=[part.strip() for part in line.strip().split(",")]
            jdtdb,deltat,rx,ry,rz,vx,vy,vz,lt,r,rr=[float(x) for x in (jdtdb,deltat,rx,ry,rz,vx,vy,vz,lt,r,rr)]
            et=str2et(gregoriantdb+" TDB")
            states.append([et,rx,ry,rz,vx,vy,vz])
    gtdb_last_1s=gregoriantdb
    with open("data/vgr1/v1_horizons_vectors.txt","rt") as inf:
        states=[]
        for line in inf:
            line=line.strip()
            if line=="$$SOE":
                break
        for line in inf:
            line=line.strip()
            if line=="$$EOE":
                break
            jdtdb,gregoriantdb,deltat,rx,ry,rz,vx,vy,vz,lt,r,rr,_=[part.strip() for part in line.strip().split(",")]
            if gregoriantdb<gtdb_last_1s:
                continue
            jdtdb,deltat,rx,ry,rz,vx,vy,vz,lt,r,rr=[float(x) for x in (jdtdb,deltat,rx,ry,rz,vx,vy,vz,lt,r,rr)]
            et=str2et(gregoriantdb+" TDB")
            states.append([et,rx,ry,rz,vx,vy,vz])
    mkspk(oufn=f'data/vgr1_horizons_vectors.bsp',
          fmt='f',
          data=states,
          input_data_type='STATES',
          output_spk_type=5,
          object_id=-31,
          object_name=f'VOYAGER 1',
          center_id=399,
          center_name='EARTH',
          ref_frame_name='J2000',
          producer_id='https://github.com/kwan3217/rocket_sim',
          data_order='EPOCH X Y Z VX VY VZ',
          input_data_units=('ANGLES=DEGREES', 'DISTANCES=km'),
          data_delimiter=' ',
          leapseconds_file='data/naif0012.tls',
          pck_file='data/gravity_EGM2008_J2.tpc',
          segment_id=f'VGR1_HORIZONS_1S',
          time_wrapper='# ETSECONDS',
          comment="""
           Export of Horizons data calculated from a kernel they have but I don't,
           Voyager_1_ST+refit2022_m. This file is at 1 second intervals from beginning
           of available data to that time plus 1 hour, then at 1 minute intervals
           to the end of 1977-09-05. From there we _do_ have a supertrajectory kernel
           that covers the rest of the mission."""
          )



def main():
    init()
    vgr1 = Voyager(vgr_id=1)
    vgr2 = Voyager(vgr_id=2)
    #target()
    #export()


if __name__=="__main__":
    main()