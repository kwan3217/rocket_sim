"""
Describe purpose of this script here

Created: 1/19/25
"""
from dataclasses import dataclass
from math import isclose
from typing import Callable

import numpy as np
from bmw import Elorb, elorb
from kwanmath.geodesy import llr2xyz
from kwanmath.vector import vnormalize, vcross, vdot
from spiceypy import furnsh, gdpool, str2et

from rocket_sim.vehicle import Vehicle, Stage, Engine


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
        self.mm = Stage(prop=self.mm_mprop, total=self.mm_mtot,name=f"Voyager {vgr_id} Mission Module")  # dry mass and RCS prop for Voyager
        # Value from TC-7 Voyager 2 Flight Data Report, p10
        self.mmpm_mtot = 4470 * lb_kg_conv
        # pm is the propulsion module
        self.pm_mtot = self.mmpm_mtot - self.mm_mtot
        # Values from AIAA79-1334 Voyager Prop System, table 5
        self.pm_mprop = [None,1045.9,1046.0][vgr_id]   # kg, SRM expelled mass
        self.pm = Stage(prop=self.pm_mprop, total=self.pm_mtot,name=f"Voyager {vgr_id} Propulsion Module")
        self.t_pm0 = [None,3722.2,3673.7][vgr_id]  # PM start from TC-6 timeline
        self.t_pm1 = [None,3767.3,3715.7][vgr_id]  # PM burnout
        dt_pm1 = self.t_pm1 - self.t_pm0
        self.pm_Itot = [None,2895392,2897042][vgr_id]  # Table 5, total impulse calculated from tracking data, N*s
        self.pm_ve = self.pm_Itot / self.pm_mprop  # Exhaust velocity, m/s
        self.pm_F = self.pm_Itot / dt_pm1  # Mean thrust assuming rectangular thrust curve
        self.pm_engine = Engine(thrust10=self.pm_F, ve0=self.pm_ve,name="Propulsion Module TE-M-364-4")
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
        self.centaur=Stage(dry=4400*lb_kg_conv,prop=self.c_mprop,name=f"Centaur D-1T {vgr_id+5}")
        self.eb={eb:Engine(thrust10=thr,ve0=self.ve_eb[eb],name=f"Centaur RL-10 C-{eb[0]} for burn {eb[1]}") for eb,thr in self.thrust_eb.items()}
        super().__init__(stages=[self.centaur,self.pm,self.mm],
                         engines=[(self.pm_engine,1)]+[(engine,0) for eb,engine in self.eb.items()],
                         extras=[tlm])
        self.i_epm=0
        self.i_eb={1:(1,1),2:(1,2),3:(2,1),4:(2,2)}
    def sequence(self, *, t: float, y: np.ndarray, dt: float):
        # Sequence the PM engine
        if self.t_pm0 <= t < self.t_pm1:
            self.engines[self.i_epm].throttle = 1
        else:
            self.engines[self.i_epm].throttle = 0
        # Sequence the centaur engines
        for i_engine,(e,b) in self.i_eb.items():
            self.engines[i_engine].throttle=1 if self.t_cb[b,0]<=t<self.t_cb[b,1] else 0
        self.stages[0].attached=(t<self.tsep_pm)


@dataclass
class TlmPoint:
    t:float
    y:np.ndarray
    mass:float
    thrust:float
    dir:np.ndarray


def tlm(*, t: float, dt: float, y: np.ndarray, major_step: bool, vehicle: Vehicle):
    if major_step:
        vehicle.tlm.append(TlmPoint(t=t,
                                    y=y.copy(),
                                    mass=vehicle.mass(),
                                    thrust=vehicle.thrust_mag(t=t, dt=dt, y=y, major_step=False),
                                    dir=vehicle.thrust_dir(t=t,dt=dt,y=y, major_step=False)))


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


# Take first reliable position as 60s after first Horizons data point.
# This line is that data, copied from data/v1_horizons_vectors_1s.txt line 146
horizons_data={1:[2443392.083615544,  # JDTDB
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
                  6.591742058881379E+00], # Range rate from center, km/s
               # Taken from data/v2_horizons_vectors.txt line 92. This one is on a 60s cadence so it's just the second record.
               2:[2443376.148289155,      #JDTDB
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
                  6.653052526114903E+00]} # RR km/s
# Following is from TC-6 flight data report table 4-3, Guidance Telemetry, so what Centaur
# thought it hit.
sim_t0 = {1:3600.03} # Time of post-MECO2 (after Centaur burn but before PM burn) target
target_a = {1:-4220.43 * 1852}  # Initial coefficient is nautical miles, convert to meters
target_e = {1:1.84171}
target_i = {1:28.5165} # degrees
target_lan = {1:170.237} # degrees
# These are the millisecond-precision timestamps of the
# ignition of the SRBs on each launch vehicle, and represent the
# official T=0 for all of the timelines in the flight data reports.
voyager_cal0={1:"1977-09-05T12:56:00.958Z",
              2:"1977-08-20T14:29:44.256Z"}

EarthGM=None
EarthRe=None
# These hold the Spice ET of the Horizons data point for each mission
horizons_et={}
# These hold the Spice ET of the official T=0 for each mission
voyager_et0={}


def init_spice():
    # Furnish the kernels
    furnsh("data/naif0012.tls")
    furnsh("data/pck00011.tpc")  # Sizes and orientations of all planets including Earth
    furnsh("data/gm_de440.tpc")  # Masses of all planets and many satellites, including Earth
    furnsh("data/gravity_EGM2008_J2.tpc") # Cover up Earth mass from gm_de440.tpc and size from pck00011.tpc
    furnsh("data/de440.bsp")     # Solar system ephemeris
    global EarthGM,EarthRe
    EarthGM = gdpool("BODY399_GM", 0, 1)[0] * 1000 ** 3
    EarthRe = gdpool("BODY399_RADII", 0, 3)[0] * 1000
    global horizons_et, voyager_et0
    for k in (1,2):
        # Time of Horizons vector, considered to be t1. Calculate
        # here because kernels aren't available up there
        horizons_et[k]=str2et(f"{horizons_data[k][1]} TDB")
        # Launch timeline T0
        voyager_et0[k]=str2et(voyager_cal0[k])
