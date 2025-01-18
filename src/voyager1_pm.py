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

import numpy as np
from bmw import elorb, Elorb
from matplotlib import pyplot as plt
from spiceypy import str2et, furnsh, etcal, gdpool

from rocket_sim.gravity import SpiceTwoBody, SpiceJ2
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Stage, Engine, Vehicle

EarthGM=None
EarthRe=None

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
                                    elorb=elorb(y[:3],y[3:],l_DU=EarthRe,mu=EarthGM,t0=t)))


class Voyager(Vehicle):
    def __init__(self,*,spice_id:int=-31):
        self.spice_id=spice_id
        lb_kg_conv = 0.45359237  # this many kg in 1 lb
        g0 = 9.80665  # Used to convert kgf to N
        lbf_N_conv = lb_kg_conv * g0  # This many N in 1 lbf

        # From The Voyager Spacecraft, Gold Medal Lecture in Mech Eng, table 2 bottom line
        # mm1 is mission module on Voyager 1, what was called the "Voyager 1 spacecraft" ever after. mm2 would be voyager 2.
        mm_mtot = 825.4
        mm_mprop = 103.4
        mm = Stage(prop=mm_mprop, total=mm_mtot)  # dry mass and RCS prop for Voyager
        # Value from TC-7 Voyager 2 Flight Data Report, p10
        mmpm_mtot = 4470 * lb_kg_conv
        # pm1 is the propulsion module on Voyager 1. pm2 would be on Voyager 2, in case both are in the same namespace
        pm_mtot = mmpm_mtot - mm_mtot
        # Values from AIAA79-1334 Voyager Prop System, table 5
        pm_mprop = 1045.9  # kg, SRM expelled mass
        pm = Stage(prop=pm_mprop, total=pm_mtot)
        self.t_pm0 = 3722.2  # PM start from TC-6 timeline
        self.t_pm1 = 3767.3  # PM burnout
        dt_pm1 = self.t_pm1 - self.t_pm0
        pm_Itot = 2895392  # Table 5, total impulse calculated from tracking data, N*s
        pm_ve = pm_Itot / pm_mprop  # Exhaust velocity, m/s
        pm_F = pm_Itot / dt_pm1  # Mean thrust assuming rectangular thrust curve
        pm_engine = Engine(pm_F, pm_ve)
        def guide(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
            # Prograde
            return y[3:]/np.linalg.norm(y[3:])
        super().__init__(stages=[pm,mm],engines=[(pm_engine,0)],extras=[tlm],guide=guide)
    def sequence(self, *, t: float, y: np.ndarray, dt: float):
        self.engines[0].throttle = 1 if self.t_pm0 < t < self.t_pm1 else 0


def main():
    # Furnish the kernels
    furnsh("data/naif0012.tls")
    furnsh("data/gravity_EGM2008_J2.tpc")
    global EarthRe, EarthGM
    EarthGM=gdpool("BODY399_GM",0,1)[0]*1000**3
    EarthRe=gdpool("BODY399_RADII",0,3)[0]*1000
    # Take first reliable position as 60s after first Horizons data point.
    # This line is that data, copied from data/v1_horizons_vectors_1s.txt line 146
    horizons_data1=[2443392.083615544,                # JDTDB
                   "A.D. 1977-Sep-05 14:00:24.3830", # Gregorian TDB
                   48.182579,                        # DeltaT -- TDB-UT
                   7.827005697472316E+03,            # X position, km, in Earth-centered frame parallel to ICRF (Spice J2000)
                  -4.769757853525854E+02,            # Y position, km
                  -5.362302110171012E+02,            # Z position, km
                   7.660453026119973E+00,            # X velocity, km/s
                   1.080867154837185E+01,            # Y velocity, km/s
                   5.581379825305540E+00,            # Z velocity, km/s
                   2.621760038208511E-02,            # Light time to geocenter, s
                   7.859838861407035E+03,            # Range from center, km
                   6.591742058881379E+00]            # Range rate from center, km/s
    # Time of above vector, considered to be t1
    horizons_et1=str2et(f"{horizons_data1[1]} TDB")
    # Launch timeline T0
    voyager1_et0=str2et(f"1977-09-05T12:56:00.958Z")
    horizons_t1=horizons_et1-voyager1_et0
    print(etcal(horizons_et1))
    print(etcal(voyager1_et0))
    print(horizons_t1)
    sim_t0=3600.03
    y1=np.array(horizons_data1[3:9])*1000.0 # Convert km to m and km/s to m/s
    vgr1=Voyager()
    earth_twobody=SpiceTwoBody(spiceid=399)
    earth_j2=SpiceJ2(spiceid=399)
    sim=Universe(vehicles=[vgr1],accs=[earth_twobody,earth_j2],t0=horizons_t1,y0s=[y1],fps=128)
    # Propellant tank "starts" out empty and fills up as time runs backwards
    vgr1.stages[0].prop_mass=0
    sim.runto(t1=sim_t0)
    ts = np.array([x.t for x in vgr1.tlm])
    states = np.array([x.y for x in vgr1.tlm])
    masses = np.array([x.mass for x in vgr1.tlm])
    thrusts = np.array([x.thrust for x in vgr1.tlm])
    accs=thrusts/masses
    elorbs=[x.elorb for x in vgr1.tlm]
    eccs=np.array([elorb.e for elorb in elorbs])
    incs=np.array([np.rad2deg(elorb.i) for elorb in elorbs])
    smis=np.array([elorb.a for elorb in elorbs])/1852 # Display in nmi to match document
    c3s=-(EarthGM/(1000**3))/(np.array([elorb.a for elorb in elorbs])/1000) # work directly in km
    plt.figure("spd")
    plt.plot(ts,np.linalg.norm(states[:, 3:6],axis=1), label='spd')
    plt.figure("ecc")
    plt.plot(ts,eccs,label='e')
    plt.figure("inc")
    plt.plot(ts,incs,label='i')
    plt.figure("a")
    plt.plot(ts,smis,label='a')
    plt.ylabel('semi-major axis/nmi')
    plt.figure("c3")
    plt.plot(ts,c3s,label='c3')
    plt.ylabel('$C_3$/(km**2/s**2)')
    plt.figure("mass")
    plt.plot(ts,masses,label='i')
    plt.show()


if __name__=="__main__":
    main()