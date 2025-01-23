"""
Model of a Titan-III/Centaur (Titan 3E)
"""
import numpy as np
from kwanmath.vector import vangle
from matplotlib import pyplot as plt
from spiceypy import pxform

from guidance.pitch_program import pitch_program
from rocket_sim.drag import INCHES, f_drag, mach_drag
from rocket_sim.gravity import SpiceTwoBody, SpiceJ2, SpiceThirdBody
from rocket_sim.planet import Earth, SpicePlanet
from rocket_sim.srm import SRM
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Stage, kg_per_lbm, Vehicle, Engine, N_per_lbf, g0
from vehicle.voyager import Voyager, init_spice, voyager_et0


class Titan3E(Vehicle):
    def __init__(self,*,tc_id:int):
        # SRMs are bog standard by-the-book.
        # mprop from total expended mass in UA1205 paper
        # F1 from initial sea level thrust lbf, table I.
        # F0 from peak of thrust curve from figure 7. Peak
        # thrust is in the startup transient.
        self.tc_id=tc_id
        self.vgr_id=tc_id-5
        self.srmL = SRM(mprop=197924, csv="data/UA1205 Thrust Curve.csv", name="Left SRM", F1F0=1199300 / 1293520,
                   nozzle_cos=np.cos(np.deg2rad(6)))
        self.srmR = SRM(mprop=197924, csv="data/UA1205 Thrust Curve.csv", name="Right SRM", F1F0=1199300 / 1293520,
                   nozzle_cos=np.cos(np.deg2rad(6)))
        self.srbL = Stage(prop=197924, dry=32447, name="Left SRB")
        self.srbR = Stage(prop=197924, dry=32447, name="Right SRB")
        # Stage 1 has by-the-book dry mass with prop residual added,
        # and prop masses and performance from the tc flight reports
        # Loaded oxidizer and fuel. Input value is in lbm
        self.mprop1_loaded={6:168446+89438,7:168482+88586}[tc_id]*kg_per_lbm
        # Residual. Document implies the engine shut down due to
        # oxidizer depletion (good to the last drop) with this
        # much fuel left, which will be accounted as dry mass
        self.mprop1_resid={6:898,7:174}[tc_id]*kg_per_lbm
        self.stage1=Stage(total=self.mprop1_loaded-self.mprop1_resid,dry=15000*kg_per_lbm+self.mprop1_resid,name=f"TC-{self.tc_id} stage 1")
        self.e1_thrust10={6:519484,7:538598}[tc_id]*N_per_lbf
        self.e1_ve0={6:301.49,7:302.54}[tc_id]*g0
        self.engine1=Engine(thrust10=self.e1_thrust10,ve0=self.e1_ve0,name=f"TC-{self.tc_id} stage 1 LR-87")
        # Stage 2 is done the same way as stage 1
        self.mprop2_loaded={6:43064+23883,7:42981+24098}[tc_id]*kg_per_lbm
        # Stage 2 residual on TC-6 (Voyager 1) was over a ton of
        # oxidizer due to an anomalous restriction in the oxidizer
        # flow. This almost caused the mission to fail, and would
        # have caused Voyager 2 to fail.
        self.mprop2_resid={6:2331,7:197}[tc_id]*kg_per_lbm
        self.stage2=Stage(total=self.mprop2_loaded-self.mprop2_resid,dry=6000*kg_per_lbm+self.mprop2_resid,name=f"TC-{self.tc_id} stage 2")
        # Because of the oxidizer restriction, the TC-6 stage 2
        # engine ran fuel-rich, lowering thrust and increasing
        # ve from nominal. The big performance hit was dragging
        # an extra ton of oxidizer all the way to cutoff. The
        # Centaur was able to make up the difference, but only
        # because it had propellant reserve for later in the launch
        # window and it launched right at window open.
        self.e2_thrust10={6:99622,7:102408}[tc_id]*N_per_lbf
        self.e2_ve0={6:319.46,7:319.05}[tc_id]*g0
        self.engine2=Engine(thrust10=self.e2_thrust10,ve0=self.e2_ve0,name=f"TC-{self.tc_id} stage 1 LR-91")
        self.interstage_and_shroud=Stage(total=10900*kg_per_lbm,prop=0,name="Interstage and Shroud")
        vgr=Voyager(vgr_id=self.vgr_id)
        self.centaur=vgr.stages[vgr.i_centaur]
        self.pm=vgr.stages[vgr.i_pm]
        self.sc=vgr.stages[vgr.i_mm]
        super().__init__(stages=[self.srbL,self.srbR,self.stage1,self.stage2,
                                self.interstage_and_shroud,self.centaur,self.pm,self.sc],
                        engines=[(self.srmL,0),(self.srmR,1)])


def plot_tlm(vehicle:Vehicle,tc_id:int,earth:SpicePlanet,
             launch_lat:float, launch_lon:float, deg:bool,
             launch_alt:float, launch_et0:float,drag_enabled:bool):
    ts=np.array([tlm_point.t for tlm_point in vehicle.tlm_points])
    states=np.array([tlm_point.y0 for tlm_point in vehicle.tlm_points]).T
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
    plt.plot(ts,alts/1000,label=f'alt {"with drag" if drag_enabled else "no drag"}')
    plt.legend()
    plt.subplot(2,2,2)
    plt.ylabel("Downrange/km")
    plt.plot(ts,downranges/1000,label=f'alt {"with drag" if drag_enabled else "no drag"}')
    plt.legend()
    plt.subplot(2,2,3)
    plt.ylabel("Alt/km vs Downrange/km")
    plt.plot(downranges/1000,alts/1000,label=f'traj {"with drag" if drag_enabled else "no drag"}')
    plt.axis('equal')
    plt.legend()
    plt.pause(0.1)


def ascend(*,vgr_id:int,drag_enabled:bool):
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
    sim.runto(t1=130.0)
    plot_tlm(titan3E, tc_id=titan3E.tc_id,earth=earth,
             launch_lat=pad41_lat,launch_lon=pad41_lon,deg=True,
             launch_alt=pad41_alt,launch_et0=et0,drag_enabled=drag_enabled)


def main():
    init_spice()
    ascend(vgr_id=1,drag_enabled=True)
    ascend(vgr_id=1,drag_enabled=False)
    plt.show()


if __name__=="__main__":
    main()


