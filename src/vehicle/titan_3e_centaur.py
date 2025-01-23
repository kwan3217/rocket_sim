"""
Model of a Titan-III/Centaur (Titan 3E)
"""
import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.srm import SRM
from rocket_sim.vehicle import Stage, kg_per_lbm, Vehicle, Engine, N_per_lbf, g0
from vehicle.voyager import Voyager, init_spice


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



