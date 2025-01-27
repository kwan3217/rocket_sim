"""
Model of a Titan-III/Centaur (Titan 3E)
"""
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.srm import SRM
from rocket_sim.vehicle import Stage, kg_per_lbm, Vehicle, Engine, N_per_lbf, g0


class Titan3E(Vehicle):
    def __init__(self,*,tc_id:int):
        # SRMs are bog standard by-the-book.
        # mprop from total expended mass in UA1205 paper
        # F1 from initial sea level thrust lbf, table I.
        # F0 from peak of thrust curve from figure 7. Peak
        # thrust is in the startup transient.
        self.tc_id=tc_id
        self.vgr_id=tc_id-5
        self.spice_id=-30-self.vgr_id
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
        self.stage1=Stage(prop=self.mprop1_loaded-self.mprop1_resid,
                          dry=15000*kg_per_lbm+self.mprop1_resid,
                          name=f"TC-{self.tc_id} stage 1")
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
        # We estimate the deployable mass of the shroud to be 6000lb
        # and the interstage and boattali which remain attached to the
        # second stage to be 10900lb-6000lb=4900lb
        self.stage2=Stage(prop=self.mprop2_loaded-self.mprop2_resid,
                          dry=6000*kg_per_lbm+self.mprop2_resid+(10900-6000)*kg_per_lbm, #Tank mass plus residual plus interstage/boattail
                          name=f"TC-{self.tc_id} stage 2")
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
        # We estimate the deployable mass of the shroud to be 6000lb
        # and the interstage and boattali which remain attached to the
        # second stage to be 10900lb-6000lb=4900lb
        self.shroud=Stage(total=6000*kg_per_lbm,prop=0,name="Centaur Standard Shroud")

        # Centaur - We will split this into two stages, for burn 1 and burn 2.
        # Stage for burn 1 will have the exact amount of prop needed to provide
        # the calculated burn 1 total impulse, based on thrust, ve, and burn
        # duration. It will have zero dry mass. Stage for burn 2 will have the
        # exact amount of prop needed for the impulse for burn 2, and the dry
        # mass will include the residuals and the standard jettison weight.
        # The Centaur will be modeled as one engine with the combined thrust
        # and mass flow rate to be equivalent to the two engines, but will
        # have a separate engine for burn 1 and 2. For all of the dictionaries,
        # they are indexed by TC number and if there is a tuple, then the first
        # element is for engine 1 and the second is engine 2.
        # -- Burn 1 --
        @dataclass
        class CentaurProps:
            """
            Encapsulate stuff into a dataclass so that it is repeatable
            and readable
            """
            #     Thrust in lbs from Voyager 1 TC-6 from Table 8-4, p87 of flight data report
            F_lb:tuple[float,float]
            #     Isp in seconds
            Isp:tuple[float,float]
            #     Mass mixture ratio - this many kg of oxidizer (LOX) is used for each kg of fuel (LH2)
            mr:tuple[float,float]
            #     Centaur burn timing
            t:tuple[float,float]
            def __init__(self,F_lb,Isp,mr,t):
                self.F_lb=F_lb
                self.Isp=Isp,
                self.mr=np.array(mr)
                self.t=t
                #     Calculate mdot for each engine
                self.mdot=np.array([F_lb*N_per_lbf/(Isp*g0) for F_lb,Isp in zip(F_lb,Isp)])
                #     Use mixture ratio to figure mdot of each propellant for each engine
                self.mdotlox=self.mdot*self.mr/(1+self.mr)
                self.mdotlh2=self.mdot*      1/(1+self.mr)
                #     Sum the mdots
                self.mdot   =np.sum(self.mdot)
                self.mdotlox=np.sum(self.mdotlox)
                self.mdotlh2=np.sum(self.mdotlh2)
                #     Equivalent total engine performance
                self.F=np.sum([f_lb*N_per_lbf for f_lb in F_lb])
                self.ve=self.F/self.mdot
                #     Engine burn time
                self.dt=t[1]-t[0]
                #     Propellants consumed
                self.mprop=self.mdot*self.dt
                self.mlox=self.mdotlox*self.dt
                self.mlh2=self.mdotlh2*self.dt
        self.centaur1_props=CentaurProps(F_lb={6:(14807,15073),7:(15033,15200)}[tc_id],
                              Isp ={6:(441.5,441.1),7:(441.8,442.0)}[tc_id],
                              mr  ={6:(4.90,5.03),7:(5.08,4.98)}[tc_id],
                              t   ={6:(484.6,594.0),7:(478.7,580.6)}[tc_id])
        self.centaur2_props=CentaurProps(F_lb={6:(14883,15242),7:(15166,15460)}[tc_id],
                              Isp ={6:(441.7,441.3),7:(441.5,441.4)}[tc_id],
                              mr  ={6:(4.86,4.97),7:(5.00,5.06)}[tc_id],
                              t   ={6:(3199.8,3535.3),7:(3148.5,3488.0)}[tc_id])
        # Centaur residuals after burn 2, flight data report section p124 (V1) and p117 (V2)
        self.c_lox_resid= {6:276,7:374}[tc_id] * kg_per_lbm
        self.c_lh2_resid= {6:36,7:47}[tc_id] * kg_per_lbm
        self.c_resid=self.c_lox_resid+self.c_lh2_resid
        # Predict time to depletion of each component
        self.lox_t_depl=self.c_lox_resid/self.centaur2_props.mdotlox
        self.lh2_t_depl=self.c_lh2_resid/self.centaur2_props.mdotlh2
        # Check which is the limiting factor, and how much of the other would be left
        if self.lox_t_depl<self.lh2_t_depl:
            print("LOX is limiting factor")
            self.lox_depl_resid=0
            self.lh2_depl_resid=(self.lh2_t_depl-self.lox_t_depl)*self.centaur2_props.mdotlh2
            self.t_depl=self.lox_t_depl
        else:
            print("LH2 is limiting factor")
            self.lh2_depl_resid=0
            self.lox_depl_resid=(self.lox_t_depl-self.lh2_t_depl)*self.centaur2_props.mdotlox
            self.t_depl = self.lh2_t_depl
        print(f"Time to depletion: {self.t_depl:.2f}")
        # Now build the engines and stage. The centaur has two engines, BUT
        # one virtual is equivalent to the two real engines firing during
        # burn 1, while the other virtual engine is equivalent to the two
        # real engines firing in burn 2.
        self.cengine1=Engine(thrust10=self.centaur1_props.F,ve0=self.centaur1_props.ve,name=f"Centaur 2x RL-10 engines for burn 1")
        self.cengine2=Engine(thrust10=self.centaur2_props.F,ve0=self.centaur2_props.ve,name=f"Centaur 2x RL-10 engines for burn 2")
        # The two virtual engines both share the same tank, whose dry mass
        # is the 4400lb documented jettison mass, plus the residual, and
        # whose prop load is the sum of the burned prop from each burn
        self.centaur=Stage(dry=4400 * kg_per_lbm + self.c_resid,
                           prop=self.centaur1_props.mprop+self.centaur2_props.mprop,name="Centaur D-1T {tc_id}")

        # From The Voyager Spacecraft, Gold Medal Lecture in Mech Eng, table 2 bottom line
        # mm is the mission module, what would be known as the "spacecraft" after Earth departure.
        self.mm_mtot = 825.4
        self.mm_mprop = 103.4
        self.mm = Stage(prop=self.mm_mprop, total=self.mm_mtot,name=f"Voyager {self.vgr_id} Mission Module")  # dry mass and RCS prop for Voyager
        # Value from TC-7 Voyager 2 Flight Data Report, p10
        self.mmpm_mtot = 4470 * kg_per_lbm
        # pm is the propulsion module
        self.pm_mtot = self.mmpm_mtot - self.mm_mtot
        # Values from AIAA79-1334 Voyager Prop System, table 5
        self.pm_mprop = {1:1045.9,2:1046.0}[self.vgr_id]   # kg, SRM expelled mass
        self.pm = Stage(prop=self.pm_mprop, total=self.pm_mtot,name=f"Voyager {self.vgr_id} Propulsion Module")
        # From Table 4-1 in TC-[67] Flight Data Report
        self.t_pm={6:(3722.2,3767.3),7:(3673.7,3715.7)}[tc_id]
        self.dt_pm1 = self.t_pm[1] - self.t_pm[0]
        self.pm_Itot = {1:2895392,2:2897042}[self.vgr_id]  # Table 5, total impulse calculated from tracking data, N*s
        self.pm_ve = self.pm_Itot / self.pm_mprop  # Exhaust velocity, m/s
        self.pm_F = self.pm_Itot / self.dt_pm1  # Mean thrust assuming rectangular thrust curve
        self.pmengine = Engine(thrust10=self.pm_F, ve0=self.pm_ve,name="Propulsion Module TE-M-364-4")

        # Stages are ordered by drop order
        self.i_srbs=(0,1)
        self.i_stage1=2
        self.i_shroud=3
        self.i_stage2=4
        self.i_centaur=5
        self.i_pm=6
        self.i_mm=7
        self.i_srms=(0,1)
        self.i_engine1=2
        self.i_engine2=3
        self.i_cengine1=4
        self.i_cengine2=5
        self.i_pmengine=6
        # Jettison times of stages, from Table 4-1 of each flight data report
        self.tdrop=[{6: 123.2,7: 121.7}[tc_id]]*2+[  # 0,1 - Both SRBs drop at the same time
                    {6: 262.2,7: 255.2}[tc_id],      #   2 - stage 1
                    {6: 272.2,7: 265.2}[tc_id],      #   3 - shroud
                    {6: 474.1,7: 468.2}[tc_id],      #   4 - stage 2
                    {6:3705.2,7:3657.7}[tc_id],      #   5 - centaur
                    {6:4445.0,7:4396.7}[tc_id]]      #   6 - prop module
        # Burn times of engines, from Table 4-1
        self.tburn= [(0, {6:123.2, 7:121.7}[tc_id])] * 2 + [   # 0,1 - Both SRMs burn together
                      {6:(112.1,262.2),7:(110.7,254.5)}[tc_id],   #   2 - stage 1 engine
                      {6:(262.2,469.9),7:(255.2,462.0)}[tc_id],   #   3 - stage 2 engine
                      self.centaur1_props.t,      #   4 - centaur burn 1
                      self.centaur2_props.t,      #   5 - centaur burn 2
                      self.t_pm] #   6 - prop module
        super().__init__(stages=[self.srbL,self.srbR,self.stage1,self.shroud,
                                 self.stage2,self.centaur,self.pm,self.mm],
                        engines=[(self.srmL,self.i_srbs[0]),(self.srmR,self.i_srbs[1]),
                                 (self.engine1,self.i_stage1),(self.engine2,self.i_stage2),
                                 (self.cengine1,self.i_centaur),(self.cengine2,self.i_centaur),
                                 (self.pmengine,self.i_pm)])
    def sequence(self,t:float,dt:float,y:np.ndarray):
        for stage,tdrop in zip(self.stages,self.tdrop):
            stage.attached=t<tdrop
        for engine,(t0,t1) in zip(self.engines, self.tburn):
            engine.throttle=t0<t<t1



