"""
Model of a Titan-III/Centaur (Titan 3E)
"""
import numpy as np

from rocket_sim.srm import SRM
from rocket_sim.universe import ZeroGRange
from rocket_sim.vehicle import Stage, kg_per_lbm, Vehicle
from test_rocket_sim.test_vehicle import tlm, plot_tlm
from voyager import Voyager


def main():
    # mprop from total expended mass in UA1205 paper
    # F1 from initial sea level thrust lbf, table I.
    # F0 from peak of thrust curve from figure 7. Peak
    # thrust is in the startup transient.
    srmL=SRM(mprop=197924,csv="data/UA1205 Thrust Curve.csv",name="Left SRM",F1F0=1199300/1293520,nozzle_cos=np.cos(np.deg2rad(6)))
    srmR=SRM(mprop=197924,csv="data/UA1205 Thrust Curve.csv",name="Right SRM",F1F0=1199300/1293520,nozzle_cos=np.cos(np.deg2rad(6)))
    print(srmL)
    print(srmR.total_impulse())
    srbL=Stage(prop=197924,dry=32447,name="Left SRB")
    srbR=Stage(prop=197924,dry=32447,name="Right SRB")
    stage1=Stage(total=277000*kg_per_lbm,dry=15000*kg_per_lbm,name="Titan 3E stage 1")
    stage2=Stage(total=23000*kg_per_lbm,dry=6000*kg_per_lbm,name="Titan 3E stage 2")
    interstage_and_shroud=Stage(total=10900*kg_per_lbm,prop=0,name="Interstage and Shroud")
    vgr1=Voyager(vgr_id=1)
    centaur=vgr1.stages[vgr1.i_centaur]
    pm=vgr1.stages[vgr1.i_pm]
    sc=vgr1.stages[vgr1.i_mm]
    titan3E=Vehicle(stages=[srbL,srbR,stage1,stage2,interstage_and_shroud,centaur,pm,sc],engines=[(srmL,0),(srmR,1)],extras=[tlm])
    print(titan3E.stages)
    print(titan3E.engines)
    if False:
        stand=TestStand(vehicles=[titan3E],fps=10)
        stand.runto(t1=130.0)
    else:
        range=ZeroGRange(vehicles=[titan3E],fps=10)
        range.runto(t1=130.0)
    plot_tlm(titan3E)


if __name__=="__main__":
    main()