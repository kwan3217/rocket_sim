"""""
Engine that uses a thrust curve to control its thrust as a function of simt

Created: 1/21/25
"""
import numpy as np
from matplotlib import pyplot as plt

from rocket_sim.universe import TestStand, ZeroGRange
from rocket_sim.vehicle import Engine, N_per_lbf, Stage, Vehicle
from test_rocket_sim.test_vehicle import plot_tlm


class SRM(Engine):
    def __init__(self,*,mprop:float,
                 ts:np.ndarray=None,Fs:np.ndarray=None,
                 csv:str=None,csvscl:float=N_per_lbf,
                 name:str,
                 F1F0:float=None,
                 nozzle_cos:float=1.0):
        """

        :param F1F0: ratio of sea-level thrust/ve to
        :param ts: Timestamps of thrust curve
        :param Fs: Thrust values of vacuum thrust curve
        :param csv: Name of csv to load thrust curve from. Must have one
        :param csvscl: Multiply each thrust value by this value to convert
                       to SI. Default is the appropriate value for lbf, as used
                       in `TitanSRM Thrust.csv`.
        """
        if ts is None:
            ts = []
            Fs = []
            with open(csv,"rt") as inf:
                header=inf.readline()
                for line in inf:
                    line=line.strip()
                    t,F=line.split(",")
                    t=float(t)
                    F=float(F)*csvscl
                    ts.append(t)
                    Fs.append(F)
            ts=np.array(ts)
            Fs=np.array(Fs)
        self.ts=ts
        self.Fs=Fs
        self.mdots=self.Fs
        Itot=self.total_impulse(verbose=False)
        self.ve0=Itot/mprop
        self.ve1=self.ve0*F1F0
        self.nozzle_cos=nozzle_cos
        self.mdots=self.Fs/self.ve0 # This is the true motor coefficient, both thrust and ve are associated
                                    # such that mdot is what the table says at every moment
        self.mdottable=lambda t:np.interp(t,self.ts,self.mdots,left=0.0,right=0.0)
        super().__init__(thrust10=Fs[0],ve0=self.ve0,name=name)
    def generate_thrust(self, t: float, dt: float, y: np.ndarray, major_step: bool) -> float:
        mdot=self.mdottable(t)
        self.thrust10=self.ve0*mdot
        self.thrust11=self.ve1*mdot
        return super().generate_thrust(t=t,dt=dt,y=y,major_step=major_step)*self.nozzle_cos
    def total_impulse(self,verbose:bool=True):
        result=0
        for t0,t1,F0,F1 in zip(self.ts[:-1],self.ts[1:],self.Fs[:-1],self.Fs[1:]):
            result+=(t1-t0)*(F0+F1)/2 #Area of each trapezoid
        if verbose:
            plt.figure(f"{self.name} thrust curve")
            plt.plot(self.ts,self.Fs)
            plt.xlabel("simt/s")
            plt.ylabel("Thrust/N")
            plt.title(f"Total impulse {result/1e6:.1f}MN*s")
            plt.show()
        return result


def main():
    # mprop from total expended mass in UA1205 paper
    # F1 from initial sea level thrust lbf, table I.
    # F0 from peak of thrust curve from figure 7. Peak
    # thrust is in the startup transient.
    srm=SRM(mprop=197924,csv="data/UA1205 Thrust Curve.csv",name="Left SRM",F1F0=1199300/1293520,nozzle_cos=np.cos(np.deg2rad(6)))
    print(srm)
    print(srm.total_impulse())
    srb=Stage(prop=197924,dry=32447,name="Left SRB")
    vehicle=Vehicle(stages=[srb],engines=[(srm,0)])
    print(vehicle.stages)
    print(vehicle.engines)
    if False:
        stand=TestStand(vehicles=[vehicle],fps=10)
        stand.runto(t1=130.0)
    else:
        range=ZeroGRange(vehicles=[vehicle],fps=10)
        range.runto(t1=130.0)
    plot_tlm(vehicle)



if __name__ == "__main__":
    main()
