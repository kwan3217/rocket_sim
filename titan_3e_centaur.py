"""
Model of a Titan-III/Centaur (Titan 3E)
"""
import numpy as np
from matplotlib import pyplot as plt


def trap(Fcurve:dict[float,float])->float:
    result=0
    F0=0
    t0=0
    for t1,F1 in Fcurve.items():
        dt=t1-t0
        Fm=(F1+F0)/2
        dA=dt*Fm
        result+=dA
        t0=t1
        F0=F1
    return result

class SolidRocketMotor:
    def __init__(self,*,M0:float,M1:float,Fsl:float,F0:float,vesl:float,ve0:float,Fcurve:dict[float,float]=None):
        """

        :param M0: Mass at ignition (stage t0), kg
        :param M1: Mass at burnout (stage t1), kg
        :param Fsl: Nominal thrust at sea level, N
        :param F0: Nominal thrust in vacuum, N
        :param vesl: effective exhaust velocity at sea level, N
        :param ve0: effective exhaust velocity in vacuum, N
        :param Fcurve: thrust curve -- dictionary of with time from ignition as key and relative thrust as value.
                       First point must be (0,0), last point must be (t1,0). Thrust and mass loss are presumed
                       to be zero before t=0 and after t=t1.
        """
        # All of this is based on mass flow rate. There is a known amount of propellant,
        # the difference between M0 and M1.
        self.mass=M0
        self.mprop=M0-M1
        # The curve passed in is proportional to thrust, chamber pressure, etc
        # but can be normalized so that the area is the total amount of propellant
        # in mass units (kg) and the time axis is in time units (s), which therefore
        # give the dependent variable being in units of mass units per time unit (kg/s)
        # This is the mass flow rate at any time.
        self.mdot0=-F0/ve0 #Mass flow rate in vacuum, kg/s
        self.mdotsl=-Fsl/vesl
        self.tburn0=self.mprop/-self.mdot0
        self.tburnsl=self.mprop/-self.mdotsl
        self.itot0=self.tburn0*F0
        self.itotsl=self.tburnsl*Fsl
        #Area of curve in y*sec, where y is whatever the units of the curve are
        Acurve=trap(Fcurve)
        self.tcurve=np.array([t for t in Fcurve])
        Fcurve=np.array([Fcurve[t] for t in Fcurve])
        #The area of Fcurve is Acurve y*s. We want mdotcurve with area -mprop
        #and units kg. so divide Fcurve by Acurve to get a curve with area 1,
        #then multiply by -mprop
        self.mdotcurve=-Fcurve*self.mprop/Acurve
        plt.plot(self.tcurve,self.mdotcurve)
        expected_area=-self.mprop
        actual_area=trap({t:mdot for t,mdot in zip(self.tcurve,self.mdotcurve)})
        assert np.isclose(expected_area,actual_area)
        pass
    def start(self,t0:float):

    def step(self,t:float,dt:float):
        """

        :param t:
        :param dt:
        :return:
        """


def main():
    SolidRocketMotor(M0=226_233.0,M1=33_798.0,
                     Fsl=5_293_300.0,vesl=238*9.80665,
                     F0=5_849_300,ve0=263*9.80665,
                     Fcurve={
                           0.00000:       0.0,
                           0.01830:     388.8,
                           0.04036:   18451.8,
                           0.06322:   54140.4,
                           0.08528:   87318.0,
                           0.10705:  165774.6,
                           0.12960:  276955.2,
                           0.15134:  413294.4,
                           0.17388:  582357.6,
                           0.19669:  726262.2,
                           0.21924:  847519.2,
                           0.24181:  941090.4,
                           0.26385: 1009486.8,
                           0.28643: 1067823.0,
                           0.30902: 1108549.8,
                           0.33081: 1136689.2,
                           0.35341: 1154752.2,
                           0.37575: 1165282.2,
                           0.39835: 1183361.4,
                           0.42069: 1186326.0,
                           0.44336: 1185532.2,
                           0.46577: 1187249.4,
                           0.48818: 1187719.2,
                           0.51052: 1183766.4,
                           0.53272: 1181077.2,
                           0.55520: 1177756.2,
                           0.57781: 1180737.0,
                           0.59981: 1174905.0,
                           0.62195: 1176606.0,
                           0.64449: 1173916.8,
                           0.66664: 1166189.4,
                           0.68898: 1166027.4,
                           0.71125: 1164585.6,
                           0.73386: 1163160.0,
                           0.75600: 1158591.6,
                           0.77827: 1162188.0,
                           0.80122: 1159515.0,
                           0.82383: 1160600.4,
                           0.84671: 1157295.6,
                           0.86952: 1156501.8,
                           0.89213: 1150669.8,
                           0.91480: 1154282.4,
                           2.20810: 1140690.0,
                           4.21810: 1163210.0,
                           6.42030: 1178320.0,
                           8.71840: 1191570.0,
                          11.01700: 1199240.0,
                          13.41180: 1201340.0,
                          15.71130: 1197850.0,
                          18.10670: 1192510.0,
                          20.31080: 1183430.0,
                          22.61090: 1172500.0,
                          24.91110: 1159700.0,
                          27.11580: 1143180.0,
                          29.22500: 1122930.0,
                          31.43000: 1102680.0,
                          33.44360: 1080560.0,
                          35.64840: 1062180.0,
                          37.75750: 1043780.0,
                          39.96220: 1027260.0,
                          42.07110: 1010730.0,
                          44.27580:  994200.0,
                          46.57590:  983270.0,
                          48.78040:  968610.0,
                          51.08050:  957680.0,
                          53.38060:  946740.0,
                          55.58480:  935800.0,
                          57.88460:  928590.0,
                          60.18450:  919520.0,
                          62.48430:  912310.0,
                          64.87980:  905100.0,
                          67.17950:  899750.0,
                          69.47920:  892540.0,
                          71.77880:  887190.0,
                          74.07850:  881840.0,
                          76.47400:  874640.0,
                          78.67810:  865560.0,
                          80.97820:  854620.0,
                          83.27830:  843690.0,
                          85.57830:  832760.0,
                          87.87810:  825550.0,
                          90.17760:  822060.0,
                          92.57300:  816720.0,
                          94.77690:  809490.0,
                          97.07690:  800420.0,
                          99.28110:  789480.0,
                         101.48540:  778540.0,
                         103.78510:  771330.0,
                         106.08480:  765980.0,
                         108.38440:  760630.0,
                         110.39780:  740360.0,
                         111.64600:  705140.0,
                         112.32010:  660550.0,
                         112.99430:  615970.0,
                         113.38120:  569500.0,
                         113.95960:  524900.0,
                         114.63370:  480320.0,
                         115.30790:  435740.0,
                         115.98190:  393010.0,
                         116.84760:  348450.0,
                         117.23290:  322440.0,
                         118.00290:  277870.0,
                         118.86850:  235160.0,
                         119.82990:  192470.0,
                         120.88690:  151650.0,
                         122.23120:  112710.0,
                         123.67120:   75650.0,
                         125.30230:   44180.0,
                         127.31570:   23920.0,
                         128.75340:   14760.0,
                         135.00000:       0.0,
                     })

if __name__=="__main__":
    main()