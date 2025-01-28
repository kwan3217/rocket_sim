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
from typing import TextIO, Callable, Any

import numpy as np
from bmw import elorb
from kwanmath.vector import vlength
from kwanspice.mkspk import mkspk
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from spiceypy import str2et, pxform, timout

import vehicle.voyager

from guidance.orbit import dprograde_guide, seq_guide, prograde_guide, yaw_rate_guide
from rocket_sim.gravity import SpiceTwoBody, SpiceJ2, SpiceThirdBody
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Vehicle, g0
from vehicle.titan_3e_centaur import Titan3E
from vehicle.voyager import horizons_data, \
    simt_track_prePM, target_a_prePM, target_e_prePM, target_i_prePM, target_lan_prePM, \
    simt_track_park,  target_a_park,  target_e_park, target_i_park, target_c3_park, \
    init_spice, horizons_et, voyager_et0, best_pm_solution,best_pm_initial_guess, \
    ParsedPM, best_centaur2_initial_guess


def report_state(*,t:float,y:np.ndarray,name:str)->str:
    """
    Generate a string representation of a state that can be turned
    back into a state with bit-for-bit accuracy

    :param t: simt at which state is valid
    :param y: state vector in the form of a (6,) numpy array. y[:3] is position in ICRF in m,
              y[3:] is velocity in ICRF in m/s
    :return: String representation of state
    """
    return f"""{name} (simt, ICRF, SI): 
simt:  {t: .13e}  rx:   {y[0]: .13e}  ry:   {y[1]: .13e}  rz:   {y[2]: .13e}
           {t.hex()}       {y[0].hex()}      {y[1].hex()}      {y[2].hex()}
                              vx:   {y[3]: .13e}  vy:   {y[4]: .13e}  vz:   {y[5]: .13e}
                                   {y[3].hex()}       {y[4].hex()}       {y[5].hex()}"""


class BackpropagationTargeter:
    """
    Targeter that given a final state that is locked down solid as a boundary constraint,
    and an incomplete set of orbital elements to hit as the other constraint, finds
    a set of guidance steering parameters that when flown backwards through the maneuver
    hits the incomplete elements as closely as possible.

    Note -- the given final state is always the final state, identified with that word
            'final' and subscript 1 such as t1, y1, etc. The achieved initial state
            is likewise always identified with the word 'initial' and
    """
    def __init__(self,*,vgr_id:int,report_name:str,t1:float,y1:np.ndarray,fps:int):
        self.vgr_id=vgr_id
        self.et0=voyager_et0[self.vgr_id]
        self.report_name=report_name
        self.t1=t1
        self.y1=y1
        self.fps=fps
        # Plot lines. Keys are names that will be inserted into self.__dict__
        #             Value is list (so mutable) of:
        #               * Callable that calculates the plot line,
        #               * Unit string if it is to be plotted or None if not
        self.plotlines:dict[str,list[Callable,str|None]]={
          "t":[lambda bpt: np.array([x.t for x in bpt.sc.tlm_points]),None],
          "state":[lambda bpt: np.array([x.y0 for x in bpt.sc.tlm_points]).T,None],
          "mass":[lambda bpt: np.array([x.mass for x in bpt.sc.tlm_points]),'kg'],
          "elorb":[lambda bpt: [
            elorb(x.y0[:3], x.y0[3:], l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM, t0=x.t)
            for x in bpt.sc.tlm_points],None]}
        self.bounds = [(-30, 30), (-30, 30), (-0.1, 0.1), (0, 0)]  # Freeze yaw rate at 0
        self.iters=0
    def guide(self):
        """
        Return a guidance program based on the given parameters passed in from optimize
        :return:
        """
        raise NotImplementedError
    def print_params(self):
        raise NotImplementedError
    def tweak_sc(self):
        raise NotImplementedError
    def cost_components(self,*, ouf: TextIO | None)->dict[str,tuple[float,float,float]]:
        raise NotImplementedError
    def plot_lines(self):
        for name,(f,units) in self.plotlines.items():
            self.__dict__[name]=f(self)
        plt.figure(f"Voyager {self.vgr_id} {self.report_name}")
        actual_plotlines=[(k,v[1]) for k,v in self.plotlines.items() if v[1] is not None]
        for i_plotline,(name,units) in enumerate(actual_plotlines):
            plt.subplot(2,3,i_plotline+1)
            plt.cla()
            if units=='':
                plt.ylabel(f'{name}')
            elif "/" in units:
                plt.ylabel(f'{name}/({units})')
            else:
                plt.ylabel(f'{name}/{units}')
            plt.xlabel('simt/s')
            plt.plot(self.t,self.__dict__[name])
        plt.pause(0.1)
    def sim(self,ouf:TextIO=None):
        self.sc:Vehicle = Titan3E(tc_id=5+self.vgr_id)
        initstate = f"""Voyager {self.vgr_id} {self.report_name}
{self.print_params()}
fps: {self.fps} 
 simt=0 in ET: {self.et0:.13e}  ({self.et0.hex()}) ({timout(self.et0, 'YYYY-MM-DDTHR:MN:SC.### ::TDB')} TDB,{timout(self.et0, 'YYYY-MM-DDTHR:MN:SC.### ::UTC')}Z)
{report_state(t=self.t1,y=self.y1,name='Initial state')}"""
        print(initstate)
        earth_twobody = SpiceTwoBody(spiceid=399)
        earth_j2 = SpiceJ2(spiceid=399)
        moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=self.et0)
        sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=self.et0)
        self.universe = Universe(vehicles=[self.sc], accs=[earth_twobody, earth_j2, moon, sun],
                       t0=self.t1, y0s=[self.y1], fps=self.fps)
        self.tweak_sc()
        self.sc.guide=self.guide()
        self.universe.runto(t1=self.t0)
        finalstate = report_state(t=self.sc.tlm_points[-1].t+self.sc.tlm_points[-1].dt,y=self.sc.tlm_points[-1].y1,name='Final state')
        print(finalstate)
        if ouf is not None:
            print(initstate, file=ouf)
            print(finalstate, file=ouf)
        self.plot_lines()
    def cost(self,ouf:TextIO|None)->float:
        cost = 0
        comps = self.cost_components(ouf=ouf)
        coststr=f"Iterations: {self.iters}\n"
        coststr += ", ".join([f"{param_name}={param_val:.6f}{' ' + param_units if param_units is not None else ''}"
                             for param_name, param_val, param_units in zip(self.param_names, self.params, self.param_units)])
        for name, (achieved, target, sigma, pane, units) in comps.items():
            cost += ((target - achieved) / sigma) ** 2
            plt.subplot(2, 3, pane)
            trange = [self.sc.tlm_points[0].t, self.sc.tlm_points[-1].t + self.sc.tlm_points[-1].dt]
            plt.plot(trange, target * np.array([1, 1]), 'k--', label=f"{name} target")
            plt.plot(trange, achieved * np.array([1, 1]), 'r--', label=f"{name} achieved")
            plt.legend()
            coststr += (f"\n{name}hist={target}{' ' + units if units is not None else ''},"
                        f"{name}calc={achieved}{' ' + units if units is not None else ''},"
                        f"d{name}={target - achieved}{' ' + units if units is not None else ''},"
                        f"{(target - achieved) / sigma:8.1f} sigma")
        plt.subplot(2, 3, 6)
        plt.cla()
        coststr += f"\nCost: {cost:.3e}"
        print(coststr)
        if ouf is not None:
            print(coststr, file=ouf)
        plt.text(0, 0.5, coststr, horizontalalignment='left', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.axis('off')
        plt.pause(0.1)
        return cost
    def __call__(self,params,ouf:TextIO|None=None)->float:
        """
        This is the interface to scipy.optimize.minimize. It takes a set
        of parameters, calculates the cost, and returns it.
        :return:
        """
        self.iters+=1
        self.accept_params(params)
        self.sim()
        cost=self.cost(ouf)
        return cost
    def accept_params(self, params):
        self.params=params
    def target(self,*,optimize:bool=False):
        initial_guess=self.initial_guess()
        for guess,(bound0,bound1) in zip(initial_guess,self.bounds):
            if not (bound0<=guess<=bound1):
                raise ValueError("Bounds do not bracket guess")
        if optimize:
            result = minimize(self,
                              initial_guess,
                              method='L-BFGS-B',
                              options={'ftol':1e-6,'disp':True},
                              bounds=self.bounds)
            print("Achieved cost:", result.fun)
            print(result)
            final_guess=result.x
        else:
            result=None
            final_guess=initial_guess
        print("Optimal parameters:", final_guess)
        print("Optimal run: ")
        self.accept_params(final_guess)
        with open(f'products/vgr{self.vgr_id}_{self.save}_optimal_run.txt',"wt") as ouf:
            sc=self.sim(ouf=ouf)
            self.cost(ouf=ouf)
            if result is not None:
                print("Optimizer results:",result,file=ouf)
        return sc
    def export(self):
        states=sorted([np.hstack((np.array(x.t),x.y0)) for x in sc.tlm_points], key=lambda x:x[0])
        decimated_states=[]
        i=0
        di=1
        done=False
        while not done:
            states[i][0]+=voyager_et0[vgr_id]
            decimated_states.append(states[i])
            try:
                if sc.tburn[sc.i_pmengine][0]<states[i+di][0]<sc.tburn[sc.i_pmengine][0]:
                    di=1
                else:
                    di=100
            except IndexError:
                break
            i+=di
        mkspk(oufn=f'products/vgr{sc.vgr_id}_pm.bsp',
              fmt=['f', '.3f', '.3f', '.3f', '.6f', '.6f', '.6f'],
              data=decimated_states,
              input_data_type='STATES',
              output_spk_type=5,
              object_id=sc.spice_id,
              object_name=f'VOYAGER {sc.vgr_id}',
              center_id=399,
              center_name='EARTH',
              ref_frame_name='J2000',
              producer_id='https://github.com/kwan3217/rocket_sim',
              data_order='EPOCH X Y Z VX VY VZ',
              input_data_units=('ANGLES=DEGREES', 'DISTANCES=m'),
              data_delimiter=' ',
              leapseconds_file='data/naif0012.tls',
              pck_file='products/gravity_EGM2008_J2.tpc',
              segment_id=f'VGR{sc.vgr_id}_PM',
              time_wrapper='# ETSECONDS',
              comment="""
               Best-estimate of trajectory through Propulsion Module burn,
               hitting historical post-MECO2 orbit elements which are
               available and matching Horizons data."""
              )
    def initial_guess(self):
        raise NotImplementedError


class PMTargeter(BackpropagationTargeter):
    def __init__(self,*,vgr_id:int):
        """

        :param vgr_id:
        """
        super().__init__(vgr_id=vgr_id,report_name="backpropagation through PM burn",
                         t1=horizons_et[vgr_id] - voyager_et0[vgr_id],
                         y1=np.array(horizons_data[vgr_id][3:9]) * 1000.0,fps=100)  # Convert km to m and km/s to m/s
        self.plotlines["spd"]=[lambda bpt:vlength(bpt.state[3:6,:]),"m/s"]
        self.plotlines["e"] = [lambda bpt:np.array([this_elorb.e for this_elorb in bpt.elorb]),""]
        self.plotlines["i"] = [lambda bpt:np.array([np.rad2deg(this_elorb.i) for this_elorb in bpt.elorb]),"deg"]
        self.plotlines["a"] = [lambda bpt:np.array([this_elorb.a for this_elorb in bpt.elorb]) / 1852,"nmi"]  # work directly in km
        self.param_names=("dpitch","dyaw","dthr","yawrate")
        self.param_units=("deg", "deg", None, "deg/s")
        self.t0=simt_track_prePM[self.vgr_id]
        self.save="pm"
    def accept_params(self, params):
        super().accept_params(params)
        self.dpitch, self.dyaw, self.dthr, self.yawrate = params
    def print_params(self)->str:
        return f"""dpitch: {self.dpitch:.13e} ({self.dpitch.hex()})
dyaw: {self.dyaw:.13e} ({self.dyaw.hex()})
dthr: {self.dthr:.13e} ({self.dthr.hex()})
yawrate: {self.yawrate:.13e} ({self.yawrate.hex()})"""
    def guide(self):
        return seq_guide({self.sc.tdrop[self.sc.i_centaur]: prograde_guide,
                          float('inf'): yaw_rate_guide(r0=self.y1[:3],v0=self.y1[3:],
                                                       dpitch=self.dpitch, dyaw=self.dyaw, yawrate=self.yawrate,
                                                       t0=self.sc.tburn[self.sc.i_pmengine][0])})
    def tweak_sc(self):
        # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
        self.sc.stages[self.sc.i_centaur].prop_mass = 0
        self.sc.stages[self.sc.i_pm].prop_mass = 0
        # Tweak engine efficiency
        self.sc.engines[self.sc.i_pmengine].eff = 1.0 + self.dthr
    def cost_components(self,*, ouf: TextIO | None)->dict[str,tuple[float,float,float,int,str]]:
        """

        :param ouf:
        :return: dict -- Key is name of component
                         value is:
                          * Targeted value of this component
                          * Achieved value of this component
                          * Sigma of this component
                          * Pane to draw target into
                          * Unit string for this component
        """
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
        y0_j = self.sc.y.copy()
        Mej = pxform('J2000', 'IAU_EARTH', simt_track_prePM[self.vgr_id] + voyager_et0[self.vgr_id])
        r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
        v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
        r0_e = Mej @ r0_j
        v0_e = Mej @ v0_j
        # In this frame, the reported ascending node is Longitude of ascending node, not
        # right ascension. It is relative to the Earth-fixed coordinates at this instant,
        # not the sky coordinates.
        elorb0 = elorb(r0_e, v0_e, l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM,
                       t0=simt_track_prePM[self.vgr_id],
                       deg=True)
        result={}
        result["a"]=(elorb0.a/1852,target_a_prePM[self.vgr_id]/1852,2e-1   ,4,"nmi")  # da in nmi
        result["e"]=(elorb0.e     ,target_e_prePM[self.vgr_id]     ,0.00007,2,"")  # de
        result["i"]=(elorb0.i     ,target_i_prePM[self.vgr_id]     ,0.01   ,3,"deg") #di
        return result
    def initial_guess(self):
        guess=best_pm_initial_guess(vgr_id=self.vgr_id)
        return guess


class Centaur2Targeter(BackpropagationTargeter):
    def __init__(self, *, pm:Vehicle|ParsedPM):
        if isinstance(pm,ParsedPM):
            vgr_id = pm.vgr_id
            t1 = pm.simt0
            y1 = pm.y0
        else:
            vgr_id=pm.vgr_id
            t1=pm.t[-1]
            y1=pm.y
        super().__init__(vgr_id=vgr_id, report_name="backpropagation through Centaur 2 burn",
                             t1=t1,y1=y1,fps=100)  # Convert km to m and km/s to m/s
        self.plotlines["spd"] = [lambda bpt: vlength(bpt.state[3:6, :]), "m/s"]
        self.plotlines["e"] = [lambda bpt:np.array([this_elorb.e for this_elorb in bpt.elorb]),""]
        self.plotlines["i"] = [lambda bpt:np.array([np.rad2deg(this_elorb.i) for this_elorb in bpt.elorb]),"deg"]
        self.plotlines["c3"] = [lambda bpt:-(vehicle.voyager.EarthGM / (1000 ** 3)) / (
                    np.array([this_elorb.a for this_elorb in bpt.elorb]) / 1000),"km**2/s**2"]  # work directly in km
        # Run back to this point at 100Hz, then back to parking solution at 10Hz
        self.sc:Vehicle = Titan3E(tc_id=5+self.vgr_id)
        self.param_names=("dpitch","dyaw","dthr","pitchrate")
        self.param_units=("deg", "deg", None, "deg/s")
        self.t0 = self.sc.tburn[self.sc.i_cengine2][0]-10
        self.bounds = [(-30, 30), (-30, 30), (-0.1, 0.1), (-1, 1)]  # Freeze yaw rate at 0
        self.save = "centaur2"
    def accept_params(self, params):
        super().accept_params(params)
        self.dpitch, self.dyaw, self.dthr, self.pitchrate = params
    def print_params(self) -> str:
        return f"""dpitch: {self.dpitch:.13e} ({self.dpitch.hex()})
dyaw: {self.dyaw:.13e} ({self.dyaw.hex()})
dthr: {self.dthr:.13e} ({self.dthr.hex()})
pitchrate: {self.pitchrate:.13e} ({self.pitchrate.hex()})"""
    def guide(self):
        return dprograde_guide(dpitch=self.dpitch, dyaw=self.dyaw, pitchrate=self.pitchrate, t0=self.sc.tburn[self.sc.i_cengine2][0])
    def tweak_sc(self):
        # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
        self.sc.stages[self.sc.i_centaur].prop_mass = 0
        # Tweak engine efficiency
        self.sc.engines[self.sc.i_cengine2].eff = 1.0 + self.dthr
    def sim(self,ouf:TextIO=None):
        super().sim(ouf=ouf)
        self.universe.change_fps(10)
        self.universe.runto(t1=simt_track_park[self.vgr_id])
        self.plot_lines()
        finalstate = report_state(t=self.sc.tlm_points[-1].t + self.sc.tlm_points[-1].dt, y=self.sc.tlm_points[-1].y1,
                                  name='State just after Centaur burn 1')
        print(finalstate)
        if ouf is not None:
            print(finalstate,file=ouf)
    def cost_components(self,*, ouf: TextIO | None)->dict[str,tuple[float,float,float,int,str]]:
        """

        :param ouf:
        :return: dict -- Key is name of component
                         value is:
                          * Targeted value of this component
                          * Achieved value of this component
                          * Sigma of this component
                          * Pane to draw target into
                          * Unit string for this component
        """
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
        y0_j = self.sc.y.copy()
        Mej = pxform('J2000', 'IAU_EARTH', simt_track_prePM[self.vgr_id] + voyager_et0[self.vgr_id])
        r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
        v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
        r0_e = Mej @ r0_j
        v0_e = Mej @ v0_j
        # In this frame, the reported ascending node is Longitude of ascending node, not
        # right ascension. It is relative to the Earth-fixed coordinates at this instant,
        # not the sky coordinates.
        elorb0 = elorb(r0_e, v0_e, l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM,
                       t0=simt_track_park[self.vgr_id],
                       deg=True)
        result={}
        result["a"]=(elorb0.a/1852,target_a_park[self.vgr_id]/1852,2e-1   ,5,"nmi")  # da in nmi
        result["e"]=(elorb0.e     ,target_e_park[self.vgr_id]     ,0.00007,3,"")  # de
        result["i"]=(elorb0.i     ,target_i_park[self.vgr_id]     ,0.01   ,4,"deg") #di
        return result
    def initial_guess(self):
        guess= best_centaur2_initial_guess(vgr_id=self.vgr_id)
        return guess


def export(vgr_id:int):
    with open(f"data/vgr{vgr_id}/v{vgr_id}_horizons_vectors_1s.txt","rt") as inf:
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
    with open(f"data/vgr{vgr_id}/v{vgr_id}_horizons_vectors.txt","rt") as inf:
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
    mkspk(oufn=f'products/vgr{vgr_id}_horizons_vectors.bsp',
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
          pck_file='products/gravity_EGM2008_J2.tpc',
          segment_id=f'VGR{vgr_id}_HORIZONS_1S',
          time_wrapper='# ETSECONDS',
          comment=f"""
           Export of Horizons data calculated from a kernel they have but I don't,
           Voyager_{vgr_id}_ST+refit2022_m. This file is at 1 second intervals from beginning
           of available data to that time plus 1 hour, then at 1 minute intervals
           to the end of the launch day. From there we _do_ have a supertrajectory kernel
           that covers the rest of the mission. I have reason to believe that every
           supertrajectory that NAIF or SSD publishes has the same prime mission 
           segment, and just re-fit the interstellar mission portion, so I think the
           post-launch trajectory in the supertrajectory from Horizons is the same as
           the one I have."""
          )


def main():
    init_spice()
    for vgr_id in (1,2):
        pm_targeter=PMTargeter(vgr_id=vgr_id)
        pm_targeter.target(optimize=True)
        pm_guess=best_pm_solution(vgr_id=vgr_id)
        centaur2_targeter=Centaur2Targeter(pm=pm_guess)
        centaur2_targeter.target(optimize=False)
    #for vgr_id in (1,2):
    #    export(vgr_id)


if __name__=="__main__":
    main()