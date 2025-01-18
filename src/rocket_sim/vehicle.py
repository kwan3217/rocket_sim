"""
Vehicle model, including engines and stages.

Created 2025-01-12
"""
from copy import copy
from typing import Callable, Any

import numpy as np


class Stage:
    """
    Describes a rocket stage. This is an element of a rocket
    which has a constant dry mass, and a variable propellant mass
    """
    def __init__(self,*,dry:float=None,prop:float=None,total:float=None):
        """

        :param dry_mass: Mass of structure, attached engines, etc.
                         Everything except for the usable propellant --
                         when the prop_mass hits zero, the engines
                         should not be able to be run. If there is unusable
                         propellant due to geometry of the fuel tank etc,
                         the unusable prop will be budgeted here.
                         SI unit is kg.
        :param prop_mass: Mass of a full load of propellant. SI unit is kg.
        """
        if dry is None:
            dry=total-prop
        elif prop is None:
            prop=total-dry
        elif total is None:
            total=dry+prop
        self.dry_mass=dry
        self.prop_mass0=prop
        self.prop_mass=prop
        self.attached=True
    def mass(self):
        """
        Calculate the current mass of the stage
        :return:
        """
        return self.prop_mass+self.dry_mass if self.attached else 0
    def change_mass(self,dt:float,major_step:bool,mdot:float):
        if major_step:
            self.prop_mass+=mdot*dt


class Engine:
    """
    Describes an engine. This is attached to a particular
    rocket stage, drawing propellant from it at a determined rate.
    The engine reduces the propellant mass of a stage, produces
    thrust in a particular direction, but does not itself weigh anything
    (its weight should be accounted in the dry mass of some stage that
    jettisons at the same time as the engine).

    Some terms: An engine is either *attached* to a vehicle or not,
    and is *connected* to a particular stage to draw propellant from it.
    Usually connections are defined when the vehicle is instantiated
    and are not changed during the flight, even at a staging event.
    """
    def __init__(self,thrust10:float,ve0:float):
        """
        :param thrust10: Nominal 100% thrust in vacuum. SI unit is N
        :param ve0: Effective exhaust velocity in vacuum. SI unit is m/s
        :param stage: Stage that we are drawing propellant from
        """
        self.thrust10=thrust10
        self.ve0=ve0
        self.attached=True
        self.stage=None
        self.throttle=1.0
    def connect_prop(self,stage:Stage):
        """
        Define the connection to a Stage that is carrying propellant for this engine
        :param stage:
        """
        self.stage=stage
    def generate_thrust(self,t:float,dt:float,y:np.ndarray,major_step:bool)->float:
        if self.attached and self.stage is not None and self.stage.attached:
            thrust_mag=self.throttle*self.thrust10
            mdot=-thrust_mag/self.ve0
            self.stage.change_mass(dt=dt,major_step=major_step,mdot=mdot)
        else:
            thrust_mag=0
        return thrust_mag


class Vehicle:
    def __init__(self,*,
                 stages:list[Stage]|None=None,
                 engines:list[tuple[Engine,int]]|None=None,
                 guide:Callable[...,np.array]|None=None,
                 extras:list[Callable[...,None]]|None=None):
        """

        :param guide: Guidance function. This calculates the direction that the vehicle
                      should be pointing. For now it returns a vector indicating the
                      direction of the z-axis of the vehicle. Later will return a full
                      SO(3) -- quaternion, 3x3 matrix, something like that when we are
                      doing 6DoF.

                      guide() is allowed to throttle the engines, but should not do
                      anything else active to the vehicle (that's what sequence() is for).
                      The type hinting system doesn't allow for named parameters. The parameters
                      of the function must be:
                      def guide_foo(*,t:float,dt:float,y:np.ndarray,major_step:bool,vehicle:Vehicle):...
        :param extra: A list of "extra" functions. Each will be called in order. These functions
                      are intended for things like sending telemetry, maybe GUI, etc. One or more
                      of them might be command interfaces, which are allowed to change
                      vehicle state.
                      The prototype of each function must be:
                      def extra_foo(*,t:float,dt:float,y:np.ndarray,major_step:bool,vehicle:Vehicle)

        """
        self.stages:list[Stage]=[copy(stage) for stage in stages] if stages is not None else []
        self.engines:list[Engine]=[copy(engine) for engine,_ in engines] if engines is not None else []
        for self_engine,(_,i_stage) in zip(self.engines,engines):
            self_engine.connect_prop(self.stages[i_stage])
        self.guide:Callable=guide
        self.extras:list[Callable]=extras if extras is not None else []
        # This is for storing the vehicle state in between integrator major steps.
        # We don't use it ourselves in our methods, preferring the y passed in
        # so we get the right one for each minor step.
        self.y:np.ndarray|None=None
    def reset(self):
        self.tlm=[]
        for stage in self.stages:
            stage.prop_mass=stage.prop_mass0
            stage.attached=True
    def sequence(self,t:float,dt:float,y:np.ndarray):
        """
        Sequence the vehicle. This is a hook -- override it in your child
        that is specific to a concrete vehicle to activate/deactivate engines.
        This is only called during major steps, so it is allowed to
        modify the state of the vehicle or any of the stages or engines
        attached to it.
        :param t:
        :param dt: Time step. Note that if this is negative, time is running in reverse.
                   Some simulations are easier if this function is able to reattach
                   stages and engines if time is in reverse and the time point flows
                   back across a staging time.
        :param y:
        """
        pass
    def mass(self)->float:
        result=0
        for stage in self.stages:
            result+=stage.mass()
        return result
    def thrust_mag(self,t:float,dt:float,y:np.ndarray, major_step:bool)->float:
        thrust_mag=0
        for engine in self.engines:
            thrust_mag+=engine.generate_thrust(t=t,dt=dt,y=y,major_step=major_step)
        return thrust_mag
    def thrust_dir(self,t:float,dt:float,y:np.ndarray,major_step:bool)->np.ndarray:
        if self.guide is not None:
            result=self.guide(t=t,dt=dt,y=y,major_step=major_step,vehicle=self)
        else:
            result=np.array((0,0,1))
        return result
    def generate_acc(self,*,t:float,dt:float,y:np.ndarray,major_step:bool):
        if major_step:
            self.sequence(t=t,dt=dt,y=y)
        if self.extras is not None:
            for extra in self.extras:
                extra(t=t,dt=dt,y=y,major_step=major_step,vehicle=self)
        Fmag=self.thrust_mag(t=t,dt=dt,y=y,major_step=major_step)
        Fdir=self.thrust_dir(t=t,dt=dt,y=y,major_step=major_step)
        m=self.mass()
        return Fmag*Fdir/m



