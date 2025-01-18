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


# Ftype describes a Callable that should take the following named arguments:
#  t: sim time
#  y: integration state as a 1D numpy array
#  dt: step size
#  major_step: True for one of the calls to F in an integrator, False for the others.
# and return:
#  the first derivative of the state vector at this time as a 1D numpy array
Ftype=Callable[...,np.ndarray]

def rk4(*,F:Ftype, t0:float=0, y0:np.ndarray=None, dt:float=None,**kwargs)->np.ndarray:
    """
    Take a fixed number of steps in a numerical integration of a differential equation using the
    fourth-order Runge-Kutta method.
    :param F: First-order vector differential equation as described above
    :param x0: Initial state
    :param t0: Initial time value
    :param nstep: Number of steps to take
    :param t1: final time value
    :param dt: Time step size
    :return: A tuple (t1,x1) of the time and state at the end of the final step. State x1
             will have same dimensionality as the input x0
    """
    dy1=dt*F(t=t0     ,y=y0      ,dt=dt,major_step=True ,**kwargs)
    dy2=dt*F(t=t0+dt/2,y=y0+dy1/2,dt=dt,major_step=False,**kwargs)
    dy3=dt*F(t=t0+dt/2,y=y0+dy2/2,dt=dt,major_step=False,**kwargs)
    dy4=dt*F(t=t0+dt  ,y=y0+dy3  ,dt=dt,major_step=False,**kwargs)
    dy=(dy1+2*dy2+2*dy3+dy4)/6
    yp=y0+dy
    return yp


class Universe:
    def __init__(self,*,vehicles:list[Vehicle],
                        forces:list[Callable]|None=None,
                        accs:list[Callable]|None=None,
                        t0:float=0.0,
                        y0s:list[np.ndarray]|None=None,
                        fps:int|float=64):
        """

        :param vehicles: List of vehicles to simultaneously propagate
        :param forces: List of functions which return force vectors (if SI,
                       then in N). Drag is a force, and its value is dependent
                       on the shape and size of the vehicle and local air properties,
                       so the exact same shape gets the exact same drag force
                       no matter what the vehicle weighs. A vehicle made
                       of wood will have the same drag as the same-shaped
                       vehicle made of metal, but the heavier metal vehicle
                       won't accelerate as much. A function in the list of
                       forces will return the same force, which is then divided
                       by the mass to get the acceleration to integrate.
        :param accs: List of functions that return accelerations. It is legendary
                     that Galileo proved that heavier things accelerated due to
                     gravity by dropping different weight cannonballs off the
                     Leaning Tower of Pisa. There is more gravitational force
                     on a heavy object, but it is exactly proportional to mass
                     so all objects have the same acceleration at the same point
                     in a gravity field. The mass proportionality on one side
                     cancels the acceleration proportionality on the other side.
        :param t0: Time (t variable) at beginning of simulation. Default is zero,
                   but you can put in different values for different times.
        :param y0s: Initial state vector of each vehicle. If not passed, each vehicle
                    starts with its position at the origin and zero velocity.
        :param fps: "Frame rate" or reciprocal of major step size. Used to make dt
                    more accurate -- t and dt for a step are calculated from an integer
                    number of frames divided by fps, so when the frame number is a
                    multiple of fps, t is an exact integer more than t0.
        """
        self.vehicles=vehicles
        for vehicle in self.vehicles:
            vehicle.reset()
        self.forces=forces if forces is not None else []
        self.accs=accs if accs is not None else []
        self.t0=t0
        self.i_step=0
        self.fps=fps
        if y0s is None:
            for vehicle in self.vehicles:
                vehicle.y = np.zeros(6)
        else:
            for vehicle,y0 in zip(self.vehicles,y0s):
                vehicle.y=y0.copy()
    def t(self):
        return self.t0+self.i_step/self.fps
    def step(self,direction:int=1):
        """
        "After all, it's a step in the right direction,
           a step in the right direction after all..."

        :param direction: 1 (default) if going forward in time, -1 if going backwards. No other
                          values are allowed.
        """
        def F(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
            a=vehicle.generate_acc(t=t,dt=dt,y=y,major_step=major_step)
            m = vehicle.mass()
            for force in self.forces:
                a+=force(t=t,dt=dt,y=y)/m
            for acc in self.accs:
                a+=acc(t=t,dt=dt,y=y)
            return np.hstack((y[3:],a))
        for vehicle in self.vehicles:
            vehicle.y=rk4(F=F,t0=self.t(),y0=vehicle.y,dt=direction/self.fps,vehicle=vehicle)
        self.i_step+=direction
    def runto(self,*,t1:float):
        direction=-1 if t1<(self.t()-0.5/self.fps) else 1
        while self.t()*direction<t1*direction:
            self.step(direction=direction)


class ZeroGRange(Universe):
    def __init__(self,*,vehicles:list[Vehicle],fps:float):
        super().__init__(vehicles=vehicles,forces=[],t0=0,fps=fps)


class TestStand(ZeroGRange):
    def step(self):
        # Remember the original state of each vehicle
        old_ys=[vehicle.y for vehicle in self.vehicles]
        # Run the step which will move the vehicles
        super().step()
        # Move the vehicles back
        for vehicle,y in zip(self.vehicles,old_ys):
            vehicle.y=y