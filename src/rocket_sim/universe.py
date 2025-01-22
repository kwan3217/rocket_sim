"""
Code which maintains several vehicles and can propagate them through space and time

Created: 1/16/25
"""
from typing import Callable

import numpy as np

from rocket_sim.vehicle import Vehicle


# Ftype describes a Callable that should take the following named arguments:
#  t: sim time
#  y: integration state as a 1D numpy array
#  dt: step size
#  major_step: True for one of the calls to F in an integrator, False for the others.
#  **kwargs: any other appropriate named arguments in **kwargs, usually vehicle= a reference to an object of Vehicle class
# and return:
#  the first derivative of the state vector at this time as a 1D numpy array
Ftype=Callable[...,np.ndarray]


def rk4(*,F:Ftype, t0:float=0, y0:np.ndarray=None, dt:float=None,**kwargs)->np.ndarray:
    """
    Take a single major step in a numerical integration of a differential equation using the
    fourth-order Runge-Kutta method. Use this one instead of the one in kwanmath because
    it only does take one step and it knows about major steps and kwargs.
    :param F: First-order vector differential equation as described above
    :param t0: Initial time value
    :param y0: Initial state vector
    :param dt: Time step size
    :param kwargs: Other named parameters will be passed by name to F()
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
    def change_fps(self,new_fps:int):
        self.t0=self.t()
        self.i_step=0
        self.fps=new_fps
    def step(self,direction:int=1):
        """
        "After all, it's a step in the right direction,
           a step in the right direction after all..."

        :param direction: 1 (default) if going forward in time, -1 if going backwards. No other
                          values are allowed.
        """
        def F(*,t:float,y:np.ndarray,dt:float,major_step:bool,vehicle:Vehicle):
            a_thr=vehicle.generate_acc(t=t,dt=dt,y=y,major_step=major_step)
            a=a_thr.copy()
            m = vehicle.mass()
            Fs=[]
            accs=[]
            for force in self.forces:
                this_F=force(t=t,dt=dt,y=y,vehicle=vehicle)
                Fs.append(this_F)
                this_a=this_F/m
                a+=this_a
            for acc in self.accs:
                this_a=acc(t=t,dt=dt,y=y,vehicle=vehicle)
                accs.append(this_a)
                a+=this_a
            if major_step:
                vehicle.tlm_point.a_thr=a_thr
                vehicle.tlm_point.Fs=Fs
                vehicle.tlm_point.accs=accs
                vehicle.tlm_point.mass=m
            return np.hstack((y[3:],a))
        dt=direction/self.fps
        for vehicle in self.vehicles:
            vehicle.start_tlm_point(t=self.t(),dt=dt)
            vehicle.tlm_point.y0=vehicle.y
            vehicle.y=rk4(F=F,t0=self.t(),y0=vehicle.y,dt=dt,vehicle=vehicle)
            vehicle.tlm_point.y1=vehicle.y
            vehicle.finish_tlm_point()
        self.i_step+=direction
    def runto(self,*,t1:float):
        direction=-1 if t1<(self.t()-0.5/self.fps) else 1
        while self.t()*direction<t1*direction:
            self.step(direction=direction)


class ZeroGRange(Universe):
    def __init__(self,*,vehicles:list[Vehicle],fps:float):
        super().__init__(vehicles=vehicles,forces=[],t0=0,fps=fps)


class TestStand(ZeroGRange):
    def step(self,direction:int=1):
        # Remember the original state of each vehicle
        old_ys=[vehicle.y for vehicle in self.vehicles]
        # Run the step which will move the vehicles
        super().step(direction=direction)
        # Move the vehicles back
        for vehicle,y in zip(self.vehicles,old_ys):
            vehicle.y=y


class VerticalRange(Universe):
    def __init__(self,*,vehicles:list[Vehicle],fps:float,g0:float=9.80665,forces:list[Callable[...,np.ndarray]]):
        def uniform_gravity(*,t:float,dt:float,y:float,vehicle:Vehicle):
            return np.array([0,0,-g0])
        super().__init__(vehicles=vehicles,forces=forces,accs=[uniform_gravity],t0=0,fps=fps)
