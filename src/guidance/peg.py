"""
PEG on a clean sheet with functions instead of objects
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from kwanmath.vector import vnormalize, vcross, vlength, vdot
from matplotlib import pyplot as plt
from numpy import log as ln

from rocket_sim.planet import SpicePlanet
from rocket_sim.vehicle import Vehicle


def tau(*,a0:float,ve:float)->float:
    """
    Normalized mass of rocket. This is the current mass divided by the current flow rate.
    We calculate it from the current acceleration and effective exhaust velocity like this:
    F=m*a
    ve=F/mdot
    tau=m/mdot
    m=F/a
    mdot=F/ve
    tau=(F/a)/(F/ve)
       =(F/a)(ve/F)
       =ve/a
    :param a: Acceleration of rocket in L/T**2 (SI: m/s^2)
    :param ve: Effective exhaust velocity in L/T (SI: m/s)
    :return: Normalized mass in T (SI: s)
    """
    return ve/a0

def a(*,a0:float,ve:float,t:float)->float:
    return a0/(1-t/tau(a0=a0,ve=ve))


def resolve_basis(*,rv:np.array,vv:np.array)->tuple[np.array,np.array,np.array]:
    rhat=vnormalize(rv) #vertical basis vector
    hhat=vnormalize(vcross(rv,vv)) #crossrange basis vector, generally pointing "left" if qhat is considered "forward".
    qhat=vnormalize(vcross(hhat,rhat)) #downrange basis vector, q for \hat{theta}
    return rhat,qhat,hhat

def steer(*,rv:np.array,vv:np.array,mu:float,t:float,
            a0:float,ve:float,
            A:float,B:float,K:float=0.0,
            )->np.array:
    """
    Calculate the thrust direction
    :param rv: Position vector
    :param vv: Velocity vector
    :param mu: Gravitational constant
    :param a0: acceleration of vehicle at epoch
    :param tau: normalized mass at epoch. This is the mass flow rate
                divided by the mass at the epoch, and therefore has
    :param A: Pitch steering constant A
    :param B: Pitch steering constant B
    :param K: Yaw steering constant K, default does no yaw steering
    :param t: Time since steering constant epoch
    :return: Requested thurst vector
    """
    # The steering program is used to generate as follows:
    # \ddot{r}=a(t)(A+Bt)
    # The control program needs a direction to point, so
    # we also have to figure that out. A steering program
    # which hits the desired vertical acceleration above is:
    # \hat{f}\cdot\hat{r}=A+Bt+\frac{\mu}{a(t)r(t)^2}-\frac{\omega(t)^2r(t)^2}{a(t)}
    # Remembering that this acceleration takes place in a
    # rotating frame with gravity, we have
    # \ddot{r}(t)=-\frac{\mu}{r(t)^2}+\omega(t)^2r(t)+a(t)\left(\hat{f}\cdot\hat{r}\right)
    #            =-\frac{\mu}{r(t)^2}+\omega(t)^2r(t)+a(t)\left(A+Bt+\frac{\mu}{a(t)r(t)^2}-\frac{\omega(t)^2r(t)^2}{a(t)}\right)
    #            =-\frac{\mu}{r(t)^2}+\omega(t)^2r(t)+a(t)A+a(t)Bt+a(t)\frac{\mu}{a(t)r(t)^2}-a(t)\frac{\omega(t)^2r(t)^2}{a(t)}
    #            =-\frac{\mu}{r(t)^2}+\omega(t)^2r(t)+a(t)A+a(t)Bt+\frac{\mu}{r(t)^2}-\omega(t)^2r(t)^2}
    #            =a(t)A+a(t)Bt
    # So, the steering program has terms relating to A and B, as well as terms which
    # cancel out gravity and centrifugal force, removing them from the integration
    # and making things much simpler.
    #
    # There is an optional yaw steering term -- the default value will generate no yaw
    # steering, keeping the thrust in the plane defined by the position and velocity vectors.

    # resolve the basis vectors
    rhat,qhat,hhat=resolve_basis(rv=rv,vv=vv)
    r=vlength(rv) #Distance from center
    vq=vdot(vv,qhat) #Downrange velocity
    omega=vq/r #angular velocity, rad/s. This is used to compute "centrifugal force"
    # Get acceleration at requested time from acceleration at epoch
    at=a(a0=a0,ve=ve,t=t)
    # vertical component of thrust vector
    fdotr=A+B*t+mu/(r**2*at)-(omega**2*r)/(at)
    # crossrange component of thrust vector
    fdoth=K*0
    # downrange component of thrust vector, whatever is left.
    fdotq=np.sqrt(1-fdoth**2-fdotr**2)
    # Compose thrust vector from components and basis vectors
    f=fdotr*rhat+fdotq*qhat+fdoth*hhat
    return f


def predict_vstate(*,rv0:np.array,vv0:np.array,mu:float,
                   A:float,B:float,t:float)->tuple[float,float]:
    """
    Predict the vertical state of the vehicle a given time after the epoch

    :param rv0: position vector at the epoch
    :param vv0: velocity vector at the epoch
    :param mu: gravitational constant
    :param A: Pitch steering coefficient at the epoch
    :param B: Pitch steering coefficient at the epoch
    :param t: Time since the epoch at which to evaluate the prediction, may be an array
    :return:
    """


def b(n:int,*,T:float,ve:float,tau:float)->float:
    if n==0:
        return -ve*ln(1-T/tau)
    else:
        return b(n-1,T=T,ve=ve,tau=tau)*tau-ve*T**n/n


def c(n:int,*,T:float,ve:float,tau:float)->float:
    if n==0:
        return b(0,T=T,ve=ve,tau=tau)*T-b(1,T=T,ve=ve,tau=tau)
    else:
        return c(n-1,T=T,ve=ve,tau=tau)-(ve*T**(n+1))/(n*(n+1))


def calcAB(*,rv:np.array,vv:np.array,
             rT:float,rdotT:float,
             a0:float,ve:float,
             T:float)->tuple[float,float]:
    """
    Calculate the steering constants. These constants will enable steering to the vertical state target
    at the given burnout time.

    :param oldA: Old value of steering constant A
    :param oldB: Old value of steering constant B
    :param T: Current estimate of time-to-go
    :param
    :return: Tuple of (newA,newB)
    """
    rhat,qhat,hhat=resolve_basis(rv=rv,vv=vv)
    r=vlength(rv)
    rdot=vdot(rhat,vv)
    kb=rdotT-rdot
    kc=rT-r-rdot*T
    tau0=tau(a0=a0,ve=ve)
    b0=b(0,T=T,ve=ve,tau=tau0)
    b1=b(1,T=T,ve=ve,tau=tau0)
    c0=c(0,T=T,ve=ve,tau=tau0)
    c1=c(1,T=T,ve=ve,tau=tau0)
    newB=(kc*b0-c0*kb)/(c1*b0-c0*b1)
    newA=kb/b0-b1*newB/b0
    return newA,newB


def calcT(*,rv:np.array,vv:np.array,
          rT:float,vqT:float,
          a0:float,ve:float,
          mu:float,
          A:float,B:float,oldT:float)->float:
    rhat,qhat,hhat=resolve_basis(rv=rv,vv=vv)
    r=vlength(rv)
    hT=rT*vqT
    hv=vcross(rv,vv)
    h=vlength(hv)
    dh=hT-h
    rbar=(rT+r)/2.0
    vq=vdot(vv,qhat) #Downrange velocity
    omega=vq/r #angular velocity, rad/s. This is used to compute "centrifugal force"
    omegaT=vqT/rT #angular velocity, rad/s. This is used to compute "centrifugal force"
    aT=a(a0=a0,ve=ve,t=oldT)
    fr=A+0*B+mu/(r**2*a0)-omega**2*r/a0
    frT=A+oldT*B+mu/(rT**2*aT)-omegaT**2*rT/aT
    frd=(frT-fr)/oldT
    #No yaw steering for now
    fh=0
    fhT=0
    fhd=(fhT-fh)/oldT
    fq=1-fr**2/2-fh**2/2
    fqd=-fr*frd-fh*fhd
    fqdd=-frd**2/2-fhd**2/2
    tau0=tau(a=a0,ve=ve)
    dvn=dh/rbar+ve*oldT*(fqd+fqdd*tau0)+fqdd*ve*oldT**2/2.0
    dvd=fq+fqd*tau0+fqdd*tau0**2
    dv=dvn/dvd
    newT=tau0*(1-np.exp(-dv/ve))
    return newT


def calcT_int(*,rv:np.array,vv:np.array,
          rT:float,vqT:float,
          a0:float,ve:float,
          mu:float,
          A:float,B:float,oldT:float,fps:int=128,
          verbose:bool=False)->float:
    """
    Calculate the time needed to reach the targets given the current
    steering constants
    :param rv: Position vector at epoch
    :param vv: Velocity vector at epoch
    :param rT: Target altitude
    :param vqT: Target downrange velocity
    :param a0: Acceleration at epoch
    :param ve: Effective exhaust velocity
    :param mu: Gravitational parameter
    :param A: Steering constant at epoch
    :param B: Steering constant at epoch
    :param oldT: Previous estimated time to burnout.
                 Not used in this routine, kept to
                 keep interface compatible with calcT()
    :param fps: "frames per second". Integration time-step is 1/fps.
                Use a power of two, so that the integration timestep
                is perfectly representable in floating-point.
    :return: New time to burnout

    This calculates the same thing as calcT() except it uses
    a numerical integration instead of approximate integration.
    For every time step, it propagates the state vector, calculates
    the steering from the steering constants and the gravitational
    and centrifugal terms, and terminates the integration once the
    target angular momentum is reached, and returns the amount of time
    that took.
    """
    frames=0
    hT=rT*vqT
    rv=rv.copy()
    vv=vv.copy()
    hv = vcross(rv, vv)
    h = vlength(hv)
    if verbose:
        hs=[]
        rs=[]
        vs=[]
        ts=[]
        aa=[]
        rdots=[]
    while h<hT:
        t=frames/fps
        at=a(a0=a0,ve=ve,t=t)
        fhat=steer(rv=rv,vv=vv,mu=mu,t=t,a0=a0,ve=ve,A=A,B=B)
        av_thrust=at*fhat
        av_grav=-rv*mu/vlength(rv)**3
        av=av_thrust+av_grav
        dvv=av/fps
        drv=vv/fps
        vv+=dvv
        rv+=drv
        hv=vcross(rv,vv)
        h=vlength(hv)
        if verbose:
            hs.append(h)
            ts.append(t)
            rs.append(vlength(rv))
            vs.append(vlength(vv))
            aa.append(vlength(av))
            rdots.append(vdot(vv,rv)/vlength(rv))
        frames+=1
    if(verbose):
        plt.figure(1)
        plt.subplot(411)
        plt.plot(ts,hs)
        plt.ylabel('h')
        plt.subplot(412)
        plt.plot(ts,rs)
        plt.ylabel('r')
        plt.subplot(413)
        plt.plot(ts,vs)
        plt.ylabel('v')
        plt.subplot(414)
        plt.plot(ts,rdots)
        plt.ylabel('rdot')
        plt.xlabel('t')
        plt.show()
    return t


def PEG_major_cycle(*,rv:np.array,vv:np.array,
                    A:float,B:float,T:float,
                    ve:float,a0:float,
                    rT:float,rdotT:float,vqT:float,mu:float,
                    dt:float,
                    n_iters:int=5)->tuple[float,float,float]:
    """
    Run the Powered Explicit Guidance major cycle.

    :param rv: Position vector at new epoch
    :param vv: Velocity vector at new epoch
    :param A: Old pitch steering constant A
    :param B: Old pitch steering constant B
    :param T: Old time-to-go
    :param ve: Effective exhaust velocity at new epoch
    :param a0: Acceleration due to thrust at new epoch
    :param rT: Target distance from center
    :param rdotT: Target vertical velocity
    :param vqT: Target downrange velocity
    :param mu: Central body gravitational constant
    :param dt: Time since last step
    :return: Tuple of new values of A, B, and T.
    """
    #Update the t=0 epoch
    T-=dt
    A+=B*dt
    for i_iter in range(n_iters):
        A,B=calcAB(rv=rv,vv=vv,rT=rT,rdotT=rdotT,a0=a0,ve=ve,tau=tau,T=T)
        T=calcT(rv=rv,vv=vv,rT=rT,vqT=vqT,a0=a0,ve=ve,mu=mu,A=A,B=B,oldT=T)



@dataclass
class PEGState:
    A: float
    B: float
    T: float


def peg_guide(*,planet:SpicePlanet,rT:float,rdotT:float,vqT:float,yaw0:float,A0:float,B0:float,T0:float)->Callable[...,np.ndarray]:
    state=PEGState(A=A0,B=B0,T=T0)
    def inner(*, t: float, y: np.ndarray, dt: float, major_step: bool, vehicle: Vehicle)->np.ndarray:
            # Code that uses rT, rdotT, vqT, A, B, and T
            vehicle_thrust=0
            vehicle_mdot=0
            for engine in vehicle.engines:
                this_engine_thrust=engine.thrust10*engine.throttle
                this_engine_mdot=this_engine_thrust/engine.ve0
                vehicle_thrust+=this_engine_thrust
                vehicle_mdot+=this_engine_mdot
            vehicle_ve=vehicle_thrust/vehicle_mdot
            vehicle_a0=vehicle_thrust/vehicle.mass()
            rv=y[:3].reshape(-1,1)
            vv=y[3:].reshape(-1,1)
            state.A,state.B,state.T=PEG_major_cycle(rv=rv,vv=vv,
                                                    A=state.A,B=state.B,T=state.T,
                                                    ve=vehicle_ve,a0=vehicle_a0,
                                                    rT=rT,rdotT=rdotT,vqT=vqT,
                                                    mu=planet.mu,dt=dt)
            # Code that calculates a guidance vector `result`.
            # Resolve the downrange basis vectors, considering the current motion
            # of the rocket which has considerable downrange motion already.
            rhat, qhat, hhat = resolve_basis(rv=rv, vv=vv)
            r = vlength(rv)
            hv = vcross(rv, vv)
            vq = vdot(vv, qhat)  # Downrange velocity
            omega = vq / r  # angular velocity, rad/s. This is used to compute "centrifugal force"
            fdotr=state.A+state.B*0+(planet.mu/r**2-omega**2*r)/vehicle_a0
            fnotdotr=np.sqrt(1-fdotr**2)
            fdotq=fnotdotr*np.cos(np.deg2rad(yaw0))
            fdoth=fnotdotr*np.sin(np.deg2rad(yaw0))
            result=fdotr*rhat+fdotq*qhat+fdoth*hhat
            return result
    return inner()

