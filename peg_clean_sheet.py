"""
PEG on a clean sheet with functions instead of objects
"""

import numpy as np
from kwanmath.vector import vnormalize, vcross, vlength, vdot


def tau(*,a:float,ve:float)->float:
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
    return ve/a

def a(*,a0:float,ve:float,t:float)->float:
    return a0/(1-t/tau(a0,ve))


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
    rhat=vnormalize(rv) #vertical basis vector
    qhat=vnormalize(vv) #downrange basis vector, q for \hat{theta}
    hhat=vnormalize(vcross(rv,vv)) #crossrange basis vector, generally pointing "left" if qhat is considered "forward".
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
        return -ve*np.ln(1-T/tau)
    else:
        return b(n-1,T=T,ve=ve,tau=tau)*tau-ve*T**n/n


def c(n:int,*,T:float,ve:float,tau:float)->float:
    if n==0:
        return b(0,T=T,ve=ve,tau=tau)*T-b(1,T=T,ve=ve,tau=tau)
    else:
        return c(n-1,T=T,ve=ve,tau=tau)-(ve*T**(n+1))/(n*(n+1))


def calcAB(*,rv:np.array,vv:np.array,mu:float,
             rT:float,rdotT:float,
             oldA:float,oldB:float,
             T:float,dt:float)->tuple[float,float]:
    """
    Calculate the steering constants. These constants will enable steering to the vertical state target
    at the given burnout time.

    :param oldA: Old value of steering constant A
    :param oldB: Old value of steering constant B
    :param T: Current estimate of time-to-go
    :param
    :return: Tuple of (newA,newB)
    """
    rhat=vnormalize(rv) #vertical basis vector
    qhat=vnormalize(vv) #downrange basis vector, q for \hat{theta}
    hhat=vnormalize(vcross(rv,vv)) #crossrange basis vector, generally pointing "left" if qhat is considered "forward".
    r=vlength(rv)
    rdot=vdot(rhat,vv)
    kb=rdotT-rdot
    kc=rT-r-rdot*T
