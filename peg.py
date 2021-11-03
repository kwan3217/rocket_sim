"""
Foo all the Bars
"""
import numpy as np
import math
from numpy import log as ln
from kwanmath.ode import rk4
from kwanmath.vector import vlength,vdot,rv as get_rv, vv as get_vv, vcomp

def ag(sv,mu):
    rv=get_rv(sv)
    r=vlength(rv)
    return -mu*rv/r**3

class PEG:
    def __init__(self):
        self.r_e=6378137 #Surface radius of Earth -- only really relevant
                         #to caclulate altitude for outside reference
        self.mu=398600.4415e9
        self.T = 250  # Initial estimate of time to go
        self.a0 = 15  # Current acceleration measurement
        self.rdotT = 0  # Target vertical speed
        self.rT = self.r_e + 185000  # Target radius
        self.v_e = 4500  # Typical value for hydrolox upper stage
        self.tau=None
        self.A=0
        self.B=0
        self.r=self.r_e+75000 #Current radius
        self.rdot=1000 #Current vertical speed
        self.qdot=1000 #Current horizontal speed
        self.T_stopsteer=7

    def calc_tau(self):
        self.tau = self.v_e / self.a0

    def a(self, t):
        return self.a0 / (1 - t / self.tau)

    def b(self, n, t=None):
        r"""
        Velocity steering integral b_n(t), equation 7a and 7b in original document

        $$\begin{eqnarray*}
        b_0(T)&=& \int_0^T    a(t) dt \\
              &=& -v_e\ln\left(1-\frac{T}{\tau}\right)\tag{7a}\\
              &=& \Delta v\\
        b_n(T)&=& \int_0^T t^na(t) dt \\
              &=& b_{n-1}(T)\tau-\frac{v_eT^n}{n}\tag{7b}
        \end{eqnarray*}$$

        $n=0$ gives ideal $\delta v$ over given time. $n>0$ gives moments, needed by other
        equations to calculate the end conditions. The equations as given in the original
        document use burnout time $T$ as the argument, but this is valid for any time
        between $t=0$ (now) and $t=T$ (burnout).

        :param n: moment of steering integral to calculate.
        :param t: Time from present at which to evaluate integral. Default value is self.T (time to burnout)
        :uses tau:
        :uses v_e:
        :return: nth moment of delta-v
        """
        if t is None:
            t=self.T
        if n == 0:
            result=-self.v_e * ln(1 - t / self.tau)
        else:
            result=self.b(n - 1, t) * self.tau - self.v_e * t ** n / n
        return result

    def c(self, n, t=None):
        """
        Distance steering integral c_n(t), equation 7c and 7d in original document.

        $$\begin{eqnarray*}
        c_0(T)&=&\int_0^T \int_0^t a(s) ds dt\\
              &=&b_0(T)T-b_1(T)\tag{7c}\\
        c_n(T)&=&\int_0^T \int_0^t s^n a(s) ds dt \\
              &=&c_{n-1}\tau-\frac{v_eT^{n+1}}{n(n+1)}\tag{7d}
        \end{eqnarray*}$$

        The zero order moment $c_0(t)$ is the integral of $b_0(s)$ from $s=0$
        to $s=t$, and is therefore the distance traveled by the powered object,
        neglecting outside forces, relative to an unpowered object with the same
        initial state. Higher order moments are needed to calculate end
        conditions. The equations as given in the original document use burnout
        time $T$ as the argument, but this is valid for any time between $t=0$
        (now) and $t=T$ (burnout).

        :param n: moment of steering integral to calculate.
        :param t: Time from present at which to evaluate integral. Default value is self.T (time to burnout)
        :return:
        """
        if t is None:
            t=self.T
        if n == 0:
            result=self.b(0, t) * t - self.b(1, t)
        else:
            result=self.c(n - 1, t) * self.tau - self.v_e * t ** (n + 1) / (n * (n + 1))
        return result

    def calculate_steering(self):
        """
        Given the current vertical state r and rdot, and current estimate of burn time
        remaining T, update the steering constants A and B

        This is done by solving the explicit guidance equations using Cramer's rule
        rdotT=rdot+b0*A+b1*A
        rT=r+rdot*T+c0*A+c1*B

        Getting the A and B terms on one side gives:
        kb=rdotT-rdot=b0*A+b1*B
        kc=rT-r-rdot*T=c0*A+c1*B

        This is formed into matrices and vectors to apply linear algebra
        [A][x]=[B]
        [A]=[b0,b1]
            [c0,c1]
        [x]=[A]
            [B]
        [B]=[kb]
            [kc]
        Watch out for the notation collision here -- B is not the
        same as [B] or b.

        d=b0*c1-b1*c0 #determinant of [A]
        nA=kb*c1-b1*kc
        nB=b0*kc-kb*c0

        A=nA/d
        B=nB/d
        """
        if self.T<self.T_stopsteer:
            # Stop calculating steering constants when less than 7 seconds to go
            return
        #Solve the explicit guidance equations
        kb=self.rdotT-self.rdot
        kc=self.rT-self.r-self.rdot*self.T
        b0=self.b(0)
        b1=self.b(1)
        c0=self.c(0)
        c1=self.c(1)
        d=b0*c1-b1*c0 #determinant of [A]
        nA=kb*c1-b1*kc #determinant of [A] with left column replaced by [B]
        nB=b0*kc-kb*c0 #determinant of [A] with right column replaced by [B]
        self.A=nA/d
        self.B=nB/d

    def estimate_tgo(self):
        """
        Given the current state and steering constants, update the time to go
        :return:
        """
        pass

    def project_steering(self, dt):
        self.A=self.A+self.B*dt
        self.T=self.T-dt

    def fly(self,dt=2.0, ddt=0.1, fps=10):
        """

        Given current steering constants, fly the rocket for one major cycle
        and update the rocket state

        :param dt: Major cycle length, can be any finite number, not necessarily
                   infinitesimal
        :param ddt: Numerical integrator step length, intended to be infinitesimal,
                    use as small a value as possible when balanced against roundoff
                    error and computation time. Doesn't have to be used, if not
                    using a numerical integrator.


        :sets r:
        :sets rdot:
        :return: None
        """
        #In the base case, since predict_r and predict_rdot are not approximations, we
        #can use them as-is to update the vertical state.
        r=self.predict_r(dt)
        rdot=self.predict_rdot(dt)
        def F(t,y):
            """
            Derivative of state vector with respect to time. This takes into
            account two-body gravity and powered flight steering.

            :param t: Input time
            :param y: Input state vector
            :return: Derivative of each element of the state vector in a form
            suitable for ODE solvers.
            """
            x,y,dx,dy=y #State vector elements
            rv=np.array([[x],[y]]) #position vector
            vv=np.array([[dx],[dy]]) #velocity vector
            r=vlength(rv) #radial distance
            v=vlength(vv) #speed
            rhat=rv/r #radial direction
            vhat=vv/v #speed direction
            vprojr=vdot(vv,rhat)*rhat #projection of speed in radial direction
            rdot=vlength(vprojr) #vertical speed
            vprojq=vv-vprojr     #projection of speed perpendicular to radial direction
            qdot=vlength(vprojq) #horizontal speed
            qhat=vprojq/qdot     #horizontal direction
            omega=qdot/r         #angular velocity
            #verification
            if not abs(vdot(vprojr,vprojq))<0.001:
                print("About to raise assertion error")
            assert abs(vdot(vprojr,vprojq))<0.001
            assert math.isclose(v**2,qdot**2+rdot**2)
            grav=self.mu/r**2    #gravity acceleration magnitude
            cent=omega**2*r      #centrifugal acceleration magnitude
            a=self.a(t) #Magnitude of thrust acceleration
            fdotr=self.A+self.B*t+(grav-cent)/a #Component of thrust direction in r direction
            fdotq=np.sqrt(1-fdotr**2)           #component of thrust direction in q direction
            fhat=fdotr*rhat+fdotq*qhat          #Thrust direction vector
            f=fhat*a  #Thrust acceleration vector
            dv=-rhat*grav+f
            ddx=dv[0][0]
            ddy=dv[1][0]
            return np.array([dx,dy,ddx,ddy])
        x0=np.array([self.r,0,self.rdot,self.qdot])
        _,(x1,y1,dx1,dy1)=rk4(F=F,t0=0,y0=x0,t1=dt,dt=ddt,fps=fps)
        rv=vcomp((x1,y1))
        vv=vcomp((dx1,dy1))
        rhat=rv/vlength(rv)
        vprojr = vdot(vv, rhat) * rhat  # projection of speed in radial direction
        rdot = vlength(vprojr)  # vertical speed
        vprojq = vv - vprojr  # projection of speed perpendicular to radial direction
        qdot = vlength(vprojq)  # horizontal speed
        qhat = vprojq / qdot  # horizontal direction
        omega = qdot / r  # angular velocity
        self.r=vlength(rv)
        self.rdot=rdot
        self.qdot=qdot
        print(self)

    def predict_r(self,t):
        """
        Predict the radius distance at some arbitrary time in the future.


        :param t: Time from now. Can be anything, but should be 0<t<T. No guarantee
        of there being fuel remaining if t>T, and there is a singularity at t=tau
        :uses r:
        :uses rdot:
        :return: radius distance

        Given the assumptions of the problem (two-body gravity with no J2
        and atmosphere, single-stage rocket with constant thrust and v_e),
        there are no approximations. This should be perfect.

        """
        result=self.r+self.rdot*t+self.c(0,t)*self.A+self.c(1,t)*self.B
        return result

    def predict_rdot(self,t):
        """
        Predict the vertical state at some arbitrary time in the future

        :param t:
        :return:
        """
        result=self.rdot+self.b(0,t)*self.A+self.b(1,t)*self.B
        return result

def main():
    pass


if __name__ == "__main__":
    main()
