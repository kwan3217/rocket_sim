"""

"""
from math import isclose

import numpy as np
from kwanmath.vector import vnormalize, vdot
from matplotlib import pyplot as plt

from guidance.peg_clean_sheet import calcAB, b, c, calcT_int, a


def test_calcAB():
    re=6_378_000.0
    h0=40_000.0
    hT=185_000.0
    rT=re+hT
    rv=np.array([[re+h0],[    0.0],[0.0]])
    vv=np.array([[1_000.0],[1_000.0],[0.0]])
    rhat=vnormalize(rv)
    r=vdot(rhat,rv)
    rdot=vdot(vv,rhat)
    rdotT=0.0
    T=200.0
    ve=4_500.0
    tau=250.0
    A,B=calcAB(rv=rv,vv=vv,mu=398600e9,rT=rT,rdotT=rdotT,ve=ve,tau=tau,T=T,dt=0.125)
    b0=b(0,T=T,ve=ve,tau=tau)
    b1=b(1,T=T,ve=ve,tau=tau)
    c0=c(0,T=T,ve=ve,tau=tau)
    c1=c(1,T=T,ve=ve,tau=tau)
    rdotTp=rdot+b0*A+b1*B
    rTp=r+rdot*T+c0*A+c1*B
    assert isclose(rdotT,rdotTp,abs_tol=0.1)
    assert isclose(rT,rTp,abs_tol=0.1)


def test_calcT_int():
    re=6_378_000.0
    h0=40_000.0
    hT=185_000.0
    rT=re+hT
    rv=np.array([[  re+h0],[    0.0],[0.0]])
    vv=np.array([[1_000.0],[1_000.0],[0.0]])
    rhat=vnormalize(rv)
    r=vdot(rhat,rv)
    rdot=vdot(vv,rhat)
    rdotT=0.0
    T=400.0
    ve=4_500.0
    tau=450.0
    mu=398600e9
    vqT=np.sqrt(mu/rT)
    a0=ve/tau
    Ts=[]
    As=[]
    Bs=[]
    for i in range(10):
        A,B=calcAB(rv=rv,vv=vv,rT=rT,rdotT=rdotT,ve=ve,a0=a0,T=T)
        T=calcT_int(rv=rv,vv=vv,rT=rT,vqT=vqT,a0=a0,ve=ve,mu=mu,A=A,B=B,oldT=T)
        As.append(A)
        Bs.append(B)
        Ts.append(T)
    plt.figure("Convergence")
    plt.subplot(311)
    plt.plot(range(10),np.array(Ts)/Ts[-1])
    plt.ylabel("T/converged T")
    plt.subplot(312)
    plt.plot(range(10),np.array(As)/As[-1])
    plt.ylabel("A/converged A")
    plt.subplot(313)
    plt.plot(range(10),np.array(Bs)/Bs[-1])
    plt.ylabel("B/converged B")
    plt.show()
