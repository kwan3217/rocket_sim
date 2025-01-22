"""
Foo all the Bars
"""

import pytest
import peg
import numpy as np
import matplotlib.pyplot as plt

def test_fly():
    fly=peg.PEG()
    fly.calc_tau()
    fly.calc_ab()
    pred=peg.PEG()
    pred.calc_tau()
    pred.calc_ab()

    t=0
    dt=2.0
    pred_results=[]
    dtype=[('t',np.float64),('A', np.float64), ('B', np.float64), ('T', np.float64), ('r', np.float64),
                             ('rdot', np.float64),('qdot',np.float64)]
    pred_result=np.array([(t,pred.A,pred.B,pred.T,pred.predict_r(t),pred.predict_rdot(t),float('NaN'))],dtype=dtype)
    pred_results.append(pred_result)
    while t<pred.T:
        t+=dt
        pred_result = np.array([(t, pred.A, pred.B, pred.T, pred.predict_r(t), pred.predict_rdot(t),float('NaN'))],dtype = dtype)
        pred_results.append(pred_result)
    pred_results=np.array(pred_results)
    plt.figure('r')
    plt.plot(pred_results['t'],pred_results['r'],label='predicted')
    plt.figure('rdot')
    plt.plot(pred_results['t'],pred_results['rdot'],label='predicted')

    t=0
    results=[]
    result = np.array([(t,fly.A, fly.B, fly.T, fly.r, fly.rdot, fly.qdot)],dtype=dtype)
    results.append(result)
    while fly.T>0:
        fly.fly(dt=dt)
        fly.update_abt(dt=dt)
        t+=dt
        result=np.array([(t,fly.A,fly.B,fly.T,fly.r,fly.rdot,fly.qdot)],dtype=dtype)
        results.append(result)
    results=np.array(results)
    plt.figure('r')
    plt.plot(results['t'],results['r'],label='integrated')
    plt.legend()
    plt.figure('rdot')
    plt.plot(results['t'],results['rdot'],label='integrated')
    plt.legend()
    plt.show()

