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
from typing import BinaryIO, TextIO

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
    init_spice, horizons_et, voyager_et0, best_pm_solution,best_pm_initial_guess


class Targeter:
    def __call__(self,params)->float:
        """
        This is the interface to scipy.optimize.minimize. It takes a set
        of parameters, calculates the cost, and returns it.
        :return:
        """
        raise NotImplementedError


class PMTargeter(Targeter):
    pass


def sim_pm(*,dpitch:float=0.0, dthr:float=0.0, dyaw:float=0.0, yawrate:float=0.0, fps:int=100, ouf:TextIO=None,vgr_id:int=1):
    horizons_t1 = horizons_et[vgr_id] - voyager_et0[vgr_id]
    y1 = np.array(horizons_data[vgr_id][3:9]) * 1000.0  # Convert km to m and km/s to m/s
    sc = Titan3E(tc_id=5+vgr_id)
    initstate=(f"""Voyager {vgr_id} backpropagation through PM burn
 dpitch: {dpitch:.13e} ({dpitch.hex()})
 dthr: {dthr:.13e} ({dthr.hex()})
 dyaw: {dyaw:.13e} ({dyaw.hex()})
 yawrate: {yawrate:.13e} ({yawrate.hex()})
 fps: {fps} 
 Initial state (simt, ICRF, SI): 
  simt=0 in ET: {voyager_et0[vgr_id]:.13e}  ({voyager_et0[vgr_id].hex()}) ({timout(voyager_et0[vgr_id],'YYYY-MM-DDTHR:MN:SC.### ::TDB')} TDB,{timout(voyager_et0[vgr_id],'YYYY-MM-DDTHR:MN:SC.### ::UTC')}Z)
  simt:  {horizons_t1: .13e}  rx:   {y1[0]: .13e}  ry:   {y1[1]: .13e}  rz:   {y1[2]: .13e}
        {horizons_t1.hex()}       {  y1[0].hex()}      {  y1[1].hex()}      {  y1[2].hex()}
                               vx:   {y1[3]: .13e}  vy:   {y1[4]: .13e}  vz:   {y1[5]: .13e}
                                    {  y1[3].hex()}       {  y1[4].hex()}       {  y1[5].hex()}""")
    print(initstate)
    sc.guide = seq_guide({sc.tdrop[sc.i_centaur]: prograde_guide,
                          float('inf'): yaw_rate_guide(r0=y1[:3], v0=y1[3:],
                                                       dpitch=dpitch, dyaw=dyaw, yawrate=yawrate,
                                                       t0=sc.tburn[sc.i_pmengine][0])})
    # Tweak engine efficiency
    sc.engines[sc.i_pmengine].eff= 1.0 + dthr
    earth_twobody = SpiceTwoBody(spiceid=399)
    earth_j2 = SpiceJ2(spiceid=399)
    moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=voyager_et0[vgr_id])
    sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=voyager_et0[vgr_id])
    sim = Universe(vehicles=[sc], accs=[earth_twobody, earth_j2, moon, sun], t0=horizons_t1, y0s=[y1], fps=fps)
    # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
    sc.stages[sc.i_centaur].prop_mass=0
    sc.stages[sc.i_pm].prop_mass=0
    sim.runto(t1=simt_track_prePM[vgr_id])
    finalstate=f"""Final state (simt, ICRF, SI): 
  simt:  {sc.tlm_points[-1].t: .13e}  rx:   {sc.tlm_points[-1].y0[0]: .13e}  ry:   {sc.tlm_points[-1].y0[1]: .13e}  rz:   {sc.tlm_points[-1].y0[2]: .13e}
        {sc.tlm_points[-1].t.hex()}       {  sc.tlm_points[-1].y0[0].hex()}      {  sc.tlm_points[-1].y0[1].hex()}      {  sc.tlm_points[-1].y0[2].hex()}
                               vx:   {sc.tlm_points[-1].y0[3]: .13e}  vy:   {sc.tlm_points[-1].y0[4]: .13e}  vz:   {sc.tlm_points[-1].y0[5]: .13e}
                                    {  sc.tlm_points[-1].y0[3].hex()}       {  sc.tlm_points[-1].y0[4].hex()}       {  sc.tlm_points[-1].y0[5].hex()}"""
    print(finalstate)
    if ouf is not None:
        print(initstate,file=ouf)
        print(finalstate,file=ouf)
    ts = np.array([x.t for x in sc.tlm_points])
    states = np.array([x.y0 for x in sc.tlm_points])
    masses = np.array([x.mass for x in sc.tlm_points])
    #thrusts = np.array([vlength(x.F_thr) for x in sc.tlm_points])
    #accs = thrusts / masses
    elorbs = [elorb(x.y0[:3], x.y0[3:], l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM, t0=x.t) for x in sc.tlm_points]
    eccs = np.array([elorb.e for elorb in elorbs])
    incs = np.array([np.rad2deg(elorb.i) for elorb in elorbs])
    c3s = -(vehicle.voyager.EarthGM / (1000 ** 3)) / (np.array([elorb.a for elorb in elorbs]) / 1000)  # work directly in km
    plt.figure(f"Voyager {vgr_id} PM burn")
    plt.subplot(231)
    plt.cla()
    plt.ylabel('spd/(m/s)')
    plt.xlabel('simt/s')
    plt.plot(ts, np.linalg.norm(states[:, 3:6], axis=1), label='spd')
    plt.subplot(232)
    plt.cla()
    plt.ylabel('ecc')
    plt.xlabel('simt/s')
    plt.plot(ts, eccs, label='e')
    plt.subplot(233)
    plt.cla()
    plt.ylabel('inc/deg')
    plt.xlabel('simt/s')
    plt.plot(ts, incs, label='i')
    plt.subplot(234)
    plt.cla()
    plt.ylabel('a/nmi')
    plt.xlabel('simt/s')
    plt.plot(ts, [elorb.a/1852 for elorb in elorbs], label='a')
    plt.subplot(235)
    plt.cla()
    plt.ylabel('mass/kg')
    plt.xlabel('simt/s')
    plt.plot(ts, masses, label='i')
    return sc


def sim_centaur2(*,simt1:float,y1:np.ndarray,dpitch:float, dthr:float, dyaw:float, pitchrate:float, fps1:int=100, fps0:int=1, verbose:bool=True,vgr_id:int=1):
    """

    :param t1: Initial time. Note that this is a backpropagation, so this is the higest time number and the
               numerical integrator will work towards the past.
    :param y1: Initial state vector in SI, Earth-centered, ICRF frame
    :param dpitch: Change in pitch from prograde
    :param dthr: Change in engine efficiency
    :param dyaw: Change in yaw from prograde
    :param fps1: Frames per second to run the burn
    :param fps0: Frames per second to run the parking orbit backpropagation
    :param verbose: If set, make some plots
    :param vgr_id: Which voyager are we working on?
    :return:
    """
    sc = Titan3E(tc_id=5+vgr_id)
    print(f"Voyager {vgr_id} backpropagation through Centaur burn 2")
    print(f" dpitch: {dpitch:.13e} ({dpitch.hex()})")
    print(f" dthr: {dthr:.13e} ({dthr.hex()})")
    print(f" dyaw: {dyaw:.13e} ({dyaw.hex()})")
    print(f" pitchrate: {pitchrate:.13e} ({pitchrate.hex()})")
    print(f" fps during burn: {fps1} ")
    print(f" fps during parking orbit: {fps0} ")
    print(" Initial state (simt, ICRF, SI): ")
    print(f"  simt=0 in ET: {voyager_et0[vgr_id]:.13e}  ({voyager_et0[vgr_id].hex()}) "
          f"({timout(voyager_et0[vgr_id],'YYYY-MM-DDTHR:MN:SC.### ::TDB')} TDB,{timout(voyager_et0[vgr_id],'YYYY-MM-DDTHR:MN:SC.### ::UTC')}Z)")
    print(f"  simt:  {simt1: .13e}  rx:   {y1[0]: .13e}  ry:   {y1[1]: .13e}  rz:   {y1[2]: .13e}")
    print(f"        {simt1.hex()}       {  y1[0].hex()}      {  y1[1].hex()}      {  y1[2].hex()}")
    print(f"                               vx:   {y1[3]: .13e}  vy:   {y1[4]: .13e}  vz:   {y1[5]: .13e}")
    print(f"                                    {  y1[3].hex()}       {  y1[4].hex()}       {  y1[5].hex()}")
    sc.guide = dprograde_guide(dpitch=dpitch, dyaw=dyaw, pitchrate=pitchrate,t0=sc.tburn[sc.i_cengine2][0])
    # Tweak engine efficiency
    sc.engines[sc.i_cengine2].eff=1.0+dthr
    earth_twobody = SpiceTwoBody(spiceid=399)
    earth_j2 = SpiceJ2(spiceid=399)
    moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=voyager_et0[vgr_id])
    sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=voyager_et0[vgr_id])
    sim = Universe(vehicles=[sc], accs=[earth_twobody, earth_j2, moon, sun], t0=simt1, y0s=[y1], fps=fps1)
    # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
    sc.stages[sc.i_centaur].prop_mass=0
    sim.runto(t1=sc.tburn[sc.i_cengine2][0]-10)
    print("State just prior to Centaur burn 2 (simt, ICRF, SI): ")
    print(f"  simt:  {sc.tlm_points[-1].t: .13e}  rx:   {sc.tlm_points[-1].y0[0]: .13e}  ry:   {sc.tlm_points[-1].y0[1]: .13e}  rz:   {sc.tlm_points[-1].y0[2]: .13e}")
    print(f"        {sc.tlm_points[-1].t.hex()}       {  sc.tlm_points[-1].y0[0].hex()}      {  sc.tlm_points[-1].y0[1].hex()}      {  sc.tlm_points[-1].y0[2].hex()}")
    print(f"                               vx:   {sc.tlm_points[-1].y0[3]: .13e}  vy:   {sc.tlm_points[-1].y0[4]: .13e}  vz:   {sc.tlm_points[-1].y0[5]: .13e}")
    print(f"                                    {  sc.tlm_points[-1].y0[3].hex()}       {  sc.tlm_points[-1].y0[4].hex()}       {  sc.tlm_points[-1].y0[5].hex()}")
    sim.change_fps(fps0)
    sim.runto(t1=simt_track_park[vgr_id])
    print("State just after Centaur burn 1 (simt, ICRF, SI): ")
    print(f"  simt:  {sc.tlm_points[-1].t: .13e}  rx:   {sc.tlm_points[-1].y0[0]: .13e}  ry:   {sc.tlm_points[-1].y0[1]: .13e}  rz:   {sc.tlm_points[-1].y0[2]: .13e}")
    print(f"        {sc.tlm_points[-1].t.hex()}       {  sc.tlm_points[-1].y0[0].hex()}      {  sc.tlm_points[-1].y0[1].hex()}      {  sc.tlm_points[-1].y0[2].hex()}")
    print(f"                               vx:   {sc.tlm_points[-1].y0[3]: .13e}  vy:   {sc.tlm_points[-1].y0[4]: .13e}  vz:   {sc.tlm_points[-1].y0[5]: .13e}")
    print(f"                                    {  sc.tlm_points[-1].y0[3].hex()}       {  sc.tlm_points[-1].y0[4].hex()}       {  sc.tlm_points[-1].y0[5].hex()}")
    ts = np.array([x.t for x in sc.tlm_points])
    states = np.array([x.y0 for x in sc.tlm_points]).T
    masses = np.array([x.mass for x in sc.tlm_points])
    accs = np.array([x.F_thr for x in sc.tlm_points]).T/masses
    elorbs = [elorb(x.y0[:3], x.y0[3:], l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM, t0=x.t) for x in sc.tlm_points]
    eccs = np.array([elorb.e for elorb in elorbs])
    incs = np.array([np.rad2deg(elorb.i) for elorb in elorbs])
    c3s = -(vehicle.voyager.EarthGM / (1000 ** 3)) / (np.array([elorb.a for elorb in elorbs]) / 1000)  # work directly in km
    plt.figure(1)
    plt.clf()
    plt.subplot(2,3,1)
    plt.plot(ts, vlength(states[3:6, :]), label='spd')
    plt.ylabel("spd/(m/s)")
    plt.subplot(2,3,2)
    plt.plot(ts, eccs, label='e')
    plt.ylabel("e")
    plt.subplot(2,3,3)
    plt.plot(ts, incs, label='i')
    plt.ylabel("inc/deg")
    plt.subplot(2,3,4)
    plt.plot(ts, c3s, label='c3')
    plt.ylabel('$C_3$/(km**2/s**2)')
    plt.subplot(2,3,5)
    plt.plot(ts, vlength(accs)/g0, label='accs')
    plt.ylabel("acc/g")
    plt.pause(0.1)
    return sc


def pm_cost(*,dpitch:float, dthr:float, dyaw:float, sc:Vehicle, vgr_id,ouf:TextIO|None):
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
    y0_j = sc.y.copy()
    Mej = pxform('J2000', 'IAU_EARTH', simt_track_prePM[vgr_id] + voyager_et0[vgr_id])
    r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
    v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
    r0_e = Mej @ r0_j
    v0_e = Mej @ v0_j
    # In this frame, the reported ascending node is Longitude of ascending node, not
    # right ascension. It is relative to the Earth-fixed coordinates at this instant,
    # not the sky coordinates.
    elorb0 = elorb(r0_e, v0_e, l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM, t0=simt_track_prePM[vgr_id],
                   deg=True)
    da = (target_a_prePM[vgr_id] - elorb0.a) / 1852  # da in nmi
    de = (target_e_prePM[vgr_id] - elorb0.e)  # de
    di = (target_i_prePM[vgr_id] - elorb0.i)  # di in deg
    dlan = (target_lan_prePM[vgr_id] - elorb0.an)  # dlan in deg -- not used for targeting
    sa = 2e-1  # difference between Antigua and Guidance measurement
    se = 0.00007  # likewise
    si = 0.01  # likewise
    # So the cost is the sum of squares of each difference
    # expressed as multiples of the uncertainty
    cost = ((da / sa) ** 2 + (de / se) ** 2 + (di / si) ** 2)
    plt.subplot(2, 3, 2)
    trange = [sc.tlm_points[0].t, sc.tlm_points[-1].t + sc.tlm_points[-1].dt]
    plt.plot(trange, target_e_prePM[vgr_id] * np.array([1, 1]), 'k--')
    plt.subplot(2, 3, 3)
    plt.plot(trange, target_i_prePM[vgr_id] * np.array([1, 1]), 'k--', label='target J1977')
    plt.plot(trange, elorb0.i * np.array([1, 1]), 'r--', label='achieved J1977')
    plt.legend()
    plt.subplot(2, 3, 4)
    plt.plot(trange, target_a_prePM[vgr_id] / 1852 * np.array([1, 1]), 'k--')
    plt.subplot(2, 3, 6)
    plt.cla()
    coststr = f"""
{dyaw=:.6f} deg, {dpitch=:.6f} deg, {dthr=:.6f}
   ahist={target_a_prePM[vgr_id] / 1852:.2f} nmi, acalc={elorb0.a / 1852:.2f} nmi, {da=:.2f} nmi, {da / sa:8.1f} sigma
   ehist={target_e_prePM[vgr_id]:.6f},    ecalc={elorb0.e:.6f},    {de=:.6f},  {de / se:8.1f} sigma
   ihist={target_i_prePM[vgr_id]:.4f} deg, icalc={elorb0.i:.4f} deg, {di=:.4f} deg, {di / si:8.1f} sigma
   lanhist={target_lan_prePM[vgr_id]:.4f} deg, lancalc={elorb0.an:.4f} deg, {dlan=:.4f} deg
Cost: {cost}"""
    print(coststr)
    if ouf is not None:
        print(coststr,file=ouf)
    plt.text(0, 0.5, coststr, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.pause(0.1)
    return cost


def opt_interface_pm_burn(*,target:np.ndarray=None, fps:int=100, vgr_id:int=1,ouf:TextIO|None)->float:
    """
    Calculate the "cost" of a set of targeting parameters by
    walking back through the PM maneuver from a known state.
    Cost is related to difference of a, e, i from documented
    pre-PM state.
    :param target: Array containing difference in degrees between latitude
                   (index 0) and longitude (index 1) of target inertial
                   position away from prograde at Horizons data point.
                   (0,0) is a valid initial guess that results in prograde
                   at Horizons.
    :return: squared length of vector whose length is weighted error in a, e, and i.
             This is zero if the target is hit, and greater than zero for any miss.
    """
    dpitch, dyaw, dthr,yawrate = target
    sc = sim_pm(dpitch=dpitch, dthr=dthr, dyaw=dyaw, fps=fps, yawrate=yawrate, vgr_id=vgr_id)
    cost = pm_cost(dpitch=dpitch, dthr=dthr, dyaw=dyaw, sc=sc, vgr_id=vgr_id,ouf=ouf)
    return cost


def opt_interface_centaur2(target:np.ndarray=None,*,simt1:float,y1:np.ndarray,verbose:bool=False, fps1:int=10, fps0=1, vgr_id:int=1)->float:
    """
    :param target: Array containing difference in degrees between latitude
                   (index 0) and longitude (index 1) of target inertial
                   position away from prograde at Horizons data point.
                   (0,0) is a valid initial guess that results in prograde
                   at Horizons.
    :return: squared length of vector whose length is weighted error in a, e, and i.
             This is zero if the target is hit, and greater than zero for any miss.
    """
    dpitch, dyaw, dthr, pitchrate = target
    sc = sim_centaur2(simt1=simt1,y1=y1,dpitch=dpitch, dthr=dthr, dyaw=dyaw, pitchrate=pitchrate,
                      fps1=fps1, fps0=fps0, verbose=verbose, vgr_id=vgr_id)
    y0_j = sc.y.copy()
    Mej = pxform('J2000', 'IAU_EARTH', simt_track_park[vgr_id] + voyager_et0[vgr_id])
    r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
    v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
    r0_e = Mej @ r0_j
    v0_e = Mej @ v0_j
    # In this frame, the reported ascending node is Longitude of ascending node, not
    # right ascension. It is relative to the Earth-fixed coordinates at this instant,
    # not the sky coordinates.
    elorb0 = elorb(r0_e, v0_e, l_DU=vehicle.voyager.EarthRe, mu=vehicle.voyager.EarthGM, t0=simt_track_prePM[vgr_id], deg=True)
    # Put each error into the units of the historical report
    da = (target_a_park[vgr_id] - elorb0.a) / 1852  # Use nautical miles to match significant figures
    de = (target_e_park[vgr_id] - elorb0.e)
    di = (target_i_park[vgr_id] - elorb0.i)
    # Divide each difference by estimated uncertainty of historical parameter. It might
    # be the size of the least significant digit, or might be guided by estimated
    # accuracy indicated by spread in different measurements. In any case, 1.0 means
    # different by about the estimated uncertainty. The cost then is the sum of squares
    # of each parameter difference normalized by this spread. This very nearly matches
    # the chi-square measure of fit in curve fitting, which is in a sense what we are
    # doing here.
    sa=2e-1 #difference between Antigua and Guidance measurement
    se=0.00007 #likewise
    si=0.01 # likewise
    # So the cost is the sum of squares of each difference
    # expressed as multiples of the uncertainty
    cost = ((da/sa) ** 2 + (de/se) ** 2 + (di/si)** 2)
    print(f"{dyaw=} deg, {dpitch=} deg, {dthr=}")
    print(f"   ahist={target_a_park[vgr_id]/1852:.2f} nmi, acalc={elorb0.a/1852:.2f} nmi, {da=:.2f} nmi, {da/sa:8.1f} sigma")
    print(f"   ehist={target_e_park[vgr_id]:.6f},    ecalc={elorb0.e:.6f},    {de=:.6f},  {de/se:8.1f} sigma")
    print(f"   ihist={target_i_park[vgr_id]:.4f} deg, icalc={elorb0.i:.4f} deg, {di=:.4f} deg, {di/si:8.1f} sigma")
    print(f"Cost: {cost}")
    return cost


def target_pm(export:bool=False, optimize:bool=False, vgr_id:int=1):
    # Parameters are:
    #   0 - dpitch, pitch angle between pure prograde and target in degrees
    #   1 - dyaw, yaw angle between pure in-plane and target in degrees
    #   2 - dthr, fractional difference in engine efficiency. 0 is 100%,
    #       +0.1 is 110%, -0.1 is 90%, etc. This value changes thrust10 and ve
    #       simultaneously so as to not affect mdot, so the propellant
    #       will drain exactly as fast as before.
    #   3 - yawrate - change in yaw vs time in deg/s.
    initial_guess= best_pm_initial_guess(vgr_id=vgr_id)
    bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(0,0)]  # Freeze yaw rate at 0
    #initial_guess = [-1.438, 13.801, +0.00599, -0.549]  # From previous four-parameter form
    #bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(-1,1)]  # Bounds on
    if optimize:
        result = minimize(lambda x:opt_interface_pm_burn(target=x,vgr_id=vgr_id,ouf=None),
                          initial_guess,
                          method='L-BFGS-B',
                          options={'ftol':1e-4,'disp':True},
                          bounds=bounds)
        print("Achieved cost:", result.fun)
        print(result)
        final_guess=result.x
    else:
        final_guess=initial_guess
    print("Optimal parameters:", final_guess)
    print("Optimal run: ")
    dpitch,dyaw,dthr,yawrate=final_guess
    with open(f'products/vgr{vgr_id}_pm_optimal_run.txt',"wt") as ouf:
        sc=sim_pm(dpitch=dpitch,dyaw=dyaw,dthr=dthr,yawrate=yawrate,vgr_id=vgr_id,ouf=ouf)
        pm_cost(dpitch=dpitch,dthr=dthr,dyaw=dyaw,sc=sc,vgr_id=vgr_id,ouf=ouf)
        try:
            print("Optimizer results:",result,file=ouf)
        except Exception:
            # No optimizer result
            pass
    if export:
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
    return sc


def target_centaur2(*,simt1:float,y1:np.ndarray,export:bool=False, fps1:int=10,fps0:int=1,optimize:bool=False, vgr_id:int=1):
    # Parameters are:
    #   0 - dpitch, pitch angle between pure prograde and target in degrees
    #   1 - dyaw, yaw angle between pure in-plane and target in degrees
    #   2 - dthr, fractional difference in engine efficiency. 0 is 100%,
    #       +0.1 is 110%, -0.1 is 90%, etc. This value changes thrust10 and ve
    #       simultaneously so as to not affect mdot, so the propellant
    #       will drain exactly as fast as before.
    #initial_guess=np.zeros(4)
    #                dpitch               dyaw                   dthr                pitchrate
    initial_guess={1:[-4.4164552842503e+00,-4.3590393660760e-02,-9.6115297747416e-03,4.8457186584881e-03], #Best Voyager 1 result
                   2:[-1.3433920246681e+01, 9.0870458935636e+00,-2.2020571000027e-02,5.1374694020877e-02]}[vgr_id] #Best Voyager 2 result

    bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(-1,1)]  # Freeze yaw rate at 0
    if optimize:
        result = minimize(lambda params:opt_interface_centaur2(params,simt1=simt1,y1=y1,fps1=fps1,fps0=fps0,vgr_id=vgr_id,verbose=True), initial_guess,
                          method='L-BFGS-B', options={'ftol':1e-12,'gtol':1e-12,'disp':True}, bounds=bounds)
        print("Achieved cost:", result.fun)
        final_guess=result.x
    else:
        final_guess=initial_guess
    print("Optimal parameters:", final_guess)
    print("Optimal run: ")
    sc=sim_centaur2(simt1=simt1,y1=y1,**{k:v for k,v in zip(('dpitch','dyaw','dthr','pitchrate'),final_guess)},fps1=fps1,fps0=fps0,verbose=True,vgr_id=vgr_id)
    if export:
        states=sorted([np.hstack((np.array(x.t),x.y0)) for x in sc.tlm_points], key=lambda x:x[0])
        decimated_states=[]
        i=0
        di=1
        done=False
        while not done:
            states[i][0]+=voyager_et0[vgr_id]
            decimated_states.append(states[i])
            try:
                if sc.t_pm0<states[i+di][0]<sc.t_pm1:
                    di=1
                else:
                    di=100
            except IndexError:
                break
            i+=di
        mkspk(oufn=f'products/vgr{sc.vgr_id}_centaur2.bsp',
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
    return sc


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
    vgr_id=1
    #target_pm(export=False, optimize=True,vgr_id=vgr_id)
    pm=best_pm_solution(vgr_id=vgr_id)
    target_centaur2(simt1=pm.simt0,y1=pm.y0,fps1=10, optimize=True, export=True,vgr_id=vgr_id)
    #for vgr_id in (1,2):
    #    export(vgr_id)


if __name__=="__main__":
    main()