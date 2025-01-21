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

import numpy as np
from bmw import elorb
from kwanspice.mkspk import mkspk
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from spiceypy import str2et, pxform, timout

import voyager
from rocket_sim.gravity import SpiceTwoBody, SpiceJ2, SpiceThirdBody
from rocket_sim.universe import Universe
from voyager import Voyager, horizons_data, prograde_guide, yaw_rate_guide, seq_guide, \
    simt_track_prePM, target_a_prePM, target_e_prePM, target_i_prePM, target_lan_prePM, \
    simt_track_park,  target_a_park,  target_e_park, target_i_park, target_c3_park, \
    init_spice, horizons_et, voyager_et0


def sim_pm(*,dpitch:float=0.0, dthr:float=0.0, dyaw:float=0.0, yawrate:float=0.0, fps:int=100, verbose:bool=True,vgr_id:int=1):
    horizons_t1 = horizons_et[vgr_id] - voyager_et0[vgr_id]
    y1 = np.array(horizons_data[vgr_id][3:9]) * 1000.0  # Convert km to m and km/s to m/s
    sc = Voyager(vgr_id=vgr_id)
    print(f"Voyager {vgr_id} backpropagation through PM burn")
    print(f" dpitch: {dpitch:.13e} ({dpitch.hex()})")
    print(f" dthr: {dthr:.13e} ({dthr.hex()})")
    print(f" dyaw: {dyaw:.13e} ({dyaw.hex()})")
    print(f" yawrate: {yawrate:.13e} ({yawrate.hex()})")
    print(f" fps: {fps} ")
    print(" Initial state (simt, ICRF, SI): ")
    print(f"  simt=0 in ET: {voyager_et0[vgr_id]:.13e}  ({voyager_et0[vgr_id].hex()}) "
          f"({timout(voyager_et0[vgr_id],'YYYY-MM-DDTHR:MN:SC.### ::TDB')} TDB,{timout(voyager_et0[vgr_id],'YYYY-MM-DDTHR:MN:SC.### ::UTC')}Z)")
    print(f"  simt:  {horizons_t1: .13e}  rx:   {y1[0]: .13e}  ry:   {y1[1]: .13e}  rz:   {y1[2]: .13e}")
    print(f"        {horizons_t1.hex()}       {  y1[0].hex()}      {  y1[1].hex()}      {  y1[2].hex()}")
    print(f"                               vx:   {y1[3]: .13e}  vy:   {y1[4]: .13e}  vz:   {y1[5]: .13e}")
    print(f"                                    {  y1[3].hex()}       {  y1[4].hex()}       {  y1[5].hex()}")
    sc.guide = seq_guide({sc.tsep_pm: prograde_guide,
                          float('inf'): yaw_rate_guide(r0=y1[:3], v0=y1[3:],
                                                       dpitch=dpitch, dyaw=dyaw, yawrate=yawrate,
                                                       t0=sc.t_pm0)})
    # Tweak engine efficiency
    sc.engines[sc.i_epm].eff= 1.0 + dthr
    earth_twobody = SpiceTwoBody(spiceid=399)
    earth_j2 = SpiceJ2(spiceid=399)
    moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=voyager_et0[vgr_id])
    sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=voyager_et0[vgr_id])
    sim = Universe(vehicles=[sc], accs=[earth_twobody, earth_j2, moon, sun], t0=horizons_t1, y0s=[y1], fps=fps)
    # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
    sc.stages[sc.i_centaur].prop_mass=0
    sc.stages[sc.i_pm].prop_mass=0
    sim.runto(t1=simt_track_prePM[vgr_id])
    print("Final state (simt, ICRF, SI): ")
    print(f"  simt:  {sc.tlm[-1].t: .13e}  rx:   {sc.tlm[-1].y[0]: .13e}  ry:   {sc.tlm[-1].y[1]: .13e}  rz:   {sc.tlm[-1].y[2]: .13e}")
    print(f"        {sc.tlm[-1].t.hex()}       {  sc.tlm[-1].y[0].hex()}      {  sc.tlm[-1].y[1].hex()}      {  sc.tlm[-1].y[2].hex()}")
    print(f"                               vx:   {sc.tlm[-1].y[3]: .13e}  vy:   {sc.tlm[-1].y[4]: .13e}  vz:   {sc.tlm[-1].y[5]: .13e}")
    print(f"                                    {  sc.tlm[-1].y[3].hex()}       {  sc.tlm[-1].y[4].hex()}       {  sc.tlm[-1].y[5].hex()}")
    if verbose:
        ts = np.array([x.t for x in sc.tlm])
        states = np.array([x.y for x in sc.tlm])
        masses = np.array([x.mass for x in sc.tlm])
        thrusts = np.array([x.thrust for x in sc.tlm])
        accs = thrusts / masses
        elorbs = [elorb(x.y[:3], x.y[3:], l_DU=voyager.EarthRe, mu=voyager.EarthGM, t0=x.t) for x in sc.tlm]
        eccs = np.array([elorb.e for elorb in elorbs])
        incs = np.array([np.rad2deg(elorb.i) for elorb in elorbs])
        smis = np.array([elorb.a for elorb in elorbs]) / 1852  # Display in nmi to match document
        c3s = -(voyager.EarthGM / (1000 ** 3)) / (np.array([elorb.a for elorb in elorbs]) / 1000)  # work directly in km
        plt.figure("spd")
        plt.plot(ts, np.linalg.norm(states[:, 3:6], axis=1), label='spd')
        plt.figure("ecc")
        plt.plot(ts, eccs, label='e')
        plt.figure("inc")
        plt.plot(ts, incs, label='i')
        plt.figure("a")
        plt.plot(ts, smis, label='a')
        plt.ylabel('semi-major axis/nmi')
        plt.figure("c3")
        plt.plot(ts, c3s, label='c3')
        plt.ylabel('$C_3$/(km**2/s**2)')
        plt.figure("mass")
        plt.plot(ts, masses, label='i')
        plt.show()
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
    sc = Voyager(vgr_id=vgr_id)
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
    sc.guide = voyager.dprograde_guide(dpitch=dpitch, dyaw=dyaw, pitchrate=pitchrate,t0=sc.t_cb[(2,0)])
    # Tweak engine efficiency
    for i,(e,b) in sc.i_eb.items():
        if b==2:
            sc.engines[i].eff= 1.0 + dthr
    earth_twobody = SpiceTwoBody(spiceid=399)
    earth_j2 = SpiceJ2(spiceid=399)
    moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=voyager_et0[vgr_id])
    sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=voyager_et0[vgr_id])
    sim = Universe(vehicles=[sc], accs=[earth_twobody, earth_j2, moon, sun], t0=simt1, y0s=[y1], fps=fps1)
    # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
    sc.stages[sc.i_centaur].prop_mass=0
    sim.runto(t1=sc.t_cb[(2,0)]-10)
    print("State just prior to Centaur burn 2 (simt, ICRF, SI): ")
    print(f"  simt:  {sc.tlm[-1].t: .13e}  rx:   {sc.tlm[-1].y[0]: .13e}  ry:   {sc.tlm[-1].y[1]: .13e}  rz:   {sc.tlm[-1].y[2]: .13e}")
    print(f"        {sc.tlm[-1].t.hex()}       {  sc.tlm[-1].y[0].hex()}      {  sc.tlm[-1].y[1].hex()}      {  sc.tlm[-1].y[2].hex()}")
    print(f"                               vx:   {sc.tlm[-1].y[3]: .13e}  vy:   {sc.tlm[-1].y[4]: .13e}  vz:   {sc.tlm[-1].y[5]: .13e}")
    print(f"                                    {  sc.tlm[-1].y[3].hex()}       {  sc.tlm[-1].y[4].hex()}       {  sc.tlm[-1].y[5].hex()}")
    sim.change_fps(fps0)
    sim.runto(t1=simt_track_park[vgr_id])
    print("State just after Centaur burn 1 (simt, ICRF, SI): ")
    print(f"  simt:  {sc.tlm[-1].t: .13e}  rx:   {sc.tlm[-1].y[0]: .13e}  ry:   {sc.tlm[-1].y[1]: .13e}  rz:   {sc.tlm[-1].y[2]: .13e}")
    print(f"        {sc.tlm[-1].t.hex()}       {  sc.tlm[-1].y[0].hex()}      {  sc.tlm[-1].y[1].hex()}      {  sc.tlm[-1].y[2].hex()}")
    print(f"                               vx:   {sc.tlm[-1].y[3]: .13e}  vy:   {sc.tlm[-1].y[4]: .13e}  vz:   {sc.tlm[-1].y[5]: .13e}")
    print(f"                                    {  sc.tlm[-1].y[3].hex()}       {  sc.tlm[-1].y[4].hex()}       {  sc.tlm[-1].y[5].hex()}")
    if verbose:
        ts = np.array([x.t for x in sc.tlm])
        states = np.array([x.y for x in sc.tlm])
        masses = np.array([x.mass for x in sc.tlm])
        thrusts = np.array([x.thrust for x in sc.tlm])
        accs = thrusts / masses
        elorbs = [elorb(x.y[:3], x.y[3:], l_DU=voyager.EarthRe, mu=voyager.EarthGM, t0=x.t) for x in sc.tlm]
        eccs = np.array([elorb.e for elorb in elorbs])
        incs = np.array([np.rad2deg(elorb.i) for elorb in elorbs])
        smis = np.array([elorb.a for elorb in elorbs]) / 1852  # Display in nmi to match document
        c3s = -(voyager.EarthGM / (1000 ** 3)) / (np.array([elorb.a for elorb in elorbs]) / 1000)  # work directly in km
        plt.figure(1)
        plt.clf()
        plt.subplot(2,3,1)
        plt.plot(ts, np.linalg.norm(states[:, 3:6], axis=1), label='spd')
        plt.title("spd")
        plt.subplot(2,3,2)
        plt.plot(ts, eccs, label='e')
        plt.title("e")
        plt.subplot(2,3,3)
        plt.plot(ts, incs, label='i')
        plt.title("inc")
        plt.subplot(2,3,4)
        plt.plot(ts, smis, label='a')
        plt.title("a")
        plt.ylim(-5000,5000)
        plt.ylabel('semi-major axis/nmi')
        plt.subplot(2,3,5)
        plt.plot(ts, c3s, label='c3')
        plt.title("c3")
        plt.ylabel('$C_3$/(km**2/s**2)')
        plt.subplot(2,3,6)
        plt.plot(ts, masses, label='i')
        plt.title("mass")
        plt.pause(0.1)
    return sc


def opt_interface_pm_burn(target:np.ndarray=None,verbose:bool=False, fps:int=100, vgr_id:int=1)->float:
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
    sc = sim_pm(dpitch=dpitch, dthr=dthr, dyaw=dyaw, fps=fps, verbose=verbose, yawrate=yawrate, vgr_id=vgr_id)
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
    elorb0 = elorb(r0_e, v0_e, l_DU=voyager.EarthRe, mu=voyager.EarthGM, t0=simt_track_prePM[vgr_id], deg=True)
    da = (target_a_prePM[vgr_id] - elorb0.a) / voyager.EarthRe  # Use Earth radius to weight da
    de = (target_e_prePM[vgr_id] - elorb0.e)
    di = np.deg2rad(target_i_prePM[vgr_id] - elorb0.i)
    dlan = np.deg2rad(target_lan_prePM[vgr_id] - elorb0.an)
    cost = (da ** 2 + de ** 2 + di ** 2 + 0 * dlan ** 2) * 1e6
    print(f"{dyaw=} deg, {yawrate=} deg/s, {dpitch=} deg, {dthr=}")
    print(f"   {da=} Earth radii")
    print(f"      ({da * voyager.EarthRe} m)")
    print(f"   {de=}")
    print(f"   {di=} rad")
    print(f"      ({np.rad2deg(di)} deg)")
    print(f"   {dlan=} rad")
    print(f"      ({np.rad2deg(dlan)} deg)")
    print(f"Cost: {cost}")
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
    elorb0 = elorb(r0_e, v0_e, l_DU=voyager.EarthRe, mu=voyager.EarthGM, t0=simt_track_prePM[vgr_id], deg=True)
    da = (target_a_park[vgr_id] - elorb0.a) / voyager.EarthRe  # Use Earth radius to weight da
    de = (target_e_park[vgr_id] - elorb0.e)
    di = np.deg2rad(target_i_park[vgr_id] - elorb0.i)
    cost = (da ** 2 + 50*de ** 2 + di ** 2)*1e3
    print(f"{dyaw=} deg, {dpitch=} deg, {dthr=}")
    print(f"   {da=} Earth radii")
    print(f"      ({da * voyager.EarthRe} m)")
    print(f"   {de=}")
    print(f"   {di=} rad")
    print(f"      ({np.rad2deg(di)} deg)")
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
    #initial_guess=np.zeros(4)
    #initial_guess = [-1.4501264557665727,
    #                  0.0076685577917476625,
    #                 -0.001724033782731768,
    #                 0]  # Best known three-parameter fit for four targets
    #initial_guess=[-1.4501304910881003,
    #               -0.13110923799828175,
    #               -0.0017211606889325382,
    #                0.0] # Best known three-parameter fit for three targets
    initial_guess=[float.fromhex("-0x1.72dc4a7dd46d7p+0"),
                   float.fromhex("-0x1.0c7d88ec0dcfdp-3"),
                   float.fromhex("-0x1.67f69c84a6779p-9"),
                   0.0] # Best known three-parameter fit at 100Hz
    bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(0,0)]  # Freeze yaw rate at 0
    #initial_guess = [-1.438, 13.801, +0.00599, -0.549]  # From previous four-parameter form
    #bounds = [(-30, 30), (-30, 30), (-0.1, 0.1),(-1,1)]  # Bounds on
    if optimize:
        result = minimize(lambda x:opt_interface_pm_burn(x,vgr_id=vgr_id),
                          initial_guess,
                          method='L-BFGS-B',
                          options={'ftol':1e-12,'gtol':1e-12,'disp':True},
                          bounds=bounds)
        print("Achieved cost:", result.fun)
        final_guess=result.x
    else:
        final_guess=initial_guess
    print("Optimal parameters:", final_guess)
    print("Optimal run: ")
    sc=sim_pm(**{k:v for k,v in zip(('dpitch','dyaw','dthr','yawrate'),final_guess)},verbose=True,vgr_id=vgr_id)
    if export:
        states=sorted([np.hstack((np.array(x.t),x.y)) for x in sc.tlm], key=lambda x:x[0])
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
    #initial_guess=np.zeros(3)
    #                dpitch               dyaw                   dthr                pitchrate
    initial_guess=[-4.4164552842503e+00,-4.3590393660760e-02,-9.6115297747416e-03,4.8457186584881e-03]
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
        states=sorted([np.hstack((np.array(x.t),x.y)) for x in sc.tlm], key=lambda x:x[0])
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
    vgr_id=2
    #pm=target_pm(export=True, optimize=True,vgr_id=vgr_id)
    pm=voyager.parse_pm(voyager.pm_solutions_str[vgr_id])
    target_centaur2(simt1=pm.simt0,y1=pm.y0,fps1=10, optimize=True, export=True,vgr_id=vgr_id)
    for vgr_id in (1,2):
        export(vgr_id)



if __name__=="__main__":
    main()