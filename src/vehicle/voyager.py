"""
Describe purpose of this script here

Created: 1/19/25
"""
import re
from collections import namedtuple
from math import isclose

import numpy as np
from spiceypy import furnsh, gdpool, str2et

from rocket_sim.vehicle import Vehicle, Stage, Engine, kg_per_lbm, g0, N_per_lbf


# Take first reliable position as 60s after first Horizons data point.
# This line is that data, copied from data/v1_horizons_vectors_1s.txt line 146
horizons_data={1:[2443392.083615544,  # JDTDB
                  "A.D. 1977-Sep-05 14:00:24.3830",  # Gregorian TDB
                  48.182579,  # DeltaT -- TDB-UT
                  7.827005697472316E+03,  # X position, km, in Earth-centered frame parallel to ICRF (Spice J2000)
                 -4.769757853525854E+02,  # Y position, km
                 -5.362302110171012E+02,  # Z position, km
                  7.660453026119973E+00,  # X velocity, km/s
                  1.080867154837185E+01,  # Y velocity, km/s
                  5.581379825305540E+00,  # Z velocity, km/s
                  2.621760038208511E-02,  # Light time to geocenter, s
                  7.859838861407035E+03,  # Range from center, km
                  6.591742058881379E+00], # Range rate from center, km/s
               # Taken from data/v2_horizons_vectors.txt line 92. This one is on a 60s cadence so it's just the second record.
               2:[2443376.148289155,      #JDTDB
                  "A.D. 1977-Aug-20 15:33:32.1830", # Gregorian TDB
                  48.182846,              #TDB-UT
                  7.513438354530380E+03,  # rx km
                 -1.311047210468180E+03,  # ry km
                  2.089389319809422E+03,  # rz km
                  5.920525058736983E+00,  # vx km/s
                  8.878861450139366E+00,  # vy km/s
                  9.461766135385785E+00,  # vz km/s
                  2.637818209251871E-02,  # LT s
                  7.907980047087768E+03,  # R  km
                  6.653052526114903E+00]} # RR km/s
# Following is from TC-6 flight data report table 4-3, Guidance Telemetry, so what Centaur
# thought it hit. Best guess is that this is in something like true of date equatorial frame.
simt_track_prePM = {1:3600.03,2:3562.04} # Time of post-MECO2 (after Centaur burn but before PM burn) target
target_a_prePM = {1: -4220.43 * 1852,2:-4417.83*1852}  # Initial coefficient is nautical miles, convert to meters
target_e_prePM = {1:1.84171,2:1.80349}
target_i_prePM = {1:28.5165,2:41.7475} # degrees
target_lan_prePM = {1:170.237,2:130.9302} # degrees
# Following is from TC-6 flight data report table 4-2, Guidance Telemetry, so what Centaur
# thought it hit. This is the parking orbit just after Centaur burn 1 cutoff.
simt_track_park={1:596.03,2:586.04}     # Time of cutoff for Voyager 1 was 594.0, so about 2s before track point
target_a_park={1:3533.81*1852,2:3553.91*1852}
target_e_park={1:0.000038,2:0.000010}
target_i_park={1:28.5201,2:37.3625}
target_c3_park={1:-60.90520,2:-60.9036} # km**2/s**2, not actually a target value, completely determined from a above.
# These are the millisecond-precision timestamps of the
# ignition of the SRBs on each launch vehicle, and represent the
# official T=0 for all of the timelines in the flight data reports.
voyager_cal0={1:"1977-09-05T12:56:00.958Z",
              2:"1977-08-20T14:29:44.256Z"}

EarthGM=None
EarthRe=None
# These hold the Spice ET of the Horizons data point for each mission
horizons_et={}
# These hold the Spice ET of the official T=0 for each mission
voyager_et0={}

# Solutions from runs of voyager1_pm.target()
pm_solutions_str={}
pm_solutions_str[1]="""Voyager 1 backpropagation through PM burn
 dpitch: -1.4486738736345e+00 (-0x1.72dc4a7dd46d7p+0)
 dthr: -2.7463022291387e-03 (-0x1.67f69c84a6779p-9)
 dyaw: -1.3109881372814e-01 (-0x1.0c7d88ec0dcfdp-3)
 yawrate: 0.0000000000000e+00 (0x0.0p+0)
 fps: 100 
 Initial state (simt, ICRF, SI): 
  simt=0 in ET: -7.0441579085945e+08  (-0x1.4fe44176e027dp+29) (1977-09-05T12:56:49.140 TDB,1977-09-05T12:56:00.957Z)
  simt:   3.8152424509525e+03  rx:    7.8270056974723e+06  ry:   -4.7697578535259e+05  rz:   -5.3623021101710e+05
        0x1.dce7c22880000p+11       0x1.ddb8f6ca362ecp+22      -0x1.d1cbf243377d8p+18      -0x1.05d4c6c0a6ef9p+19
                               vx:    7.6604530261200e+03  vy:    1.0808671548372e+04  vz:    5.5813798253055e+03
                                    0x1.dec73f9851185p+12       0x1.51c55f54c0b63p+13       0x1.5cd613c3b317dp+12
Final state (simt, ICRF, SI): 
  simt:   3.6000324509525e+03  rx:    6.1670620991322e+06  ry:   -2.5491823405092e+06  rz:   -1.6001965204496e+06
        0x1.c20109d6947aep+11       0x1.7868586582e92p+22      -0x1.372df2b95ce84p+21      -0x1.86ac4853c2edfp+20
                               vx:    8.1875867795104e+03  vy:    8.9161670584519e+03  vz:    4.5422888535830e+03
                                    0x1.ffb96372e96ecp+12       0x1.16a15622bddc9p+13       0x1.1be49f24ef458p+12
"""
pm_solutions_str[2]="""Voyager 2 backpropagation through PM burn
 dpitch: -1.2225338175589e+00 (-0x1.38f7f9ecab6aap+0)
 dthr: -4.1988923976743e-03 (-0x1.132db98705ebdp-8)
 dyaw: -5.5151257204035e-01 (-0x1.1a5fdb187e056p-1)
 yawrate: 0.0000000000000e+00 (0x0.0p+0)
 fps: 100 
 Initial state (simt, ICRF, SI): 
  simt=0 in ET: -7.0579256756118e+08  (-0x1.508c51bc7d4dap+29) (1977-08-20T14:30:32.438 TDB,1977-08-20T14:29:44.256Z)
  simt:   3.7797441831827e+03  rx:    7.5134383545304e+06  ry:   -1.3110472104682e+06  rz:    2.0893893198094e+06
        0x1.d877d05940000p+11       0x1.ca95796b0a031p+22      -0x1.4014735e13e1ep+20      0x1.fe1ad51df07c0p+20
                               vx:    5.9205250587370e+03  vy:    8.8788614501394e+03  vz:    9.4617661353858e+03
                                    0x1.720866a3fd7d2p+12       0x1.1576e43ff87dbp+13       0x1.27ae210b96d21p+13
Final state (simt, ICRF, SI): 
  simt:   3.5620441831827e+03  rx:    6.1733164958585e+06  ry:   -3.0265564102650e+06  rz:    1.7615473922352e+05
        0x1.bd4169f2d999ap+11       0x1.78ca11fbc253bp+22      -0x1.7173e34839074p+21      0x1.580d5e9ee0532p+17
                               vx:    6.6401840157381e+03  vy:    7.0938382526628e+03  vz:    8.3795263368295e+03
                                    0x1.9f02f1ba7c933p+12       0x1.bb5d697b9fc83p+12       0x1.05dc35f015698p+13
"""
centaur2_solutions_str={}
centaur2_solutions_str[1]="""Voyager 1 backpropagation through Centaur burn 2
 dpitch: -4.4164552842171e+00 (-0x1.1aa734107d5bdp+2)
 dthr: -9.6115299246784e-03 (-0x1.3af35b58734eep-7)
 dyaw: -4.3590393700061e-02 (-0x1.6517ae6b919f3p-5)
 pitchrate: 4.8457188530925e-03 (0x1.3d91abffe89a4p-8)
 fps during burn: 10 
 fps during parking orbit: 1 
 Initial state (simt, ICRF, SI): 
  simt=0 in ET: -7.0441579085945e+08  (-0x1.4fe44176e027dp+29) (1977-09-05T12:56:49.140 TDB,1977-09-05T12:56:00.957Z)
  simt:   3.6000324509525e+03  rx:    6.1670620991322e+06  ry:   -2.5491823405092e+06  rz:   -1.6001965204496e+06
        0x1.c20109d6947aep+11       0x1.7868586582e92p+22      -0x1.372df2b95ce84p+21      -0x1.86ac4853c2edfp+20
                               vx:    8.1875867795104e+03  vy:    8.9161670584519e+03  vz:    4.5422888535830e+03
                                    0x1.ffb96372e96ecp+12       0x1.16a15622bddc9p+13       0x1.1be49f24ef458p+12
State just prior to Centaur burn 2 (simt, ICRF, SI): 
  simt:   3.1898324509525e+03  rx:    3.0090531901394e+06  ry:   -5.0533493988030e+06  rz:   -2.8456944924418e+06
        0x1.8ebaa3702e148p+11       0x1.6f50e98567c9cp+21      -0x1.346e95985fcdfp+22      -0x1.5b5ff3f085542p+21
                               vx:    6.9351764858879e+03  vy:    3.2685011881941e+03  vz:    1.5259224433481e+03
                                    0x1.b172d2e2ddcecp+12       0x1.989009bbd2d90p+11       0x1.7d7b094fd32c8p+10
State just after Centaur burn 1 (simt, ICRF, SI): 
  simt:   5.9673245095253e+02  rx:   -3.2645531625213e+06  ry:    4.9345660010038e+06  rz:    2.7971187106339e+06
        0x1.2a5dc0f3eb850p+9       -0x1.8e81494cd7f6ap+21       0x1.2d2e98010722ep+22       0x1.5571f5af60ce4p+21
                               vx:   -6.7589049552103e+03  vy:   -3.5314790162879e+03  vz:   -1.6589917219292e+03
                                   -0x1.a66e7ab25088fp+12      -0x1.b96f5419f75f7p+11      -0x1.9ebf785f412c1p+10
"""
centaur2_solutions_str[2]="""
Voyager 2 backpropagation through Centaur burn 2
 dpitch: -1.3433920246681e+01 (-0x1.ade2acb692223p+3)
 dthr: -2.2020571000027e-02 (-0x1.68c8f81232fcdp-6)
 dyaw: 9.0870458935636e+00 (0x1.22c91478436bdp+3)
 pitchrate: 5.1374694020877e-02 (0x1.a4dc8ad52c6efp-5)
 fps during burn: 10 
 fps during parking orbit: 1 
 Initial state (simt, ICRF, SI): 
  simt=0 in ET: -7.0579256756118e+08  (-0x1.508c51bc7d4dap+29) (1977-08-20T14:30:32.438 TDB,1977-08-20T14:29:44.256Z)
  simt:   3.5620441831827e+03  rx:    6.1733164958585e+06  ry:   -3.0265564102650e+06  rz:    1.7615473922352e+05
        0x1.bd4169f2d999ap+11       0x1.78ca11fbc253bp+22      -0x1.7173e34839074p+21      0x1.580d5e9ee0532p+17
                               vx:    6.6401840157381e+03  vy:    7.0938382526628e+03  vz:    8.3795263368295e+03
                                    0x1.9f02f1ba7c933p+12       0x1.bb5d697b9fc83p+12       0x1.05dc35f015698p+13
State just prior to Centaur burn 2 (simt, ICRF, SI): 
  simt:   3.1385441831827e+03  rx:    3.3673651185549e+06  ry:   -5.0913568946721e+06  rz:   -2.4289355979399e+06
        0x1.885169f2d999ap+11       0x1.9b0e28f2cce77p+21      -0x1.36c0739424ecdp+22      -0x1.28803cc894b05p+21
                               vx:    6.3846958105032e+03  vy:    2.4277676702363e+03  vz:    3.7575642528021e+03
                                    0x1.8f0b220a31b45p+12       0x1.2f7890c12bda5p+11       0x1.d5b20e5be470dp+11
State just after Centaur burn 1 (simt, ICRF, SI): 
  simt:   5.8644418318272e+02  rx:   -3.9897510197326e+06  ry:    4.8158140413814e+06  rz:    2.0521515876652e+06
        0x1.2538dafe9999cp+9       -0x1.e707b8286996fp+21      0x1.25ef182a5fe16p+22      0x1.f503796713981p+20
                               vx:   -5.8589228054144e+03  vy:   -3.1246211690210e+03  vz:   -4.0582323317824e+03
                                    -0x1.6e2ec3cf9c38ep+12       -0x1.8693e09ddaccdp+11       -0x1.fb476f430feb2p+11
"""
parsed_pm=namedtuple("parsed_pm","vgr_id dpitch dthr dyaw yawrate fps et_t0 simt1 y1 simt0 y0")
def parse_pm(pm_solution_str:str):
    """

    :param pm_solution_str:
    :return:
    """
    lines=[x.strip() for x in pm_solution_str.split("\n")]
    if match:=re.match(r"Voyager (?P<vgr_id>\d+) backpropagation through PM burn",lines[0]):
        vgr_id=int(match.group("vgr_id"))
    else:
        raise ValueError("Couldn't parse vgr_id")
    def parse_steer_param(steer_param:str,line:str):
        if match:=re.match(fr"{steer_param}:\s+(?P<decval>[-+]?[0-9].[0-9]+e[-+][0-9]+)\s+\((?P<hexval>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)\)",line):
            decval=float(match.group("decval"))
            hexval=float.fromhex(match.group("hexval"))
            if not isclose(decval,hexval):
                raise ValueError(f"{steer_param} dec and hex don't match")
        else:
            raise ValueError(f"Couldn't parse {steer_param}")
        return hexval
    dpitch=parse_steer_param("dpitch",lines[1])
    dthr=parse_steer_param("dthr",lines[2])
    dyaw=parse_steer_param("dyaw",lines[3])
    yawrate=parse_steer_param("yawrate",lines[4])
    if match:=re.match(r"fps:\s+(?P<fps>\d+)",lines[5]):
        fps=match.group("fps")
    else:
        raise ValueError("Couldn't parse fps")
    if not lines[6]=="Initial state (simt, ICRF, SI):":
        raise ValueError("Unexpected initial state header")
    if match:=re.match("simt=0 in ET:\s+(?P<decval>[-+]?[0-9].[0-9]+e[-+][0-9]+)\s+"
                       "\((?P<hexval>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)\)\s+"
                       "\((?P<isotdb>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)\s+TDB,\s*"
                         "(?P<isoutc>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)Z\)",lines[7]):
        simt0_dec=float(match.group("decval"))
        et_t0=float.fromhex(match.group("hexval"))
        isotdb=match.group("isotdb")
        isotdb=isotdb[:10]+' '+isotdb[11:]+" TDB"
        isoutc=match.group("isoutc")+"Z"
        if not isclose(simt0_dec, et_t0):
            raise ValueError("simt0 dec and hex don't match")
        if not isclose(et_t0,str2et(isotdb),abs_tol=1e-3):
            raise ValueError(f"simt0 and Gregorian TDB don't match: {et_t0=}, {isotdb=} {str2et(isotdb)=}")
        if not isclose(et_t0, str2et(isoutc), abs_tol=1e-3):
            raise ValueError(f"simt0 and Gregorian UTC don't match: {et_t0=}, {isoutc=} {str2et(isoutc)=}")
    else:
        raise ValueError("Couldn't parse simt=0")
    def parse_state(lines:list[str]):
        if match:=re.match(r"simt:\s+(?P<simt_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)"
                           r"\s+rx:\s+(?P<rx_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)"
                           r"\s+ry:\s+(?P<ry_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)"
                           r"\s+rz:\s+(?P<rz_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)",lines[0]):
            simt_dec=float(match.group("simt_dec"))
            rx_dec = float(match.group("rx_dec"))
            ry_dec = float(match.group("ry_dec"))
            rz_dec = float(match.group("rz_dec"))
        else:
            raise ValueError("Can't parse first line of state vector")
        if match:=re.match(r"(?P<simt>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)"
                           r"\s+(?P<rx>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)"
                           r"\s+(?P<ry>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)"
                           r"\s+(?P<rz>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)",lines[1]):
            simt=float.fromhex(match.group("simt"))
            rx = float.fromhex(match.group("rx"))
            ry = float.fromhex(match.group("ry"))
            rz = float.fromhex(match.group("rz"))
            if not isclose(simt_dec,simt):
                raise ValueError("Discrepancy in simt")
            if not isclose(rx_dec, rx):
                raise ValueError("Discrepancy in rx")
            if not isclose(ry_dec, ry):
                raise ValueError("Discrepancy in ry")
            if not isclose(rz_dec, rz):
                raise ValueError("Discrepancy in rz")
        else:
            raise ValueError("Can't parse second line of state vector")
        if match:=re.match(r"vx:\s+(?P<vx_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)"
                           r"\s+vy:\s+(?P<vy_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)"
                           r"\s+vz:\s+(?P<vz_dec>[-+]?[0-9].[0-9]+e[-+][0-9]+)",lines[2]):
            vx_dec = float(match.group("vx_dec"))
            vy_dec = float(match.group("vy_dec"))
            vz_dec = float(match.group("vz_dec"))
        else:
            raise ValueError("Can't parse third line of state vector")
        if match:=re.match(r"(?P<vx>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)"
                           r"\s+(?P<vy>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)"
                           r"\s+(?P<vz>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)",lines[3]):
            vx = float.fromhex(match.group("vx"))
            vy = float.fromhex(match.group("vy"))
            vz = float.fromhex(match.group("vz"))
            if not isclose(vx_dec, vx):
                raise ValueError("Discrepancy in vx")
            if not isclose(vy_dec, vy):
                raise ValueError("Discrepancy in vy")
            if not isclose(vz_dec, vz):
                raise ValueError("Discrepancy in vz")
        else:
            raise ValueError("Can't parse second line of state vector")
        return simt,np.array((rx,ry,rz,vx,vy,vz))
    simt1,y1=parse_state(lines[8:12])
    if not lines[12]=="Final state (simt, ICRF, SI):":
        raise ValueError("Unexpected final state header")
    simt0,y0=parse_state(lines[13:])
    result=parsed_pm(
        vgr_id=vgr_id,
        dpitch=dpitch,dthr=dthr,dyaw=dyaw,yawrate=yawrate,
        fps=fps,
        et_t0=et_t0,
        simt1=simt1,y1=y1,
        simt0=simt0,y0=y0
    )
    print(result)
    return result


def best_pm_initial_guess(*,vgr_id:int):
    # Parameters are:
    #   0 - dpitch, pitch angle between pure prograde and target in degrees
    #   1 - dyaw, yaw angle between pure in-plane and target in degrees
    #   2 - dthr, fractional difference in engine efficiency. 0 is 100%,
    #       +0.1 is 110%, -0.1 is 90%, etc. This value changes thrust10 and ve
    #       simultaneously so as to not affect mdot, so the propellant
    #       will drain exactly as fast as before.
    #   3 - yawrate - change in yaw vs time in deg/s.
    pm_solution=parse_pm(pm_solutions_str[vgr_id])
    if not vgr_id==pm_solution.vgr_id:
        raise AssertionError("Solution does not match request")
    return [pm_solution.dpitch,pm_solution.dyaw,pm_solution.dthr,pm_solution.yawrate]
    #initial_guess=[float.fromhex("-0x1.72dc4a7dd46d7p+0"),
    #               float.fromhex("-0x1.0c7d88ec0dcfdp-3"),
    #               float.fromhex("-0x1.67f69c84a6779p-9"),
    #               0.0] # Best known Voyager 1 three-parameter fit at 100Hz
    initial_guess=[float.fromhex("-0x1.38f7f9ecab77ap+0"),
                   float.fromhex("-0x1.1a5fdb187dc17p-1"),
                   float.fromhex("-0x1.132db98957525p-8"),
                   0.0] # Best known Voyager 2 three-parameter fit at 100Hz


def init_spice():
    # Furnish the kernels
    furnsh("data/naif0012.tls")
    furnsh("data/pck00011.tpc")  # Sizes and orientations of all planets including Earth
    furnsh("data/gm_de440.tpc")  # Masses of all planets and many satellites, including Earth
    furnsh("products/gravity_EGM2008_J2.tpc") # Cover up Earth mass from gm_de440.tpc and size from pck00011.tpc
    furnsh("data/de440.bsp")     # Solar system ephemeris
    global EarthGM,EarthRe
    EarthGM = gdpool("BODY399_GM", 0, 1)[0] * 1000 ** 3
    EarthRe = gdpool("BODY399_RADII", 0, 3)[0] * 1000
    global horizons_et, voyager_et0
    for k in (1,2):
        # Time of Horizons vector, considered to be t1. Calculate
        # here because kernels aren't available up there
        horizons_et[k]=str2et(f"{horizons_data[k][1]} TDB")
        # Launch timeline T0
        voyager_et0[k]=str2et(voyager_cal0[k])


def main():
    init_spice()
    parse_pm(pm_solutions_str[1])


if __name__=="__main__":
    main()
