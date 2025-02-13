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
import re
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import TextIO, Callable, Any

import numpy as np
from bmw import elorb
from kwanmath.vector import vlength
from kwanspice.mkspk import mkspk
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from spiceypy import str2et, pxform, timout, furnsh, gdpool

from guidance.orbit import dprograde_guide, seq_guide, prograde_guide, yaw_rate_guide
from rocket_sim.gravity import SpiceTwoBody, SpiceJ2, SpiceThirdBody
from rocket_sim.planet import Earth, Planet
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Vehicle, g0
from vehicle.titan_3e_centaur import Titan3E


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
# Following is from TC-[67] flight data report table 4-3, Guidance Telemetry, so what Centaur
# thought it hit. Best guess is that this is in something like true of date equatorial frame.
simt_track_prePM = {1:3600.03,2:3562.04} # Time of post-MECO2 (after Centaur burn but before PM burn) target
target_a_prePM = {1: -4220.43 * 1852,2:-4417.83*1852}  # Initial coefficient is nautical miles, convert to meters
target_e_prePM = {1:1.84171,2:1.80349}
target_i_prePM = {1:28.5165,2:41.7475} # degrees
target_lan_prePM = {1:170.237,2:130.9302} # degrees
# Following is from TC-[67] flight data report table 4-2, Guidance Telemetry, so what Centaur
# thought it hit. This is the parking orbit just after Centaur burn 1 cutoff.
simt_track_park={1:596.03,2:586.04}     # Time of cutoff for Voyager 1 was 594.0, so about 2s before track point
target_a_park={1:3533.81*1852,2:3553.91*1852}
target_e_park={1:0.000038,2:0.000010}
target_i_park={1:28.5201,2:37.3625}
target_c3_park={1:-60.90520,2:-60.9036} # km**2/s**2, not actually a target value, completely determined from a above.
# These are the millisecond-precision timestamps of the
# ignition of the SRBs on each launch vehicle, and represent the
# official T=0 for all of the timelines in the flight data reports.
# Note that this is in UTC.
voyager_cal0={1:"1977-09-05T12:56:00.958Z",
              2:"1977-08-20T14:29:44.256Z"}
earth:Planet=None
# These hold the Spice ET of the Horizons data point for each mission
horizons_et={}
# These hold the Spice ET of the official T=0 for each mission
voyager_et0={}
# Solutions from runs of voyager1_pm.target()

pm_solutions_str={1:"""Voyager 1 backpropagation through PM burn
 dpitch: -1.4486738736345e+00 (-0x1.72dc4a7dd46d7p+0)
 dthr: -2.7463022291387e-03 (-0x1.67f69c84a6779p-9)
 dyaw: -1.3109881372814e-01 (-0x1.0c7d88ec0dcfdp-3)
 yawrate: 0.0000000000000e+00 (0x0.0p+0)
 fps: 100 
  simt=0 in ET: -7.0441579085945e+08  (-0x1.4fe44176e027dp+29) (1977-09-05T12:56:49.140 TDB,1977-09-05T12:56:00.957Z)
 Initial state (simt, ICRF, SI): 
  simt:   3.8152424509525e+03  rx:    7.8270056974723e+06  ry:   -4.7697578535259e+05  rz:   -5.3623021101710e+05
        0x1.dce7c22880000p+11       0x1.ddb8f6ca362ecp+22      -0x1.d1cbf243377d8p+18      -0x1.05d4c6c0a6ef9p+19
                               vx:    7.6604530261200e+03  vy:    1.0808671548372e+04  vz:    5.5813798253055e+03
                                    0x1.dec73f9851185p+12       0x1.51c55f54c0b63p+13       0x1.5cd613c3b317dp+12
Final state (simt, ICRF, SI): 
  simt:   3.6000324509525e+03  rx:    6.1670620991322e+06  ry:   -2.5491823405092e+06  rz:   -1.6001965204496e+06
        0x1.c20109d6947aep+11       0x1.7868586582e92p+22      -0x1.372df2b95ce84p+21      -0x1.86ac4853c2edfp+20
                               vx:    8.1875867795104e+03  vy:    8.9161670584519e+03  vz:    4.5422888535830e+03
                                    0x1.ffb96372e96ecp+12       0x1.16a15622bddc9p+13       0x1.1be49f24ef458p+12
""",2:"""Voyager 2 backpropagation through PM burn
 dpitch: -1.2225338175589e+00 (-0x1.38f7f9ecab6aap+0)
 dthr: -4.1988923976743e-03 (-0x1.132db98705ebdp-8)
 dyaw: -5.5151257204035e-01 (-0x1.1a5fdb187e056p-1)
 yawrate: 0.0000000000000e+00 (0x0.0p+0)
 fps: 100 
  simt=0 in ET: -7.0579256756118e+08  (-0x1.508c51bc7d4dap+29) (1977-08-20T14:30:32.438 TDB,1977-08-20T14:29:44.256Z)
 Initial state (simt, ICRF, SI): 
  simt:   3.7797441831827e+03  rx:    7.5134383545304e+06  ry:   -1.3110472104682e+06  rz:    2.0893893198094e+06
        0x1.d877d05940000p+11       0x1.ca95796b0a031p+22      -0x1.4014735e13e1ep+20      0x1.fe1ad51df07c0p+20
                               vx:    5.9205250587370e+03  vy:    8.8788614501394e+03  vz:    9.4617661353858e+03
                                    0x1.720866a3fd7d2p+12       0x1.1576e43ff87dbp+13       0x1.27ae210b96d21p+13
Final state (simt, ICRF, SI): 
  simt:   3.5620441831827e+03  rx:    6.1733164958585e+06  ry:   -3.0265564102650e+06  rz:    1.7615473922352e+05
        0x1.bd4169f2d999ap+11       0x1.78ca11fbc253bp+22      -0x1.7173e34839074p+21      0x1.580d5e9ee0532p+17
                               vx:    6.6401840157381e+03  vy:    7.0938382526628e+03  vz:    8.3795263368295e+03
                                    0x1.9f02f1ba7c933p+12       0x1.bb5d697b9fc83p+12       0x1.05dc35f015698p+13
"""}
centaur2_solutions_str={1:"""Voyager 1 backpropagation through Centaur 2 burn
dpitch: -4.4164552842503e+00 (-0x1.1aa73410867dbp+2)
dyaw: -4.3590393660760e-02 (-0x1.6517ae662ad7fp-5)
dthr: -9.6115297747416e-03 (-0x1.3af35b0605959p-7)
pitchrate: 4.8457186584881e-03 (0x1.3d91ab29f0544p-8)
fps: 100 
 simt=0 in ET: -7.0441579085945e+08  (-0x1.4fe44176e027dp+29) (1977-09-05T12:56:49.140 TDB,1977-09-05T12:56:00.957Z)
Initial state (simt, ICRF, SI): 
simt:   3.6000224509525e+03  rx:    6.1669802228837e+06  ry:   -2.5492715020224e+06  rz:   -1.6002419432390e+06
           0x1.c200b7eb0f5c2p+11       0x1.786710e43b9e3p+22      -0x1.3730bc042457ep+21      -0x1.86af1f1781d1bp+20
                              vx:    8.1876629249787e+03  vy:    8.9161355826285e+03  vz:    4.5422690399699e+03
                                   0x1.ffba9b5738f38p+12       0x1.16a115ac58590p+13       0x1.1be44dfcdafe5p+12
Final state (simt, ICRF, SI): 
simt:   3.1897924509525e+03  rx:    3.0091300275335e+06  ry:   -5.0530993104807e+06  rz:   -2.8455620430381e+06
           0x1.8eb95bc219998p+11       0x1.6f53503863771p+21      -0x1.346aad3deea85p+22      -0x1.5b5bd05824607p+21
                              vx:    6.9348978341117e+03  vy:    3.2674891753943e+03  vz:    1.5253834628964e+03
                                   0x1.b16e5d874d2dcp+12       0x1.986fa7532812dp+11       0x1.7d588aa7f5cc3p+10
State just after Centaur burn 1 (simt, ICRF, SI): 
simt:   5.9599245095253e+02  rx:   -3.2482120645243e+06  ry:    4.9397376863150e+06  rz:    2.7993087051377e+06
           0x1.29ff08a1fffffp+9       -0x1.8c82a08425574p+21      0x1.2d7f66bec95b0p+22      0x1.55b665a41f3e8p+21
                              vx:   -6.7709701234338e+03  vy:   -3.5183080653125e+03  vz:   -1.6513636562164e+03
                                   -0x1.a72f85a026560p+12       -0x1.b7c9dbabc94bbp+11       -0x1.9cd74624b914fp+10
Iterations: 0
dpitch=-4.416455 deg, dyaw=-0.043590 deg, dthr=-0.009612, pitchrate=0.004846 deg/s
ahist=3533.81 nmi,acalc=3532.8733026493824 nmi,da=0.9366973506175782 nmi,     4.7 sigma
ehist=3.8e-05 ,ecalc=0.0003007568171320429 ,de=-0.0002627568171320429 ,    -3.8 sigma
ihist=28.5201 deg,icalc=28.52002426976721 deg,di=7.573023279050517e-05 deg,     0.0 sigma
Cost: 3.603e+01
""",2:"""Voyager 2 backpropagation through Centaur burn 2
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
"""}


@dataclass
class ParsedPM:
    """
    Contains data for a navigation solution for the Propulsion Module burn. This solution
    is the pitch, yaw, and efficiency difference from nominal that goes from the semi-docuemnted
    pre-PM trajectory solution in the TC-6 and -7 flight data reports to the perfectly
    specified (but of uncertain provenance) Horizons data solution soon after PM burnout.

    Targeting actually goes backwards -- we figure the location of the spacecraft
    at the Horizons solution and step the integrator with a negative time step
    through the burn to the pre-burn state. This state is completely determined
    by the Horizons solution and the steering parameters, and its orbital elements
    can be compared with those of the flight data reports partially-documented
    elements.
    """
    vgr_id:int       # Voyager number
    dpitch:float     # Pitch difference from nominal (prograde) in local VNC frame
    dthr:float       # Differential engine efficiency
    dyaw:float       # Yaw difference from nominal (prograde)
    yawrate:float    # Yaw rate -- yaw at any simt t is dyaw+yawrate*(t-simt0)
    fps:int          # Simulation frame rate. Different fps generate slightly different optimum solutions
    et_t0:float      # et of simt=0 for this solution
    simt1:float      # Time of state vector after the burn. From Horizons, used as integrator "initial condition"
    y1:np.ndarray    # Position and velocity in ICRF Earth-centered frame at simt1
    simt0:float      # Time of state vector before the burn, from the flight data report post-MECO2 trajectory
    y0:np.ndarray    # Achieved state at simt0, given state at simt1 and steering parameters


def parse_pm(pm_solution_str:str)->ParsedPM:
    """

    :param pm_solution_str:
    :return:
    """
    lines=[x.strip() for x in pm_solution_str.split("\n")]
    if match:=re.match(r"Voyager (?P<vgr_id>\d+) backpropagation through PM burn",lines[0]):
        vgr_id=int(match.group("vgr_id"))
    else:
        raise ValueError("Couldn't parse vgr_id")
    def parse_steer_params(steer_params:list[str],lines:list[str]):
        result=[None]*len(steer_params)
        for line in lines:
            if match:=re.match(fr"(?P<tag>[a-zA-Z_][a-zA-Z0-9_]*):\s+(?P<decval>[-+]?[0-9].[0-9]+e[-+][0-9]+)\s+\((?P<hexval>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)\)",line):
                if match.group("tag") not in steer_params:
                    raise ValueError(f"Unrecognized steering parameter {match.group('tag')}")
                decval=float(match.group("decval"))
                hexval=float.fromhex(match.group("hexval"))
                if not isclose(decval,hexval):
                    raise ValueError(f"{match.group('tag')} dec and hex don't match")
                result[steer_params.index(match.group("tag"))]=hexval
            else:
                raise ValueError(f"Couldn't parse line `{line}`")
        return result
    dpitch,dyaw,dthr,yawrate=parse_steer_params(["dpitch","dyaw","dthr","yawrate"],lines[1:5])
    if match:=re.match(r"fps:\s+(?P<fps>\d+)",lines[5]):
        fps=match.group("fps")
    else:
        raise ValueError("Couldn't parse fps")
    if match:=re.match("simt=0 in ET:\s+(?P<decval>[-+]?[0-9].[0-9]+e[-+][0-9]+)\s+"
                       "\((?P<hexval>[-+]?0x[01].[0-9a-fA-F]+p[-+][0-9]+)\)\s+"
                       "\((?P<isotdb>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)\s+TDB,\s*"
                         "(?P<isoutc>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]+)Z\)",lines[6]):
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
    if not lines[7]=="Initial state (simt, ICRF, SI):":
        raise ValueError("Unexpected initial state header")
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
    result=ParsedPM(
        vgr_id=vgr_id,
        dpitch=dpitch,dthr=dthr,dyaw=dyaw,yawrate=yawrate,
        fps=fps,
        et_t0=et_t0,
        simt1=simt1,y1=y1,
        simt0=simt0,y0=y0
    )
    print(result)
    return result


def best_pm_solution(*,vgr_id:int)->ParsedPM:
    # Parameters are:
    #   0 - dpitch, pitch angle between pure prograde and target in degrees
    #   1 - dyaw, yaw angle between pure in-plane and target in degrees
    #   2 - dthr, fractional difference in engine efficiency. 0 is 100%,
    #       +0.1 is 110%, -0.1 is 90%, etc. This value changes thrust10 and ve
    #       simultaneously so as to not affect mdot, so the propellant
    #       will drain exactly as fast as before.
    #   3 - yawrate - change in yaw vs time in deg/s.
    # Check for existence of optimal run in data/ or products/. The one in data/ has been
    # selected by a human to be the best, the one in products/ is the latest run.
    ps=[y for y in [x/f"vgr{vgr_id}_pm_optimal_run.txt" for x in (Path("data"),Path("products"))] if y.exists()]
    if len(ps)>0:
        print(f"Using {ps[0]} as optimal run")
        with open(ps[0],"rt") as inf: soln_str=inf.read()
    else:
        soln_str=pm_solutions_str[vgr_id]
    pm_solution=parse_pm(soln_str)
    if not vgr_id==pm_solution.vgr_id:
        raise AssertionError("Solution does not match request")
    return pm_solution


def best_centaur2_initial_guess(*,vgr_id:int)->list[float,float,float,float]:
    initial_guess={1:[-4.4164552842503e+00,-4.3590393660760e-02,-9.6115297747416e-03,4.8457186584881e-03], #Best Voyager 1 result
                   2:[-1.3433920246681e+01, 9.0870458935636e+00,-2.2020571000027e-02,5.1374694020877e-02]} #Best Voyager 2 result
    return initial_guess[vgr_id]


def best_pm_initial_guess(*,vgr_id:int)->list[float,float,float,float]:
    pm_solution=best_pm_solution(vgr_id=vgr_id)
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
    global earth
    earth=Earth()
    global horizons_et, voyager_et0
    for k in (1,2):
        # Time of Horizons vector, considered to be t1. Calculate
        # here because kernels aren't available up there
        horizons_et[k]=str2et(f"{horizons_data[k][1]} TDB")
        # Launch timeline T0
        voyager_et0[k]=str2et(voyager_cal0[k])


def report_state(*,t:float,y:np.ndarray,name:str)->str:
    """
    Generate a string representation of a state that can be turned
    back into a state with bit-for-bit accuracy

    :param t: simt at which state is valid
    :param y: state vector in the form of a (6,) numpy array. y[:3] is position in ICRF in m,
              y[3:] is velocity in ICRF in m/s
    :return: String representation of state
    """
    return f"""{name} (simt, ICRF, SI): 
simt:  {t: .13e}  rx:   {y[0]: .13e}  ry:   {y[1]: .13e}  rz:   {y[2]: .13e}
           {t.hex()}       {y[0].hex()}      {y[1].hex()}      {y[2].hex()}
                              vx:   {y[3]: .13e}  vy:   {y[4]: .13e}  vz:   {y[5]: .13e}
                                   {y[3].hex()}       {y[4].hex()}       {y[5].hex()}"""


class BackpropagationTargeter:
    """
    Targeter that given a final state that is locked down solid as a boundary constraint,
    and an incomplete set of orbital elements to hit as the other constraint, finds
    a set of guidance steering parameters that when flown backwards through the maneuver
    hits the incomplete elements as closely as possible.

    Note -- the given final state is always the final state, identified with that word
            'final' and subscript 1 such as t1, y1, etc. The achieved initial state
            is likewise always identified with the word 'initial' and
    """
    def __init__(self,*,vgr_id:int,report_name:str,t1:float,y1:np.ndarray,fps:int,out_path:Path=None):
        if out_path is None:
            out_path=Path('products')
        self.out_path=out_path
        self.vgr_id=vgr_id
        self.et0=voyager_et0[self.vgr_id]
        self.report_name=report_name
        self.t1=t1
        self.y1=y1
        self.fps=fps
        # Plot lines. Keys are names that will be inserted into self.__dict__
        #             Value is list (so mutable) of:
        #               * Callable that calculates the plot line,
        #               * Unit string if it is to be plotted or None if not
        self.plotlines:dict[str,list[Callable,str|None]]={
          "t":[lambda bpt: np.array([x.t for x in bpt.sc.tlm_points]),None],
          "state":[lambda bpt: np.array([x.y0 for x in bpt.sc.tlm_points]).T,None],
          "mass":[lambda bpt: np.array([x.mass for x in bpt.sc.tlm_points]),'kg'],
          "elorb":[lambda bpt: [earth.elorb(rv=x.y0[:3], vv=x.y0[3:], tref=x.t) for x in bpt.sc.tlm_points],None]}
        self.bounds = [(-30, 30), (-30, 30), (-0.1, 0.1), (0, 0)]  # Freeze yaw rate at 0
        self.iters=0
    def guide(self):
        """
        Return a guidance program based on the given parameters passed in from optimize
        :return:
        """
        raise NotImplementedError
    def print_params(self):
        raise NotImplementedError
    def tweak_sc(self):
        raise NotImplementedError
    def cost_components(self,*, ouf: TextIO | None)->dict[str,tuple[float,float,float]]:
        raise NotImplementedError
    def plot_lines(self):
        for name,(f,units) in self.plotlines.items():
            self.__dict__[name]=f(self)
        plt.figure(f"Voyager {self.vgr_id} {self.report_name}")
        actual_plotlines=[(k,v[1]) for k,v in self.plotlines.items() if v[1] is not None]
        for i_plotline,(name,units) in enumerate(actual_plotlines):
            plt.subplot(2,3,i_plotline+1)
            plt.cla()
            if units=='':
                plt.ylabel(f'{name}')
            elif "/" in units:
                plt.ylabel(f'{name}/({units})')
            else:
                plt.ylabel(f'{name}/{units}')
            plt.xlabel('simt/s')
            plt.plot(self.t,self.__dict__[name])
        plt.pause(0.1)
    def sim(self,ouf:TextIO=None):
        self.sc:Vehicle = Titan3E(tc_id=5+self.vgr_id)
        initstate = f"""Voyager {self.vgr_id} {self.report_name}
{self.print_params()}
fps: {self.fps} 
 simt=0 in ET: {self.et0:.13e}  ({self.et0.hex()}) ({timout(self.et0, 'YYYY-MM-DDTHR:MN:SC.### ::TDB')} TDB,{timout(self.et0, 'YYYY-MM-DDTHR:MN:SC.### ::UTC')}Z)
{report_state(t=self.t1,y=self.y1,name='Initial state')}"""
        print(initstate)
        earth_twobody = SpiceTwoBody(spiceid=399)
        earth_j2 = SpiceJ2(spiceid=399)
        moon = SpiceThirdBody(spice_id_center=399, spice_id_body=301, et0=self.et0)
        sun = SpiceThirdBody(spice_id_center=399, spice_id_body=10, et0=self.et0)
        self.universe = Universe(vehicles=[self.sc], accs=[earth_twobody, earth_j2, moon, sun],
                       t0=self.t1, y0s=[self.y1], fps=self.fps)
        self.tweak_sc()
        self.sc.guide=self.guide()
        self.universe.runto(t1=self.t0)
        finalstate = report_state(t=self.sc.tlm_points[-1].t+self.sc.tlm_points[-1].dt,y=self.sc.tlm_points[-1].y1,name='Final state')
        print(finalstate)
        if ouf is not None:
            print(initstate, file=ouf)
            print(finalstate, file=ouf)
        self.plot_lines()
    def cost(self,ouf:TextIO|None)->float:
        cost = 0
        comps = self.cost_components(ouf=ouf)
        coststr=f"Iterations: {self.iters}\n"
        coststr += ", ".join([f"{param_name}={param_val:.6f}{' ' + param_units if param_units is not None else ''}"
                             for param_name, param_val, param_units in zip(self.param_names, self.params, self.param_units)])
        for name, (achieved, target, sigma, pane, units) in comps.items():
            cost += ((target - achieved) / sigma) ** 2
            plt.subplot(2, 3, pane)
            trange = [self.sc.tlm_points[0].t, self.sc.tlm_points[-1].t + self.sc.tlm_points[-1].dt]
            plt.plot(trange, target * np.array([1, 1]), 'k--', label=f"{name} target")
            plt.plot(trange, achieved * np.array([1, 1]), 'r--', label=f"{name} achieved")
            plt.legend()
            coststr += (f"\n{name}hist={target}{' ' + units if units is not None else ''},"
                        f"{name}calc={achieved}{' ' + units if units is not None else ''},"
                        f"d{name}={target - achieved}{' ' + units if units is not None else ''},"
                        f"{(target - achieved) / sigma:8.1f} sigma")
        plt.subplot(2, 3, 6)
        plt.cla()
        coststr += f"\nCost: {cost:.3e}"
        print(coststr)
        if ouf is not None:
            print(coststr, file=ouf)
        plt.text(0, 0.5, coststr, horizontalalignment='left', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.axis('off')
        plt.pause(0.1)
        return cost
    def __call__(self,params,ouf:TextIO|None=None)->float:
        """
        This is the interface to scipy.optimize.minimize. It takes a set
        of parameters, calculates the cost, and returns it.
        :return:
        """
        self.iters+=1
        self.accept_params(params)
        self.sim()
        cost=self.cost(ouf)
        return cost
    def accept_params(self, params):
        self.params=params
    def target(self,*,optimize:bool=False):
        initial_guess=self.initial_guess()
        for guess,(bound0,bound1) in zip(initial_guess,self.bounds):
            if not (bound0<=guess<=bound1):
                raise ValueError("Bounds do not bracket guess")
        if optimize:
            result = minimize(self,
                              initial_guess,
                              method='L-BFGS-B',
                              options={'ftol':1e-6,'disp':True},
                              bounds=self.bounds)
            print("Achieved cost:", result.fun)
            print(result)
            final_guess=result.x
        else:
            result=None
            final_guess=initial_guess
        print("Optimal parameters:", final_guess)
        print("Optimal run: ")
        self.accept_params(final_guess)
        with open(self.out_path/f'vgr{self.vgr_id}_{self.save}_optimal_run.txt',"wt") as ouf:
            sc=self.sim(ouf=ouf)
            self.cost(ouf=ouf)
            if result is not None:
                print("Optimizer results:",result,file=ouf)
        return sc
    def export(self):
        states=sorted([np.hstack((np.array(x.t)+voyager_et0[self.vgr_id],x.y0)) for x in self.sc.tlm_points], key=lambda x:x[0])
        decimated_states=[]
        i=0
        di=1
        mkspk(oufn=self.out_path/f'vgr{self.vgr_id}_{self.save}.bsp',
              fmt=['f', '.3f', '.3f', '.3f', '.6f', '.6f', '.6f'],
              data=states,
              input_data_type='STATES',
              output_spk_type=5,
              object_id=self.sc.spice_id,
              object_name=f'VOYAGER {self.vgr_id}',
              center_id=399,
              center_name='EARTH',
              ref_frame_name='J2000',
              producer_id='https://github.com/kwan3217/rocket_sim',
              data_order='EPOCH X Y Z VX VY VZ',
              input_data_units=('ANGLES=DEGREES', 'DISTANCES=m'),
              data_delimiter=' ',
              leapseconds_file='data/naif0012.tls',
              pck_file='products/gravity_EGM2008_J2.tpc',
              segment_id=f'VGR{self.vgr_id}_PM',
              time_wrapper='# ETSECONDS',
              comment="""
               Best-estimate of trajectory through Propulsion Module burn,
               hitting historical post-MECO2 orbit elements which are
               available and matching Horizons data."""
              )
    def initial_guess(self):
        raise NotImplementedError


class PMTargeter(BackpropagationTargeter):
    def __init__(self,*,vgr_id:int,out_path:Path=None):
        """

        :param vgr_id:
        """
        super().__init__(vgr_id=vgr_id,report_name="backpropagation through PM burn",
                         t1=horizons_et[vgr_id] - voyager_et0[vgr_id],out_path=out_path,
                         y1=np.array(horizons_data[vgr_id][3:9]) * 1000.0,fps=100)  # Convert km to m and km/s to m/s
        self.plotlines["spd"]=[lambda bpt:vlength(bpt.state[3:6,:]),"m/s"]
        self.plotlines["e"] = [lambda bpt:np.array([this_elorb.e for this_elorb in bpt.elorb]),""]
        self.plotlines["i"] = [lambda bpt:np.array([np.rad2deg(this_elorb.i) for this_elorb in bpt.elorb]),"deg"]
        self.plotlines["a"] = [lambda bpt:np.array([this_elorb.a for this_elorb in bpt.elorb]) / 1852,"nmi"]  # work directly in km
        self.param_names=("dpitch","dyaw","dthr","yawrate")
        self.param_units=("deg", "deg", None, "deg/s")
        self.t0=simt_track_prePM[self.vgr_id]
        self.save="pm"
    def accept_params(self, params):
        super().accept_params(params)
        self.dpitch, self.dyaw, self.dthr, self.yawrate = params
    def print_params(self)->str:
        return f"""dpitch: {self.dpitch:.13e} ({self.dpitch.hex()})
dyaw: {self.dyaw:.13e} ({self.dyaw.hex()})
dthr: {self.dthr:.13e} ({self.dthr.hex()})
yawrate: {self.yawrate:.13e} ({self.yawrate.hex()})"""
    def guide(self):
        return seq_guide({self.sc.tdrop[self.sc.i_centaur]: prograde_guide,
                          float('inf'): yaw_rate_guide(r0=self.y1[:3],v0=self.y1[3:],
                                                       dpitch=self.dpitch, dyaw=self.dyaw, yawrate=self.yawrate,
                                                       t0=self.sc.tburn[self.sc.i_pmengine][0])})
    def tweak_sc(self):
        # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
        self.sc.stages[self.sc.i_centaur].prop_mass = 0
        self.sc.stages[self.sc.i_pm].prop_mass = 0
        # Tweak engine efficiency
        self.sc.engines[self.sc.i_pmengine].eff = 1.0 + self.dthr
    def cost_components(self,*, ouf: TextIO | None)->dict[str,tuple[float,float,float,int,str]]:
        """

        :param ouf:
        :return: dict -- Key is name of component
                         value is:
                          * Targeted value of this component
                          * Achieved value of this component
                          * Sigma of this component
                          * Pane to draw target into
                          * Unit string for this component
        """
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
        y0_j = self.sc.y.copy()
        Mej = pxform('J2000', 'IAU_EARTH', simt_track_prePM[self.vgr_id] + voyager_et0[self.vgr_id])
        r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
        v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
        r0_e = Mej @ r0_j
        v0_e = Mej @ v0_j
        # In this frame, the reported ascending node is Longitude of ascending node, not
        # right ascension. It is relative to the Earth-fixed coordinates at this instant,
        # not the sky coordinates.
        elorb0 = earth.elorb(rv=r0_e, vv=v0_e, tref=simt_track_prePM[self.vgr_id],deg=True)
        result={}
        result["a"]=(elorb0.a/1852,target_a_prePM[self.vgr_id]/1852,2e-1   ,4,"nmi")  # da in nmi
        result["e"]=(elorb0.e     ,target_e_prePM[self.vgr_id]     ,0.00007,2,"")  # de
        result["i"]=(elorb0.i     ,target_i_prePM[self.vgr_id]     ,0.01   ,3,"deg") #di
        return result
    def initial_guess(self):
        guess=best_pm_initial_guess(vgr_id=self.vgr_id)
        return guess


class Centaur2Targeter(BackpropagationTargeter):
    def __init__(self, *, pm:Vehicle|ParsedPM,out_path:Path=None):
        if isinstance(pm,ParsedPM):
            vgr_id = pm.vgr_id
            t1 = pm.simt0
            y1 = pm.y0
        else:
            vgr_id=pm.vgr_id
            t1=pm.t[-1]
            y1=pm.y
        super().__init__(vgr_id=vgr_id, report_name="backpropagation through Centaur 2 burn",
                             t1=t1,y1=y1,fps=100,out_path=out_path)  # Convert km to m and km/s to m/s
        self.plotlines["spd"] = [lambda bpt: vlength(bpt.state[3:6, :]), "m/s"]
        self.plotlines["e"] = [lambda bpt:np.array([this_elorb.e for this_elorb in bpt.elorb]),""]
        self.plotlines["i"] = [lambda bpt:np.array([np.rad2deg(this_elorb.i) for this_elorb in bpt.elorb]),"deg"]
        self.plotlines["c3"] = [lambda bpt: -(earth.mu / (1000 ** 3)) / (
                    np.array([this_elorb.a for this_elorb in bpt.elorb]) / 1000),"km**2/s**2"]  # work directly in km
        # Run back to this point at 100Hz, then back to parking solution at 10Hz
        self.sc:Vehicle = Titan3E(tc_id=5+self.vgr_id)
        self.param_names=("dpitch","dyaw","dthr","pitchrate")
        self.param_units=("deg", "deg", None, "deg/s")
        self.t0 = self.sc.tburn[self.sc.i_cengine2][0]-10
        self.bounds = [(-30, 30), (-30, 30), (-0.1, 0.1), (-1, 1)]  # Freeze yaw rate at 0
        self.save = "centaur2"
    def accept_params(self, params):
        super().accept_params(params)
        self.dpitch, self.dyaw, self.dthr, self.pitchrate = params
    def print_params(self) -> str:
        return f"""dpitch: {self.dpitch:.13e} ({self.dpitch.hex()})
dyaw: {self.dyaw:.13e} ({self.dyaw.hex()})
dthr: {self.dthr:.13e} ({self.dthr.hex()})
pitchrate: {self.pitchrate:.13e} ({self.pitchrate.hex()})"""
    def guide(self):
        return dprograde_guide(dpitch=self.dpitch, dyaw=self.dyaw, pitchrate=self.pitchrate, t0=self.sc.tburn[self.sc.i_cengine2][0])
    def tweak_sc(self):
        # Propellant tank "starts" out empty and fills up as time runs backwards (but not mission module)
        self.sc.stages[self.sc.i_centaur].prop_mass = 0
        # Tweak engine efficiency
        self.sc.engines[self.sc.i_cengine2].eff = 1.0 + self.dthr
    def sim(self,ouf:TextIO=None):
        super().sim(ouf=ouf)
        self.universe.change_fps(10)
        self.universe.runto(t1=simt_track_park[self.vgr_id])
        self.plot_lines()
        finalstate = report_state(t=self.sc.tlm_points[-1].t + self.sc.tlm_points[-1].dt, y=self.sc.tlm_points[-1].y1,
                                  name='State just after Centaur burn 1')
        print(finalstate)
        if ouf is not None:
            print(finalstate,file=ouf)
    def cost_components(self,*, ouf: TextIO | None)->dict[str,tuple[float,float,float,int,str]]:
        """

        :param ouf:
        :return: dict -- Key is name of component
                         value is:
                          * Targeted value of this component
                          * Achieved value of this component
                          * Sigma of this component
                          * Pane to draw target into
                          * Unit string for this component
        """
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
        y0_j = self.sc.y.copy()
        Mej = pxform('J2000', 'IAU_EARTH', simt_track_prePM[self.vgr_id] + voyager_et0[self.vgr_id])
        r0_j = y0_j[:3].reshape(-1, 1)  # Position vector reshaped to column vector for matrix multiplication
        v0_j = y0_j[3:].reshape(-1, 1)  # Velocity vector
        r0_e = Mej @ r0_j
        v0_e = Mej @ v0_j
        # In this frame, the reported ascending node is Longitude of ascending node, not
        # right ascension. It is relative to the Earth-fixed coordinates at this instant,
        # not the sky coordinates.
        elorb0 = earth.elorb(rv=r0_e, vv=v0_e, tref=simt_track_park[self.vgr_id],deg=True)
        result={}
        result["a"]=(elorb0.a/1852,target_a_park[self.vgr_id]/1852,2e-1   ,5,"nmi")  # da in nmi
        result["e"]=(elorb0.e     ,target_e_park[self.vgr_id]     ,0.00007,3,"")  # de
        result["i"]=(elorb0.i     ,target_i_park[self.vgr_id]     ,0.01   ,4,"deg") #di
        return result
    def initial_guess(self):
        guess= best_centaur2_initial_guess(vgr_id=self.vgr_id)
        return guess


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
    for vgr_id in (1,2):
        pm_targeter=PMTargeter(vgr_id=vgr_id)
        pm_targeter.target(optimize=False)
        pm_targeter.export()
        pm_guess=best_pm_solution(vgr_id=vgr_id)
        centaur2_targeter=Centaur2Targeter(pm=pm_guess)
        centaur2_targeter.target(optimize=False)
        centaur2_targeter.export()
    #for vgr_id in (1,2):
    #    export(vgr_id)


if __name__=="__main__":
    main()
