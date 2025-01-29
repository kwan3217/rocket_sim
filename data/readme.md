Files containing data extracted from various reports. The orignals
of all of these are primary sources -- flight data reports, etc.

## `vgr1_st.bsp`
From NAIF, originally called `Voyager_1.a52406u_V0.2_merged.bsp`
```
; Voyager_1.ST+1991_a54418u.merged.bsp LOG FILE
; Created 2004-04-26/14:11:05.00.
;
; This file is a merge of at least two files:
;
; #1) 1977-Sep-5 to 1986-Jan-1:
;
;      The Voyager 1 "supertrajectory" nio file obtained from George Lewis by
;      Chuck Acton on 23 May 1996. The original producer was S. Mastousek who
;      last updated it in 1987 on the VAX ("Groucho") system.
;
;      This interval provides a patched conic mission-design type trajectory
;      in which the conics are constrained to match specific events (satellite
;      encounters), providing a rough accuracy which was used throughout the
;      Voyager mission. The file started shortly after launch and originally
;      extended to 2050. See IOM Voy-NAV-87-16, dated 31 Mar 87.
;
; #2) 1986-Jan-1 to 2021-Jan-1:
;
;      pfile_a54418u.nio, provided 2004-Apr-22 by George Lewis (JPL) to
;      Jon Giorgini.
;
;      This solution is described in IOM 314.7-175, 22 May 92. It is based on
;      actual tracking data from 1986 Mar 3 to 1992 Apr 24, a data arc
;      extending one year beyond that of the last formal solution actually
;      delivered to the Voyager Project for DSN tracking.
;
```
This is used instead of the file labeled `a54206u` because the interstellar
tracking data on that kernel has the following note:
```
#2) 1986-Jan-1 to 2030-Dec-31
 
    "pfile_a54206u" based on data March 1986 to Feb 1990. Note that the
    previously used solution "54418u" used in the merged SPK file
    "voyager_1.ST+1991_a54418u.merged.bsp" was based on data Mar 1986
    to April 1992 and included modeling of the additional two years of
    turn maneuvers and non-gravs since "54206u", so using the
    "pfile_a54206u" in this merged SPK is a regression to an older
    trajectory that runs out longer.
 
```
Note that as of this writing, we have actually run off the end of the chosen
kernel, but my interest is not related to the interstellar mission.

## `vgr2_st.bsp`
From NAIF, originally called `voyager_2.ST+1992_m05208u.merged.bsp`
```
; Voyager_2.ST+1992_m05208u.merged.bsp LOG FILE
 
; Created 2004-04-26/14:28:37.00.
;
; This file is a merge of at least two files:
;
; #1) 1977-Aug-20 to 1990-Jan-23:
;
;      The Voyager 2 "supertrajectory" nio file obtained from George Lewis by
;      Chuck Acton on 23 May 1996. The original producer was S. Mastousek who
;      last updated it in 1987 on the VAX ("Groucho") system.
;
;      This section is is a patched conic mission-design type trajectory in
;      which the conics are constrained to match specific events (satellite
;      encounters), providing a rough accuracy, which was used throughout the
;      Voyager mission. The file started shortly after launch and originally
;      extended to 2050. See IOM Voy-NAV-87-16, dated 31 Mar 87.
;
; #2) 1990-Jan-23 to 2021-Jan-5:
;
;      Voyager Nav Team solution of Feb 23, 1993: pfile_m05208u.nio, provided
;      2004-Apr-22 by George Lewis (JPL) to Jon Giorgini.
;
;      This solution is described in IOM 314.7-227, 02 Mar 93. It is based on
;      actual tracking data from 1990 Jan 23 to 1992 Oct 30, a data arc
;      extending beyond that of the last formal solution delivered to the
;      Voyager Project for DSN tracking (M05016U) and is based on an additional
;      year of tracking after the Neptune encounter.
;
```
Part 1 of the merged supertrajectory is a prediction of the Neptune encounter.
Part 2 should include it, since it includes tracking data through 1993.

Again, we use this rather than the other supertrajectory because it has the note:
```
#2) 1989-10-01 to 2030-Dec-31
 
    "pfile_m05016u_V0.2", based on tracking data Oct 1989 - Oct 1990.
    Note that the previously used solution  "m05208" used in the merged
    SPK file "voyager_2.ST+1992_m05208u.merged.bsp" was based on Jan
    1990 - Oct 1992 data and included modeling of additional maneuver
    delta-v's that occurred over the two year span since m05016u, so
    using the "pfile_m05016u_V0.2" in this merged SPK is a regression
    to an older trajectory that runs out longer.
```

## `TitanSRM start curve.dig`
Trace of SRM 68 start transient from TC-6 (Voyager 1 launch vehicle)
from figure 7-3, p54 of the TC-6 flight data report. Since the scale is
in PSI Absolute, the chart has been manually edited such that the first
point is t=0,P=14.7

## `TitanSRM thrust curve.dig`
Trace of generic thrust curve from Titan 3E Centaur D-1T Systems Summary,
figure 6-15, p205. Curve is interpreted as being at standard conditions
and at sea level throughout the burn -- it's probably imported data
from a test on a test stand.

## `UA1205 Thrust Curve.dig`
Trace of a generic UA1205 motor, from the 120" Potential, figure 1. Curve
is stated to be vacuum thrust at grain temperature of 80degF, very near
the Voyager launch temperature of 82.5degF.