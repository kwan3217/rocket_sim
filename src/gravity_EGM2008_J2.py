"""
Create a text planetary constant kernel from WGS-84 and EGM2008. This program
quines itself into the header of the kernel it generates, for maximum documentation.
The file can be used as a python script to generate the kernel, or as the kernel
itself.

References:
    [WGS84] WORLD GEODETIC SYSTEM 1984 -- Its Definition and Relationships with Local Geodetic Systems
        2014-07-08 version 1.0.0. Retrieved from link at https://nsgreg.nga.mil/doc/view?i=4085
    [EGM2008] Earth Gravity Model 2008 internal documentation README_WGS84_2.pdf and data file
              EGM2008_to2190_TideFree in EGM2008_Spherical_Harmonics.zip downloaded from
              https://earth-info.nga.mil/php/download.php?file=egm-08spherical

The EGM2008 uses the WGS84 ellipsoid by reference, but has two inconsistencies. These are annoying
but insignificant at the accuacy I generally work with.

1. The equatorial radius for applying the gravity field coefficients is 6,378,136.3m. This is the
   Re value that should be used in any formula which applies a gravity field coefficient to determine
   the potential or acceleration due to this field component. It differs from the WGS84 value of
   exactly 6,378,137 m. The latter value should be used for geodetic latitude calculations, along
   with the other defining constants of WGS84.
2. The GM is different between EGM2008 and WGS84 in the last decimal place.

The process then is to:
* Calculate J2 from Cbar20 using the formula [WGS84] 3-1
* Calculate polar radius from equatorial radius and flattening and [WGS84] B-2
* Write WGS84 radii and GM as BODY399_RADII and BODY399_GM, and EGM2008 derived J2
  as BODY399_J2

Usage:
I recommend loading the kernels in the following order:
pck00011.tpc # Sizes and orientations of all planets including Earth
gm_de440.tpc # Masses of all planets and many satellites, including Earth
this kernel  # Cover up Earth mass from gm_de440.tpc and size from pck00011.tpc

Created: 1/17/25
"""
import numpy as np
from spiceypy import furnsh, gdpool

# The EGM2008 gravity model includes the WGS-84 ellipsoid by reference. BUT it also
# includes the following values in the documentation PDF. If you want to use the gravity
# model to full consistency, you have to use these coefficients. But, I don't care
# that much -- even my own obsession for absurd accuracy has limits. Also, as it turns
# out, we don't need them to normalize anything.

a_normalize = 6_378_136.3    # m
GM_normalize= 3.986004415e8  # m**3/s**2

#The first line of a 2 million line file describes the cbar20 coefficient
             #           1         2         3         4         5         6         7         8         9
             # 0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789
             #order l degree m   cbar_{lm}               sbar_{lm}              \sigma cbar           \sigma sbar
EGM2008_line1="    2    0   -0.484165143790815D-03    0.000000000000000D+00    0.7481239490D-11    0.0000000000D+00"
cbar20=float(EGM2008_line1[13:35].strip().replace("D","E")) #Note that the text has 15 significant decimal digits
                                                            #after the decimal point
# Equation 3-1 in the WGS84 document gives the relation between J2 and cbar20:
# $\bar{C}_{2,0}=-1\times\frac{J_2}{\sqrt{5}}$
# from which we have: J2=-sqrt{5}cbar20
J2=-np.sqrt(5)*cbar20

# This is slightly inconsistent -- the J2 is for the spheroid of radius a_normalize above, but we are going to use
# it as-is with the slightly different WGS-84 radius.

# WGS84 is used for shape model only
a_wgs84=6_378_137.0     # m,         equation 3-2
GM_wgs84=3.986004418e14 # m**3/s**2, equation 3-4
f_wgs84=1.0/298.257223563 # equation 3-3
# Polar radius b=a(1-f) from WGS84 equation B-2
b_wgs84=a_wgs84*(1-f_wgs84)
# Rotation rate is included in this kernel but PLEASE don't use it
# to actually calculate rotation. Use IAU_EARTH or a better model
# which does a better job of tracking rotation. The value in pck00011.tpc
# that best matches this is the prime meridian rate, the second
# coefficient of BODY399_PM. That value is in degrees per day, and
# has to be converted to rad/s to compare. See below for this comparison.
omega_wgs84=7.292115e-5 # rad/s, equation 3-5

def main():
    with open("products/gravity_EGM2008_J2.tpc","wt") as ouf:
        print("KPL/PCK",file=ouf)
        with open("src/gravity_EGM2008_J2.py","rt") as inf:
            for line in inf:
                print(line,file=ouf,end='')
        print('"""',file=ouf)
        print(r"\begindata",file=ouf)
        print(f"BODY399_RADII=({a_wgs84/1000:.3f},{a_wgs84/1000:.3f},{b_wgs84/1000:.9f})",file=ouf)
        print(f"BODY399_GM={GM_wgs84/1e9:.9e}",file=ouf) # Print to 9 figures after decimal to match input precision
        print(f"BODY399_J2={J2:.15e}",file=ouf) # Print to 15 figures after decimal to match input precision
        print(f"BODY399_OMEGA_WGS84={omega_wgs84:.6e}",file=ouf) # Print to given figures after decimal to match input precision
        print(r"\begintext",file=ouf)
        print('"""',file=ouf)
    # Now test that we can furnsh the file we just generated:
    furnsh("products/gravity_EGM2008_J2.tpc")
    print(gdpool("BODY399_RADII",0,3))
    print(gdpool("BODY399_GM",0,1)[0])
    print(gdpool("BODY399_J2",0,1)[0])
    print(gdpool("BODY399_OMEGA_WGS84",0,1)[0])
    # Check WGS84 rotation rate against the given value in pck00011.tpc
    try:
        furnsh("data/pck00011.tpc")
        print(np.deg2rad(gdpool("BODY399_PM",0,3)[1])/86400)
    except Exception:
        print("Didn't find pck00011.tpc")


if __name__ == "__main__":
    main()
