"""
Describe purpose of this script here

Created: 1/16/25
"""
import numpy as np
import pytest
from matplotlib import pyplot as plt
from spiceypy import gdpool, furnsh, kclear

from rocket_sim.gravity import aTwoBody, aJ2, SpiceTwoBody, SpiceJ2, SpiceThirdBody
from rocket_sim.universe import Universe
from rocket_sim.vehicle import Stage, Vehicle


def plot_tlm(vehicle:Vehicle):
    states=np.array([tlm_point.y0 for t,tlm_point in vehicle.tlm_points.items()])
    plt.figure("pos")
    plt.plot(states[:,0],states[:,1],label='r')
    plt.axis('equal')
    plt.show()


@pytest.fixture
def rose():
    """
    Fixture to create a rose object represented by a Vehicle instance.

    This fixture constructs a Vehicle object that represents a rose with no propulsion,
    suitable for use in orbital dynamics tests.

    :return: A Vehicle instance representing the rose with no stages or engines.
    """
    rose = Vehicle(stages=[Stage(dry=1, prop=0)], engines=[])
    return rose


def test_aTwoBody(rose):
    """
    Test aTwoBody on planet B612. This perfectly spherical planet
    has a gravity field such that a circular orbit with a radius of
    100m/(2*pi) has a circumference of 100m, a speed of 1m/s, and
    a period of 100 s. It's radius is such that this circular orbit
    doesn't intersect the surface (10m is small enough).

    We test by putting a rose (no propulsion) into the above
    described circular orbit and then propagate the orbit for exactly
    100 seconds such that the rose **should** return to its
    initial condition.
    :return: None, but raise an exception if the test fails.
    """
    # Calculate GM based on the given conditions
    C=100.0 # Circumference of orbit
    r = C / (2 * np.pi)  # Radius of the orbit in meters
    v = 1.0  # Velocity in m/s
    T=C/v # Orbit period in s
    gm = r * v ** 2  # Gravitational parameter in m³/s²
    two_body_B612=aTwoBody(gm=gm)

    # Initial state vector
    y0 = np.array([r, 0, 0, 0, v, 0])

    # Construct a test universe
    B612=Universe(vehicles=[rose],accs=[two_body_B612],y0s=[y0],fps=16)
    B612.runto(t1=T)
    assert np.allclose(rose.y,y0)
    plot_tlm(rose)


def test_aJ2(rose):
    """
    It turns out that B612 is not a perfect sphere -- it has a noticeable J2.
    Test the J2 effect by putting a spacecraft in an orbit with a noticeable inclination
    :param rose:
    :return:
    """
    # Calculate GM based on the given conditions
    C=100.0 # Circumference of orbit
    r = C / (2 * np.pi)  # Radius of the orbit in meters
    re=10.0
    assert re<r,"Orbit too small -- inside of planet."
    j2=0.1

    v = 1.0  # Velocity in m/s
    T=C/v # Orbit period in s
    gm = r * v ** 2  # Gravitational parameter in m³/s²
    print(gm,re,j2)
    two_body_B612=aTwoBody(gm=gm)
    j2_B612=aJ2(j2=j2,gm=gm,re=re)

    # Initial state vector - 45deg inclination, initial node over prime meridian
    y0 = np.array([r, 0, 0, 0, v*np.sqrt(2)/2, v*np.sqrt(2)/2])

    # Construct a test universe
    B612=Universe(vehicles=[rose],accs=[two_body_B612,j2_B612],y0s=[y0],fps=16)
    B612.runto(t1=T*1)
    #assert np.allclose(rose.y,y0)
    plot_tlm(rose)


def test_SpiceJ2(rose):
    """
    Now we use
    :param rose:
    :return:
    """
    # Calculate GM based on the given conditions
    C=100.0 # Circumference of orbit
    r = C / (2 * np.pi) # Radius of the orbit in meters
    B612_id=20_000_000+0xB612
    furnsh("data/gravity_B612.tpc")
    re=gdpool(f"BODY{B612_id}_RADII",0,3)[0]*1000.0 # Radius of the planet from the kernel
    assert re<r,"Orbit too small -- inside of planet."

    v = 1.0  # Velocity in m/s
    T=C/v # Orbit period in s
    two_body_B612=SpiceTwoBody(B612_id)
    j2_B612=SpiceJ2(B612_id)

    # Initial state vector - 45deg inclination, initial node over prime meridian
    y0 = np.array([r, 0, 0, 0, v*np.sqrt(2)/2, v*np.sqrt(2)/2])

    # Construct a test universe
    B612=Universe(vehicles=[rose],accs=[two_body_B612,j2_B612],y0s=[y0],fps=16)
    B612.runto(t1=T*1)
    #assert np.allclose(rose.y,y0)
    plot_tlm(rose)
    kclear()


def test_SpiceThirdBody(rose):
    """
    Now we use
    :param rose:
    :return:
    """
    # Calculate GM based on the given conditions
    h=185000 # Altitude of orbit over Earth
    furnsh("data/de440.bsp")
    furnsh("data/pck00011.tpc")
    furnsh("data/gm_de440.tpc")
    furnsh("products/gravity_EGM2008_J2.tpc")
    center_id=399
    body_id=301 # Moon has the strongest third-body perturbation since it is closest
    re=gdpool(f"BODY{center_id}_RADII",0,3)[0]*1000.0 # Radius of the planet from the kernel
    gm=gdpool(f"BODY{center_id}_GM",0,1)[0]*1000.0**3 # GM of the planet from the kernel
    a=re+h

    v = np.sqrt(gm/a)  # Velocity in m/s
    atwo_body=SpiceTwoBody(center_id)
    aj2=SpiceJ2(center_id)
    athird_body=SpiceThirdBody(spice_id_center=center_id,spice_id_body=body_id,et0=0)

    # Initial state vector - 45deg inclination, initial node over prime meridian
    y0 = np.array([a, 0, 0, 0, v*np.sqrt(2)/2, v*np.sqrt(2)/2])

    # Construct a test universe
    B612=Universe(vehicles=[rose],accs=[atwo_body,aj2,athird_body],y0s=[y0],fps=1)
    B612.runto(t1=6000)
    #assert np.allclose(rose.y,y0)
    plot_tlm(rose)


