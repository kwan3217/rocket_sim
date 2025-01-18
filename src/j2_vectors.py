import numpy as np
import matplotlib.pyplot as plt

from rocket_sim.gravity import aJ2


def plot_j2_acceleration():
    """
    Plot the acceleration due to J2 on the XZ plane through the planet B612.
    This function uses the aJ2 function to compute J2 acceleration at various points
    on the XZ plane and visualizes it with a vector plot.
    """
    # Constants for B612
    gm = 15.915494309189535  # Gravitational parameter in m**3/s**2, from previous tests
    Re = 10.0  # Equatorial radius of B612 in meters
    J2 = 0.1  # J2 value for dramatic effect

    # Create J2 gravity function
    j2_B612 = aJ2(j2=J2, gm=gm, re=Re)

    # Define the grid for XZ plane
    xs = np.linspace(-2*Re, 2*Re, 20)
    zs = np.linspace(-2*Re, 2*Re, 20)

    # Plotting
    fig, axis = plt.subplots(figsize=(10, 10))

    # Compute J2 acceleration at each grid point
    for x in xs:
        for z in zs:
            # Set y to 0 since we're on the XZ plane
            if x**2+z**2>Re**2:
                y = np.array([x, 0, z, 0, 0, 0])
                # Time and other parameters are set to arbitrary values since they don't affect J2
                t = 0
                dt = 1
                # Compute acceleration
                ax,ay,az = j2_B612(t=t, y=y, dt=dt)
            else:
                ax,ay,az=np.zeros(3)*float('nan')
            scale=100
            axis.arrow(x,z,ax*scale,az*scale,color='b',width=0.1)


    # Add a circle to represent B612
    circle = plt.Circle((0, 0), Re, color='r', fill=False)
    axis.add_artist(circle)

    axis.set_xlabel('X position (m)')
    axis.set_ylabel('Z position (m)')
    axis.set_title('Acceleration due to J2 on XZ Plane')
    plt.axis('equal')
    plt.show()


# Run the plotting function
plot_j2_acceleration()