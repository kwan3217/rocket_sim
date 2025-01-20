# rocket_sim

`rocket_sim` is designed to be a scriptable framework for evaluating
astrodynamics problems, in particular powered flight including launch
from the surface. This project aims to simulate the dynamics of
rocket propulsion systems, from static test stands to complex orbital
maneuvers, offering insights into vehicle performance under various
conditions.

## Key Features

- **Modular Vehicle Description**: The simulation uses a class-based
  approach (`VehicleDesc`) to define the physical attributes of
  launch vehicles and spacecraft. This includes multiple stages,
  propellant management, and engine characteristics.

- **Scriptable Guidance**: Guidance logic can be attached to vehicles,
  allowing for customizable control strategies. We currently support
  basic thrust direction control and plan to expand to include
- sophisticated algorithms like Powered Explicit Guidance (PEG).

- **Test Stand Simulation**: Before delving into full orbital 
  simulations, a static test stand simulation mode allows for the
  verification of vehicle performance metrics like thrust, mass
  change over time, and delta-V capability.

- **Orbital Dynamics**: The simulator includes models for two-body
  gravity and J2 perturbations, key for accurate Earth orbit
  simulations. It is also possible to implement a zero-g test range
  to evaluate only thrust dynamics for testing purposes.

- **Reverse Time Simulation**: An innovative feature for debugging
  and understanding the reverse propagation of trajectories,
  particularly useful for reconstructing or backtracking mission
  profiles from known endpoints.

- **Integration with RK4**: Utilizes the Runge-Kutta 4th order
  method for accurate numerical integration of motion equations,
  which can be run forward or backward in time.

## Implementation Roadmap

- **Phase 1 - Test Stand** Complete!:
  - Implement a basic vehicle model with a single propulsion module.
  - Simulate static firings to verify thrust and mass flow
    characteristics.
  - Provide plots for mass, thrust, acceleration, and delta-V over time.

- **Phase 2 - Orbital Motion**:
  - Incorporate gravitational forces, including simple two-body
    dynamics and J2 perturbations. Complete!
  - Extend the vehicle model to include multi-stage rockets like the
    Titan-Centaur for the Voyager mission. Complete!
  - Incorporate drag forces and an atmosphere model valid all the way
    down to the surface.
  - Simulate the launch into parking orbit and subsequent maneuvers
    like the dogleg for Voyager 2. In progress.

- **Phase 3 - Trajectory Optimization**:
  - Use optimization libraries like `scipy.optimize` for tuning
    initial conditions or guidance parameters to match historical data
    or specific mission requirements. Complete!
  - Implement reverse time integration for trajectory reconstruction,
    starting from known final states (e.g., from SPICE kernels). Complete!

## Road map
Current MVP achieved are:

* Vehicle can be run on a test stand to get thrust, acceleration, and mass
* Vehicle can be run on zero-g test range to get delta-v
* Gravity and sequencing is implemented to support PM maneuver for Voyager
* Simulation can run in reverse to support de-boost, propagating backwards
  through the PM maneuver to reconstruct the pre-PM orbit from the PM maneuver
  and known post-PM orbit.
- Orbital simulation includes basic two-body, J2 of central body, and third-body
  perturbations.
* Integration with `scipy.optimize.minimize` to tune PM maneuver to better hit
  the partially-documented pre-PM orbit. Completed!
* Export Spice kernels for import to Cosmographia. Completed!

Next MVP(s):
* De-boost through Centaur burn 2 to get to parking orbit.
* Implement the rest of the Titan-Centaur 3E vehicle
* Implement drag
* Import the Powered Explicit Guidance maneuver
* Import the documented stage 0 pitch program
* Detailed thrust profiles for solid and liquid engines.
* Export POV and/or Blender to visualize the results
* Extend to 6DoF, including rotational dynamics and kinematics, moments of inertia,
  and steering by actuators (like RCS, engine gimbal, control surface, etc) instead
  of just pointing the direction Guidance says.

