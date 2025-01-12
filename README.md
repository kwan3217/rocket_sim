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

- **Phase 1 - Test Stand**:
  - Implement a basic vehicle model with a single propulsion module.
  - Simulate static firings to verify thrust and mass flow
    characteristics.
  - Provide plots for mass, thrust, acceleration, and delta-V over time.

- **Phase 2 - Orbital Motion**:
  - Incorporate gravitational forces, including simple two-body
    dynamics and J2 perturbations.
  - Extend the vehicle model to include multi-stage rockets like the
    Titan-Centaur for the Voyager mission.
  - Incorporate drag forces and an atmosphere model valid all the way
    down to the surface.
  - Simulate the launch into parking orbit and subsequent maneuvers
    like the dogleg for Voyager 2.

- **Phase 3 - Trajectory Optimization**:
  - Use optimization libraries like `scipy.optimize` for tuning
    initial conditions or guidance parameters to match historical data
    or specific mission requirements.
  - Implement reverse time integration for trajectory reconstruction,
    starting from known final states (e.g., from SPICE kernels).

## Current Status
Nothing yet, but the current MVP is:

- Basic vehicle description and guidance classes are implemented.
- Test stand simulation is operational, providing insights into
  vehicle dynamics under static conditions.
- Orbital simulation includes basic two-body and J2 effects with thrust.

## Future Enhancements

- Detailed thrust profiles for solid and liquid engines.
- Support for more complex guidance algorithms.
- Integration with external astrodynamics tools for validation and
  more complex scenarios.

## Usage

For detailed instructions on how to use the simulator, please refer to
the `docs/` folder or run the example scripts in `examples/`.
Contributions and suggestions are welcome; please see our
[CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.
