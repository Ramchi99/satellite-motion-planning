# Satellite Motion Planning

*🏆 **Ranked 1st out of 120 student groups** on the private test cases in the Planning and Decision Making for Autonomous Robots (PDM4AR) master's course at ETH Zurich.*

## Overview
This project implements a highly optimized motion planning algorithm for a satellite to safely navigate out of a docking bay. The objective was to compute collision-free, optimal trajectories in a complex environment using advanced convexification techniques.

## Visuals
| Config 1 | Config 2 | Config 3 |
| :---: | :---: | :---: |
| <img src="animation_1.gif" width="300" /> | <img src="animation_2.gif" width="300" /> | <img src="animation_3.gif" width="300" /> |

## The Problem
* **Objective:** Navigate a satellite through a constrained space (docking bay) to a target while adhering to non-linear orbital dynamics and strict control limits.
* **Challenges:** Non-linear dynamics require complex optimization, and control inputs must be rigorously mapped to the physical simulation.

## Our Approach & Crucial Considerations
Our solution relies on a highly tuned implementation of the **Successive Convexification (SCvx)** algorithm, inspired by *[Szmuk et al., "Successive Convexification: A Superlinearly Convergent Algorithm for Non-convex Optimal Control Problems" (arXiv:2106.09125)](https://arxiv.org/abs/2106.09125)*.

* **Successive Convexification (SCvx):** We solved the non-linear optimal control problem by successively linearizing the dynamics (using First-Order Hold discretization) and the non-convex obstacle constraints. We utilized a dynamic Trust Region mechanism that actively scales the allowed update steps based on the ratio of actual vs. predicted cost reduction. The convex subproblems were efficiently solved using CVXPY with the CLARABEL solver.
* **Robustness Considerations:** While we capitalized on the deterministic nature of the simulation, our design recognized that in a noisy, real-world environment, a low-level controller would be necessary. To accommodate this, our SCvx solution restricted the control bounds ($U_{max}$) slightly below the absolute physical limits, reserving "space" for a theoretical low-level LQR/PID controller to make corrections.
* **Docking Half-Plane Constraints:** The final docking maneuver required extreme precision. We implemented strict half-plane constraints mapped to the geometry of the docking bay, ensuring the satellite only entered from the correct approach vector.

### 🔑 The "Secret Sauce": Feed-Forward Control & Impulse Preservation
Because the simulation environment was deterministic, we could rely entirely on feed-forward control without the computational overhead of constantly replanning (like in traditional MPC). However, when you discretize a continuous-time system, naive control mapping often results in energy loss or trajectory drift between the solver's discrete steps and the simulation's continuous engine.

To solve this, we developed an **Impulse Preservation** mapping:

<p align="center">
  <img src="impulse_preservation.jpeg" alt="Impulse Preservation" width="600"/>
</p>

Instead of simply applying the solver's discrete control value directly (which might under-actuate or over-actuate during a timestep), we mathematically guarantee that the *integral of the control input* (the impulse) over the time step $\Delta t$ exactly matches the impulse intended by the optimizer's discretization scheme. 
By preserving the exact impulse across the piecewise control commands, our feed-forward trajectory tracked the simulation physics perfectly. This insight allowed us to achieve state-of-the-art precision using a pure feed-forward architecture, avoiding replanning and drastically lowering execution time compared to other groups.

## Setup and Execution
Due to the complex academic dependencies required for the physics engine and solvers (e.g., `dg-commons`, `cvxpy`, and `casadi`), this project is configured to run flawlessly inside a **VS Code Devcontainer**. This isolates the environment and guarantees it runs regardless of your host operating system.

### Recommended Installation (Docker + VS Code)
1. Ensure you have **Docker Desktop** running and **Visual Studio Code** installed with the "Dev Containers" extension.
2. Clone this repository and open it in VS Code.
3. VS Code will detect the `.devcontainer` folder and prompt you to **"Reopen in Container"**. Click it. (If it doesn't prompt, press `Ctrl+Shift+P` and type `Dev Containers: Reopen in Container`).
4. Docker will automatically pull the base image, install all the Poetry dependencies, and configure the Python environment.

### Running the Code
Once inside the Devcontainer, you have two simple ways to execute the simulation:

**Option 1: Using the VS Code Debugger (Recommended)**
* Open the "Run and Debug" panel in VS Code (`Ctrl+Shift+D`).
* Select the pre-configured **"Exercise13 - Run"** or **"Exercise13 - Debug"** profile and hit play.

**Option 2: Using the Terminal**
Run the built-in PDM4AR course CLI script directly:
```bash
poetry run pdm4ar-exercise --exercise 13
```

## Acknowledgements
This codebase was developed as the final project for the **Planning and Decision Making for Autonomous Robots (PDM4AR)** master's course at **ETH Zurich** (Fall 2025). The simulation environment, physics engine, and base exercise framework were provided by the [IDSC Frazzoli Lab](https://github.com/PDM4AR). You can find the original exercise descriptions [here](https://pdm4ar.github.io/exercises/).
