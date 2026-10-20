# Robot, task and observation model

## Evaluated execution profile

The reference experiments use `contact_proxy`: a seven-revolute-joint kinematic arm, rate-limited joint servo, parallel finger pads, and an inertial spherical object. Normal contact uses a bounded Kelvin–Voigt law; tangential contact uses regularized Coulomb friction. The object evolves under gravity, contact forces and torque. Opening the fingers releases the object through those dynamics.

The arm and finger positions form prescribed kinematic boundaries for the object dynamics. Contact reaction does not modify the arm's joint dynamics in this profile. All comparison methods execute through the same backend and servo.

| Parameter | Value |
|---|---:|
| Control period | 0.04 s |
| Physics substeps per action | 80 |
| Task horizon | 130 actions |
| Object radius | 0.017 m |
| Randomized object mass | 0.045–0.075 kg |
| Nominal friction coefficient | 0.75–1.10 |
| Low-friction stress multiplier | 0.45 |
| Cartesian command scale | 0.012 m |
| Gripper opening range | 0.012–0.038 m |

## Task completion

An episode must first lift the object by at least 35 mm relative to its resting height. Final placement must be within 27 mm of the goal in the table plane and 7 mm vertically, with object speed below 0.10 m/s. The fingers must be open beyond 30 mm, the tool must be at least 45 mm from the object, and all final conditions must hold for four consecutive steps.

Timeouts and physical failures retain their recorded outcomes. Evaluation uses the same seeded object/goal distributions for every policy.

## Observation schema

The 32-dimensional observation concatenates: object/tool displacement (3), goal/object displacement (3), tool/table height (1), object velocity (3), tool velocity (3), normalized joint position (7), scaled joint velocity (7), finger opening (1), two contact measurements (2), previous gripper action (1), and normalized episode time (1). Four observations form the policy history.

Object and tool measurements are simulator-state observations. Task phases, planning subgoals, expert controller memory and friction/mass parameters are outside the policy input. The local metric uses the 20-dimensional transform specified in [METHOD.md](METHOD.md).

## MuJoCo adapter

`mujoco_env.py` and `assets/koral_recovery.xml` provide a separate joint-dynamics/contact adapter. It has its own environment identifier and execution tests. Installation and checks are listed in [INSTALL.md](INSTALL.md); the reference tables identify the backend actually evaluated. No backend fallback is performed when a requested runtime is missing.
