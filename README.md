### This is a public version of the developing project

# NN_optctrl

## Hardware-Aware Quantum Optimal Control with Neural Networks

This project implements a workflow for designing and experimentally calibrating high-fidelity quantum control pulses. The approach systematically bridges the gap between simulations and experiments by combining traditional numerical optimization with a multi-stage neural network based approach.

---
## The Optimization Workflow

The process is broken down into three main stages:

### 1. Informed Initialization (offline)
An ideal control pulse is first generated in a noiseless simulation using a standard numerical method (e.g., GRAPE). A neural network is then pre-trained via supervised learning (MSE loss) to replicate this pulse. This step effectively "distills" the numerical solution into a smooth, compact, and differentiable function generator, providing an excellent, physically-motivated starting point. This can be used to quickly test a hardware optimization of any pre-generated pulse.
We have added custom initialization which works without the need for a pre-optimized pulse. This can be used to generate new optimal control pulse. 

### 2. Robustness Training (offline)
The pre-trained network is further fine-tuned using backpropagation against a simulated fidelity loss. This stage is ideal for introducing noise models (dephasing, amplitude noise, etc.) into the physics simulation, and including variation of Hamiltonian parameters, forcing the network to learn pulses that are robust to common experimental imperfections.

### 3. Hardware-in-the-Loop Calibration (online)
To close the simulation to experiment gap, the final layers of the network are fine-tuned directly on the quantum hardware. A gradient-free algorithm, **Simultaneous Perturbation Stochastic Approximation (SPSA)**, is used to optimize the pulse based on real-time experimental feedback, correcting for unmodeled physical effects and calibration errors. The final, hardware-calibrated network serves as an efficient, high-performance pulse generator for experiments.
Other options of using bayesian learning, reinforcement learning and upgrades to SPSA (second order/ QN SPSA) are currently under development.
