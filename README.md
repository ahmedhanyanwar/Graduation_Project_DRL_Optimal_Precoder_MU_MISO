# DRL Approach for Optimal Precoder Design and Power Allocation in MU-MISO

## Overview
This project applies Deep Reinforcement Learning (DRL) using the **Deep Deterministic Policy Gradient (DDPG)** algorithm to optimize precoder vector design and power allocation in Multi-User Multiple Input Single Output (MU-MISO) systems. The main objective is to maximize the sum rate in both high and low SNR environments under different power constraints, with the models trained to adapt to these conditions. This project features three versions, each exploring unique configurations of SNR conditions and power constraints, as detailed below.

## Versions

### Version 1: Low SNR with Equal Power Constraint
In this version, the DRL agent is trained to operate under **Low SNR** conditions with an **equal power constraint** applied to each user. This constraint ensures that each user has unit power, making it challenging to optimize the reward while maintaining consistent power allocation across users.

### Version 2: High SNR with Equal Power Constraint
This version explores the DRL model’s performance in **High SNR** conditions with the same **equal power constraint** as in Version 1. Training under high SNR allows the model to make more efficient decisions, as the effects of noise are less severe. Results from this version provide insights into the model's adaptability and robustness in scenarios with better signal quality.

### Version 3: Power Allocation with SNR as Input
In this latest version, the DRL agent has been designed to handle **variable power allocation** in **low and high SNR** conditions, allowing dynamic power distribution among users based on the current SNR. By introducing SNR as an input parameter, this model adapts to changing channel conditions and leverages flexible power allocation strategies to maximize overall performance.

## Project Structure

- **buffer_numpy.py**: Handles data buffering and management.
- **DDPG_MultiUser.py**: Implements the DDPG agent, with specific variations in code for Versions 1 & 2 (equal power constraint) and Version 3 (power allocation).
- **main.ipynb**: The main Jupyter notebook for model training and evaluation across versions.
- **networks_MultiUser.py**: Defines the neural network architecture used for the DRL agent, customized for each version with differences in power allocation handling.
- **utils.py**: Contains utility functions for auxiliary project tasks.

## Evaluation and Performance

Each version has been evaluated in both **Numerical Evaluation** and **Single Instance Evaluation** settings, where the agent’s performance has been benchmarked against traditional methods, such as **MRT**, **ZF**, and **MMSE** precoding.

### Version 1 & 2: Equal Power Constraint
- **Average Reward and Percentage Performance**: Evaluated under both **low and high SNR**, showing how the equal power constraint affects the DRL model's reward performance.
- **Comparisons**: Demonstrated a robust performance by achieving high percentage scores relative to MRT, ZF, and MMSE methods.

### Version 3: Power Allocation with SNR Input
- **Enhanced Adaptability**: Allowed for dynamic power allocation with SNR as an input parameter, leading to more responsive and efficient decisions in varying channel conditions.
- **Performance Metrics**: Exhibited improved adaptability, especially under changing SNR conditions, as compared to equal power-constrained versions.

## Current Improvements
Ongoing work includes the addition of **skip connections** and **batch normalization layers** to the neural network model to potentially enhance the performance. These improvements are expected to refine the network's ability to generalize and increase stability during training by reducing internal covariate shifts.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Additional libraries as specified in the project files.

### Usage
Run the `main.ipynb` notebook to initiate training and evaluation. Adjust parameters as needed based on the desired configuration or version.

## Acknowledgments
This project was developed as part of a graduation project. Special thanks to mentors and peers for their support and valuable feedback throughout its development.
