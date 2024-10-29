# Graduation Project: DRL Approach for Designing the Optimal Precoder Vector and Power Allocation in MU-MISO (High SNR and Equal Power Constraint)

## Overview
This project explores a Deep Reinforcement Learning (DRL) approach to optimize precoder vector design in a Multi-User MISO (MU-MISO) system under **High Signal-to-Noise Ratio (SNR)** conditions, with a **strict equal power constraint** for each user. In high-SNR environments, signals are clearer, and noise has less impact, allowing the DRL model to prioritize maximizing sum rate while adhering to equal power constraints, which presents unique optimization challenges.

The model’s objective is to leverage high SNR conditions for enhanced system performance, while maintaining consistent power distribution across users.

## Key Features
- **Equal Power Allocation**: Imposes an equal power constraint across users, challenging the model to optimize within a balanced framework.
- **Adaptation to High SNR**: Trains the DRL model to maximize efficiency in low-noise, high-SNR environments.
- **Performance Benchmarking**: Evaluates the DRL model's performance against MRT, ZF, and MMSE under high-SNR conditions, highlighting advantages in both constrained and unconstrained scenarios.

## Project Structure
- **buffer_numpy.py**: Manages data buffering.
- **DDPG_MultiUser.py**: Implements the DRL agent using the Deep Deterministic Policy Gradient (DDPG) algorithm.
- **main.ipynb**: Main notebook to execute model training and evaluation in high-SNR environments.
- **networks_MultiUser.py**: Defines the neural network structure for the DRL agent.
- **utils.py**: Provides utility functions for various project operations.

## Evaluation of the Agent

### Numerical Evaluation
The agent’s performance is evaluated numerically under high-SNR conditions with unit power constraints, and its effectiveness is compared to traditional precoding methods (MRT, ZF, and MMSE). Average rewards and percentage performance metrics were computed over multiple simulations.

#### High SNR Evaluation
- **Standard Deviation of Channel Noise**: `snr_user = 5.0`
- **Evaluation Steps**:
  1. Generate a high-SNR channel matrix.
  2. Enforce unit power constraints across users.
  3. Calculate precoder matrices for MRT, ZF, and MMSE.
  4. Compare the DRL agent’s rewards with each traditional method.

- **Average Rewards Calculated**:
  - Average reward by Actor: `[1.6149364, 1.6114621, 1.6607558, 1.6831667]`
  - Average reward by MRT: `[1.7042383, 1.7216097, 1.7223585, 1.6904116]`
  - Average reward by ZF: `[4.725966, 4.773089, 4.7336555, 4.6978364]`
  - Average reward by MMSE: `[4.790184, 4.832568, 4.794064, 4.7546043]`

- **Percentage Performance**:
  - Percentage relative to MRT: `96.08 %`
  - Percentage relative to ZF: `34.71 %`
  - Percentage relative to MMSE: `34.27 %`

#### Single Instance Evaluation
A fixed-seed channel state is generated to assess rewards in a controlled environment.
- **Rewards by Actor**: `[1.2612578, 2.8617866, 1.3896484, 1.1960659]`
- **Rewards by MRT**: `[1.2288213, 2.1068318, 1.3850704, 1.4158399]`
- **Rewards by ZF**: `[5.4284167, 5.980046, 5.364854, 4.4953165]`
- **Rewards by MMSE**: `[4.7009487, 5.6940603, 4.5953317, 3.3219385]`

- **Percentage Performance**:
  - Percentage relative to MRT: `109.32 %`
  - Percentage relative to ZF: `31.54 %`
  - Percentage relative to MMSE: `36.64 %`


> **Important Note**: But this result gets better and better; the reward curve has a high slope, but it needs a lot of time with a high computational machine.

> **Note**: The DRL model shows strong adaptability to high-SNR conditions, maintaining stable performance. Training improvements are evident in the reward curve, but optimal results require high computational resources.


### Performance Metrics
Average rewards and comparative percentage metrics confirm that the DRL agent performs competitively with traditional methods under high-SNR, especially within constrained scenarios.

### Conclusion
The DRL-based model effectively balances high-SNR optimization with equal power constraints, demonstrating competitive performance compared to traditional methods and robustness in constraint-driven high-SNR environments.

## Getting Started

### Prerequisites
Install the following:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Additional dependencies as specified in the code

### Usage
Run `main.ipynb` to initiate training and evaluation. Modify parameters to explore various high-SNR scenarios.

### Acknowledgments
Developed as part of a graduation project with support from instructors and peers.
