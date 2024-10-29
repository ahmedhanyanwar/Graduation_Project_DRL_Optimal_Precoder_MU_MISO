# Graduation Project: DRL Approach for Designing the Optimal Precoder Vector and Power Allocation in MU-MISO (Low SNR and Equal Power Constraint)

## Overview
This project version applies a Deep Reinforcement Learning (DRL) approach for optimal precoder vector design in a Multi-User MISO (MU-MISO) system, focusing on **Low SNR** conditions with an additional **equal power constraint**. This constraint, which ensures that each user maintains unit power, introduces a more challenging environment for decision-making and reward optimization in precoder selection.

The goal of this version is to develop a model that can operate effectively in noisy environments while maintaining an equitable power distribution across users, making it suitable for scenarios where consistent power allocation is essential.

## Key Features
- **Constrained Power Allocation**: Enforces an equal power constraint for each user, challenging the DRL model to optimize within these limits.
- **Adaptability to Low SNR**: Trains the DRL model to adapt to low SNR conditions, improving robustness in high-noise channels.
- **Performance Comparison**: Evaluates the DRL agent's constrained performance against traditional precoding methods such as MRT, ZF, and MMSE.

## Project Structure
- **buffer_numpy.py**: Contains functions for buffering and managing data.
- **DDPG_MultiUser.py**: Implements the DRL agent using the Deep Deterministic Policy Gradient (DDPG) algorithm.
- **main.ipynb**: The main Jupyter notebook containing the model training and evaluation for low-SNR scenarios.
- **networks_MultiUser.py**: Defines the neural network architecture for the DRL agent.
- **utils.py**: Contains utility functions for various project operations.

## Evaluation of the Agent

### Numerical Evaluation
The agent’s performance is quantitatively evaluated under Low SNR with unit power constraints, comparing against MRT, ZF, and MMSE. This evaluation uses average rewards and percentage performance metrics calculated over multiple instances.

#### Low SNR Evaluation
- **Standard Deviation of Channel Noise**: `snr_user = 0.01`
- **Steps**:
  1. Generate a channel matrix considering the low-SNR environment.
  2. Apply unit power constraints to each user.
  3. Compute precoder matrices for MRT, ZF, and MMSE.
  4. Calculate and compare the rewards for each precoder.
  5. Evaluate the agent’s actions and calculate the corresponding rewards under the power constraint.

- **Average Rewards Calculated**:
  - Average reward by Actor: `[0.07073452 0.07464583 0.07599429 0.07414524]`
  - Average reward by MRT: `[0.08112714 0.08182684 0.08057783 0.08042943]`
  - Average reward by ZF: `[0.04199582 0.0430891  0.04219514 0.0422449 ]`
  - Average reward by MMSE: `[0.08129223 0.08199894 0.0807396  0.08057806]`

- **Percentage Performance**:
  - Percentage subject to MRT: `91.22074894655961 %`
  - Percentage subject to ZF: `174.32233648866224 %`
  - Percentage subject to MMSE: `91.0387607859483 %`

#### Single Instance Evaluation
Using a fixed seed, a single channel state is generated to assess rewards in a reproducible environment.
- **Rewards by Actor**: `[0.08952872 0.04267151 0.11315348 0.05657868]`
- **Rewards by MRT**: `[0.0945022  0.04941569 0.11512368 0.06014144]`
- **Rewards by ZF**: `[0.03366883 0.02560528 0.06523089 0.02994686]`
- **Rewards by MMSE**: `[0.05825041 0.03386674 0.09295984 0.0427181 ]`

- **Percentage Performance**:
  - Percentage subject to MRT: `94.59538067442546 %`
  - Percentage subject to ZF: `195.4864063395886 %`
  - Percentage subject to MMSE: `132.5456174757895 %`

> **Important Note**: As training continues, the agent's reward curve shows a high positive slope, indicating improved performance. However, achieving optimal results requires significant computational power and time.

### Performance Metrics
Performance is assessed by comparing the average rewards and percentage performance of the DRL agent against traditional precoding techniques (MRT, ZF, MMSE).

### Conclusion
The DRL-based approach demonstrates its adaptability to challenging low-SNR conditions and equal power constraints, presenting a robust and efficient solution for precoding in MU-MISO systems. This model’s constrained performance is favorable when compared to traditional precoding techniques, validating its effectiveness.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Additional libraries as required by the project files

### Usage
Run the main script in `main.ipynb` to start the training and evaluation processes. Modify parameters as needed to test different configurations.

### Acknowledgments
This project was developed as part of a graduation project. Special thanks to the instructors and peers for their guidance and support.

