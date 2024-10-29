# Graduation Project: DRL Approach for Designing the Optimal Precoder Vector and Power Allocation in MU-MISO (SNR as Input Version and Power Allocation)

## Overview
This version of the project implements a Deep Reinforcement Learning (DRL) approach to design optimal precoder vectors and allocate power in Multi-User Multiple Input Single Output (MU-MISO) systems, incorporating Signal-to-Noise Ratio (SNR) as an input parameter.

The inclusion of SNR allows the model to adapt the precoder selection based on varying channel conditions, leading to improved performance in diverse environments. The size of the input channel is increased by one to accommodate this additional information.

## Key Features
- **Dynamic Precoding**: The precoder vector selection is influenced by both the SNR and the channel model, enabling the system to optimize performance under different conditions.
- **Adaptability**: The model is designed to adapt to varying SNR levels, making it versatile for real-world applications.
- **Improved Performance**: By incorporating SNR as an input, the model aims to outperform traditional precoder methods.

## Project Structure
- **buffer_numpy.py**: Contains functions for buffering and managing data.
- **DDPG_MultiUser.py**: Includes the agent class that implements the Deep Deterministic Policy Gradient algorithm.
- **main.ipynb**: The main Jupyter notebook containing the implementation of the DRL approach for optimal precoder design with SNR as input.
- **networks_MultiUser.py**: Defines the neural network architecture used for the agent.
- **utils.py**: Contains utility functions for various operations in the project.

## Evaluation of the Agent

### Numerical Evaluation
The agent's performance is quantitatively evaluated using a set of metrics calculated over multiple instances. The evaluation includes two scenarios: **High SNR** and **Low SNR**.

#### High SNR Evaluation
- **Standard Deviation of Channel Noise**: `snr_user = 5.0`
- **Steps**:
  1. Generate a channel matrix without the PAE effect.
  2. Apply the PAE effect to the channel matrix.
  3. Compute precoder matrices for MRT, ZF, and MMSE.
  4. Calculate the rewards for each precoder.
  5. Evaluate the agent’s actions and calculate the corresponding rewards.

- **Average Rewards Calculated**:
  - Average reward by Actor: `[1.8077662  2.5782685  0.32928637 0.13635688]`
  - Average reward by MRT: `[1.4897913 1.4548547 1.4428203 1.4338859]`
  - Average reward by ZF: `[1.9929861 1.9929857 1.9929857 1.9929857]`
  - Average reward by MMSE: `[2.160452  2.1545606 2.1523845 2.1511772]`

- **Percentage Performance**:
  - Percentage subject to MRT: `83.34280369661167 %`
  - Percentage subject to ZF: `60.85941835143447 %`
  - Percentage subject to MMSE: `56.29328345079189 %`

#### Low SNR Evaluation
- **Standard Deviation of Channel Noise**: `snr_user = 0.01`
- The same evaluation steps are followed as in the High SNR scenario to ensure comparability.

- **Average Rewards Calculated**:
  - Average reward by Actor: `[0.02731172 0.04411576 0.00358495 0.00134194]`
  - Average reward by MRT: `[0.02339773 0.02429889 0.02290916 0.02424937]`
  - Average reward by ZF: `[0.00890983 0.00890983 0.00890983 0.00890983]`
  - Average reward by MMSE: `[0.02320887 0.02402772 0.02272518 0.02398323]`

- **Percentage Performance**:
  - Percentage subject to MRT: `80.49573792598257 %`
  - Percentage subject to ZF: `214.2418306528997 %`
  - Percentage subject to MMSE: `81.27559464271948 %`

#### Single Instance Evaluation
A single channel state is generated using a fixed seed for reproducibility. The following rewards are calculated:
- Reward by Actor: `[0.01973256 0.08081265 0.00160834 0.0007194]`
- Reward by MRT: `[0.01952235 0.05102535 0.00830825 0.0231408]`
- Reward by ZF: `[0.01045827 0.01045827 0.01045827 0.01045827]`
- Reward by MMSE: `[0.01603819 0.01450478 0.01131923 0.01225454]`

- **Percentage Performance**:
  - Percentage subject to MRT: `100.85905746706793 %`
  - Percentage subject to ZF: `245.91281863543722 %`
  - Percentage subject to MMSE: `190.09449603434413 %`


> **Important Note**: But this result gets better and better; the reward curve has a high slope, but it needs a lot of time with a high computational machine.

### Performance Metrics
The performance of the agent is compared against traditional techniques (MRT, ZF, MMSE) in terms of average rewards and percentage performance.

### Conclusion
The evaluation metrics and plots generated provide insights into the effectiveness of the DRL-based approach for precoder design in MU-MISO systems. This approach is compared with established techniques to highlight its advantages and performance.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Other required libraries as per the implementation

### Usage
Run the main script to initiate training and evaluation. Modify the parameters as needed to experiment with different scenarios.

### Acknowledgments
This project was developed as part of the graduation project. Special thanks to the instructors and peers for their support and guidance.
