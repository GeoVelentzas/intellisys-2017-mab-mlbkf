# Multi-Armed Bandit Simulation with Kalman Filter

This repository contains Python code for simulating a **Multi-Armed Bandit (MAB)** problem using a **Kalman Filter (KF)**-based agent. The code demonstrates how an intelligent agent learns to balance exploration and exploitation to maximize rewards in a binary reward environment.

## Files

- `bandits.py`: Contains the implementation of the `MLB_KF` class, a Kalman Filter-based agent for solving the MAB problem.
- `environments.py`: Defines the environment, including a function to generate binary reward sequences (`generate_binary_sequence`).
- `simulate.py`: Runs the simulation, logs the agent's internal states, and generates plots to visualize the agent's learning process.

## How to Run

To execute the simulation, simply run:
`python simulate.py`

## Outputs

The `simulate.py` script runs a simulation and generates the following plots:

1. **Actions Taken vs True Sequence**:
   - Visualizes how well the agent's actions align with the true reward sequence.

2. **Action Probabilities**:
   - Shows the probabilities of selecting each arm over time.

3. **Means of Agent's Internal State**:
   - Plots the evolving mean beliefs (`mu`) for each arm.

4. **Logarithm of Variances of the Agent's Internal State**:
   - Visualizes the uncertainty in the agent's beliefs (`log(variance)`).

## Key Components

### `MLB_KF` Class (in `bandits.py`)
A Kalman Filter-based agent designed to solve the MAB problem. Key features include:
- **Softmax-based Action Selection**: Balances exploration and exploitation by scaling action values (`q`) with an inverse temperature (`beta`).
- **Kalman Filter Updates**: Adjusts the agent's beliefs (`mu` and `variance`) based on observed rewards.
- **Learning Parameters**:
  - `eta`: Step size for `beta` updates.
  - `phi`: Exploration parameter for incorporating uncertainty into action values.
  - `am`, `al`: Smoothing parameters for reward averages.

### `generate_binary_sequence` Function (in `environments.py`)
Generates a binary reward sequence where blocks of zeros and ones alternate, simulating a two-armed bandit environment.

### Simulation Script (in `simulate.py`)
- Initializes the environment and agent.
- Runs the agent through a sequence of actions, logging its state at each timestep.
- Generates plots to analyze the agent's performance and learning process.

## Example Usage

In `simulate.py`, the simulation is configured as follows:
- **Environment**: A binary reward sequence with `block_size=100` and `iterations=10`.
- **Agent**: Configured with 2 arms and parameters fine-tuned for learning in the binary environment.

After running the simulation, the script generates plots showing the agent's decision-making process and learning progress.

## Customization

### Modify the Environment
You can change the reward structure by editing the parameters of `generate_binary_sequence` in `simulate.py`:
```python
sequence = generate_binary_sequence(block_size=50, iterations=20)
```


### Adjust Agent Parameters
Tune the agent's learning parameters in `simulate.py`:
```
agent = MLB_KF(
    narms=2,
    aq=0.1,
    am=1/10,
    al=1/300,
    eta=0.5,
    phi=1.2,
    mu0=0.0,
    var0=1.0,
    var_ob=0.01**2,
    var_tr=0.01**2
)
```