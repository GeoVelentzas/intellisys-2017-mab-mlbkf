from bandits import MLB_KF
from environments import generate_binary_sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




#--------------------------------------------------------------------
# ENVIRONMENTS
#--------------------------------------------------------------------
# Create a sequence of arms that return a binary reward
sequence = generate_binary_sequence(
	block_size = 100, iterations = 10)
#--------------------------------------------------------------------




#--------------------------------------------------------------------
# AGENTS
#--------------------------------------------------------------------
# Choose your bandit agent
agent = MLB_KF(
	narms = 2,
	aq = 0.14,
	am = 1/15,
	al = 1/350,
	eta = 0.44,
	phi = 1.5,
	mu0 = 0.5,
	var0 = 0.5,
	var_ob = 0.01**2,
	var_tr = 0.01**2)
#--------------------------------------------------------------------




#--------------------------------------------------------------------
# SIMULATIONS - LOG - OBERVATION - ACTION - UPDATE
#--------------------------------------------------------------------
# Run the simulation
logs = []
for a_star in sequence:

	# Keep the agent's internal state for the logs
	logs.append(agent.state())

	# Choose an action
	a = agent.decide()

	# Observe the reward
	r = a_star==a

	# Update the agent
	agent.update(r)
#--------------------------------------------------------------------





#--------------------------------------------------------------------
# PROCESS AGENT'S INTERNAL STATE AT EACH ACTION FOR PLOTS
#--------------------------------------------------------------------
# Plot the results
df = pd.DataFrame(logs)

# Action taken by the agent - len:T
actions = df['action'].to_list()

# Means of agent's internal state - shape:(T, narms)
M = np.array(df['mu'].to_list())

# Variances of agent's internal state - shape:(T, narms)
V = np.array(df['var'].to_list())

# Action probabilities from agent's internal state - shape:(T, narms)
P = np.array(df['probs'].to_list())
#--------------------------------------------------------------------





#--------------------------------------------------------------------
# PLOT THE RESULTS
#--------------------------------------------------------------------
sns.set_theme(style="darkgrid", context="talk")
palette = sns.color_palette("dark", n_colors=max(P.shape[1], M.shape[1], 2))
plt.figure(figsize=(24, 14))

# Plot 1: Actions taken by the agent vs true sequence
plt.subplot(2, 2, 1)
plt.plot(actions, label="Actions Taken", color=palette[0], alpha=0.8, linewidth=3)
plt.plot(sequence, label="True Sequence", color=palette[1], alpha=0.8, linestyle='--', linewidth=3)
plt.title("Actions vs True Sequence", fontsize=18, fontweight='bold')
plt.xlabel("Timestep", fontsize=16)
plt.ylabel("Action", fontsize=16)
plt.legend(fontsize=14, loc='upper right')
plt.tick_params(axis='both', labelsize=14)

# Plot 2: Action probabilities (P)
plt.subplot(2, 2, 2)
for arm in range(P.shape[1]):
    plt.plot(P[:, arm], label=f"Arm {arm}", color=palette[arm], linewidth=2.5, linestyle='-')
plt.title("Action Probabilities", fontsize=18, fontweight='bold')
plt.xlabel("Timestep", fontsize=16)
plt.ylabel("Probability", fontsize=16)
plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True, fancybox=True)
plt.tick_params(axis='both', labelsize=14)

# Plot 3: Means of the agent's internal state (M)
plt.subplot(2, 2, 3)
for arm in range(M.shape[1]):
    plt.plot(M[:, arm], label=f"Arm {arm}", color=palette[arm], linewidth=2.5, linestyle='-')
plt.title("Agent's Internal State (Means)", fontsize=18, fontweight='bold')
plt.xlabel("Timestep", fontsize=16)
plt.ylabel("Mean Value", fontsize=16)
plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True, fancybox=True)
plt.tick_params(axis='both', labelsize=14)

# Plot 4: Logarithm of Variances of the agent's internal state (V)
plt.subplot(2, 2, 4)
for arm in range(V.shape[1]):
    plt.plot(np.log(V[:, arm]), label=f"Arm {arm}", color=palette[arm], linewidth=2.5, linestyle='-')
plt.title("Agent's Internal State (log of Variances)", fontsize=18, fontweight='bold')
plt.xlabel("Timestep", fontsize=16)
plt.ylabel("log(Variance)", fontsize=16)
plt.legend(fontsize=14, loc='upper right', frameon=True, shadow=True, fancybox=True)
plt.tick_params(axis='both', labelsize=14)

# Add spacing between subplots
plt.tight_layout()
plt.show()
#--------------------------------------------------------------------
