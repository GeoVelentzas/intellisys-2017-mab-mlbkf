import numpy as np

class MLB_KF:
    """
    Multi-Armed Bandit agent using a Kalman Filter for decision-making and learning.

    Parameters:
    -----------
    narms : int
        Number of arms in the bandit problem.
    aq : float
        Learning rate parameter for updating the mean of rewards.
    am : float
        Smoothing parameter for the estimated average reward (rbar).
    al : float
        Smoothing parameter for the smoothed average reward (rbbar).
    eta : float
        Step size for updating the inverse temperature (beta).
    phi : float
        Exploration factor, controlling the contribution of uncertainty to action value.
    mu0 : float
        Initial mean estimate for each arm.
    var0 : float
        Initial variance estimate for each arm.
    var_tr : float
        Transition variance (uncertainty added per time step).
    var_ob : float
        Observation variance (uncertainty in reward observations).
    """

    def __init__(self, narms, aq, am, al, eta, phi, mu0, var0, var_tr, var_ob):
        # Number of arms in the bandit
        self.narms = narms
        
        # Learning rate for action-value updates
        self.aq = aq
        
        # Smoothing parameters for rewards
        self.am = am  # Weight for updating rbar (average reward)
        self.al = al  # Weight for updating rbbar (smoothed rbar)
        
        # Step size for inverse temperature (beta)
        self.eta = eta
        
        # Exploration factor for action-value computation
        self.phi = phi
        
        # Transition and observation variances
        self.var_tr = var_tr  # Transition variance
        self.var_ob = var_ob  # Observation variance
        
        # Initial beliefs about each arm's reward distribution
        self.mu = mu0 * np.ones(narms)      # Initial mean for each arm
        self.var = var0 * np.ones(narms)    # Initial variance for each arm
        
        # Exploration-exploitation parameter (inverse temperature)
        self.beta = 0
        
        # Reward averages
        self.rbar = 0   # Average reward
        self.rbbar = 0  # Smoothed average reward
        
        # Agent's decision state
        self.action_index = 0                # Last chosen action
        self.q = np.zeros(narms)             # Action values (means + uncertainty)
        self.probs = np.ones(narms) / narms  # Action probabilities

    def decide(self):
        """
        Select an action based on the current state of the agent.
        Uses a softmax policy on the action values (q) scaled by beta.
        """
        # Compute action values (means + exploration factor scaled by variance)
        self.q = self.mu + self.phi * (self.var**0.5)
        
        # Compute logits for the softmax
        logits = self.beta * self.q
        logits = logits - np.max(logits)  # Numerical stability trick
        
        # Compute action probabilities
        self.probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Select an action based on probabilities
        self.action_index = np.random.choice(np.arange(self.narms), p=self.probs)
        return self.action_index

    def update(self, r):
        """
        Update the agent's internal state based on the observed reward.

        Parameters:
        -----------
        r : float
            Reward observed after taking the chosen action.
        """
        # Update reward averages
        self.rbar = self.am * r + (1 - self.am) * self.rbar
        self.rbbar = self.al * self.rbar + (1 - self.al) * self.rbbar
        
        # Update the mean and variance for the selected arm
        all_indices = np.arange(self.narms)
        a = self.action_index              # Chosen action
        a_not = np.delete(all_indices, a)  # Other arms
        
        # Kalman filter update for the chosen arm
        self.mu[a] = ((self.var[a] + self.var_tr) * r + self.var_ob * self.mu[a]) / (
            self.var[a] + self.var_tr + self.var_ob
        )
        self.var[a] = ((self.var[a] + self.var_tr) * self.var_ob) / (
            self.var[a] + self.var_tr + self.var_ob
        )
        
        # Increase uncertainty for unchosen arms
        self.var[a_not] = self.var[a_not] + self.var_tr
        
        # Update beta (inverse temperature)
        self.beta = max(self.beta + self.eta * (self.rbar - self.rbbar), 0)

    def reset(self):
        """
        Reset the agent's internal state to its initial values.
        """
        self.mu = mu0 * np.ones(self.narms)
        self.var = var0 * np.ones(self.narms)
        self.q = np.zeros(self.narms)
        self.rbar = 0
        self.rbbar = 0
        self.beta = 0
        self.action_index = 0

    def state(self):
        """
        Return the agent's current internal state.

        Returns:
        --------
        observations : dict
            A dictionary containing:
            - 'mu': List of current mean estimates for each arm.
            - 'var': List of current variance estimates for each arm.
            - 'action': Last action taken by the agent.
            - 'probs': List of action probabilities for each arm.
        """
        observations = {
            'mu': list(self.mu),
            'var': list(self.var),
            'action': self.action_index,
            'probs': list(self.probs)
        }
        return observations
