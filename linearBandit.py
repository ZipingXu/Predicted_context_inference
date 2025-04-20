from curses import raw
import scipy.stats as stats
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from utility import prob_clip, compute_coverage, weighted_theta_est, estimate_variance, softmax


@dataclass
class BanditExperimentLogger:
    """Data class for managing and tracking bandit experiment results."""
    x_list: np.ndarray
    x_tilde_list: np.ndarray
    potential_reward_list: np.ndarray
    at_dag_list: Optional[np.ndarray] = None
    
    # Initialize tracking variables - will be populated during experiment
    theta_est_list: np.ndarray = field(init=False)
    at_list: np.ndarray = field(init=False)
    pi_list: np.ndarray = field(init=False)
    pi_list_test: np.ndarray = field(init=False)
    coverage_list: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize arrays based on dimensions from input data."""
        T = self.x_list.shape[0]
        d = self.x_list.shape[1]
        n_action = self.potential_reward_list.shape[1]
        
        self.theta_est_list = np.zeros((n_action, T, d))
        self.at_list = np.zeros(T)
        self.pi_list = np.zeros((T, n_action))
        self.pi_list_test = np.zeros((T, n_action))
        self.policy_func = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the experiment results to a dictionary."""
        return {
            "x_list": self.x_list,
            "x_tilde_list": self.x_tilde_list,
            "potential_reward_list": self.potential_reward_list,
            "at_dag_list": self.at_dag_list,
            "theta_est_list": self.theta_est_list,
            "at_list": self.at_list,
            "pi_list": self.pi_list,
            "pi_list_test": self.pi_list_test,
            "coverage_list": self.coverage_list,
            "policy_func": self.policy_func
        }
    
    def update_step(self, t: int, theta_est: np.ndarray, at: int, 
                    pi_t: np.ndarray, pi_test: Optional[np.ndarray] = None):
        """Update logger with data from a single timestep.
        
        Args:
            t: Current timestep
            theta_est: Parameter estimates for each action
            at: Action taken
            pi_t: Policy probabilities
            pi_test: Policy probabilities for test context (optional)
        """
        n_action = pi_t.shape[0]
        for a in range(n_action):
            self.theta_est_list[a, t, :] = theta_est[a, :]
        
        self.at_list[t] = at
        self.pi_list[t, :] = pi_t
        
        if pi_test is not None:
            self.pi_list_test[t, :] = pi_test
    
    def add_coverage(self, coverage: float):
        """Add a coverage statistic to the list."""
        self.coverage_list.append(coverage)


class LinBandit:
    def __init__(self, theta=None, sigma=None, n_action=None, args=None):
        """Initialize a bandit model with parameters theta and phi
        r = <theta_a, s> + eta, eta~N(0, sigma^2)
        
        Args:
            theta (array): Each column is a theta_a value
            sigma (float): Standard deviation of noise
            n_action (int): Number of actions
        """
        self.theta = theta
        self.sigma = sigma 
        self.n_action = n_action if theta is None else theta.shape[1]
        self.dim = None if theta is None else theta.shape[0]
        self.args = args
        self.coverage_freq = args.coverage_freq
        
        # These will be set by generate_potential_reward_w_xtilde
        self.x_list = None
        self.x_tilde_list = None
        self.potential_reward_list = None
        self.at_dag_list = None
        self.p0 = args.p0
        self.sigma_e = args.sigma_e

    def mean_reward(self, s, a):
        """Compute mean reward for state s and action a"""
        if (self.theta is None) or (a > self.theta.shape[1]-1):
            return None
        return np.dot(self.theta[:, a], s)

    def realized_reward(self, s, a):
        """Compute realized reward with noise for state s and action a"""
        mu = self.mean_reward(s, a)
        if mu is None:
            return None
        return mu + np.random.normal(0, self.sigma)

    def _generate_x_tilde(self, x, Sigma_e, dist_ops=0., rho=0.9):
        """Helper to generate noisy context observation"""
        d = self.dim
        if dist_ops == 0:
            return x + np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma_e)
        elif dist_ops == -1:
            return x + 2 * rho * np.random.binomial(1, 0.5) - rho
        else:
            nu = dist_ops
            Y = np.random.multivariate_normal(mean=np.zeros(d), cov=(nu-2)/nu*Sigma_e)
            U = np.random.chisquare(df=nu)
            return x + np.sqrt(nu / U) * Y

    def generate_potential_reward_w_xtilde(self, x_list, x_tilde_list):
        """Generate potential reward history with provided contexts
        
        Args:
            x_list: True context list (T x d)
            x_tilde_list: Predicted context list (T x d)
            
        Returns:
            Dictionary containing history
        """
        if self.theta is None:
            return None
            
        T = x_list.shape[0]
        d = self.dim
        n_action = self.n_action
        
        potential_reward_list = np.zeros((T, n_action))
        at_dag_list = np.zeros(T)

        for t in range(T):
            x_tilde_t = x_tilde_list[t,:]
            
            for a in range(n_action):
                potential_reward_list[t,a] = self.realized_reward(x_list[t,:], a)
                
            mean_reward_dag = np.matmul(x_tilde_t.reshape((1,d)), self.theta).reshape(n_action)
            at_dag_list[t] = np.argmax(mean_reward_dag)

        # Store as instance variables
        self.x_list = x_list
        self.x_tilde_list = x_tilde_list
        self.potential_reward_list = potential_reward_list
        self.at_dag_list = at_dag_list

        return {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list
        }
        
    def _initialize_algorithm_state(self, l=1.):
        """Initialize common algorithm state for ridge regression-based methods
        
        Args:
            l: Ridge regression regularization parameter
            
        Returns:
            Initialized Vt, Vt_inv, and bt arrays
        """
        d = self.dim
        self.Vt = np.zeros((self.n_action, d, d))
        self.Vt_inv = np.zeros((self.n_action, d, d))
        self.w_Vt = np.zeros((self.n_action, d, d))
        self.w_Vt_inv = np.zeros((self.n_action, d, d))
        self.bt = np.zeros((self.n_action, d))
        self.w_bt = np.zeros((self.n_action, d))

        self.theta_est = np.zeros((self.n_action, d))
        self.w_theta_est = np.zeros((self.n_action, d))
        
        for a in range(self.n_action):
            self.Vt[a,:,:] = l * np.eye(d)
            self.Vt_inv[a,:,:] = (1.0 / l) * np.eye(d)  # Inverse of l * I is (1/l) * I
            self.w_Vt[a,:,:] = l * np.eye(d)
            self.w_Vt_inv[a,:,:] = (1.0 / l) * np.eye(d)  # Inverse of l * I is (1/l) * I
            
        return self.Vt, self.Vt_inv, self.bt, self.w_Vt, self.w_Vt_inv, self.w_bt
    
    def _compute_regret(self, x_t, pi_t):
        """Compute regret for the current timestep
        
        Args:
            x_t: Current context
            pi_t: Current policy probabilities
            p_0: Minimum selection probability
            
        Returns:
            Regret for the current timestep
        """
        if np.dot(x_t, self.theta[:,0]) > np.dot(x_t, self.theta[:,1]):
            reward_oracle_t = np.dot(x_t, self.theta[:,0]) * (1-self.p0) + np.dot(x_t, self.theta[:,1]) * self.p0
        else:
            reward_oracle_t = np.dot(x_t, self.theta[:,1]) * (1-self.p0) + np.dot(x_t, self.theta[:,0]) * self.p0
            
        reward_policy = np.dot(x_t, self.theta[:,0]) * pi_t[0] + np.dot(x_t, self.theta[:,1]) * pi_t[1]
        return reward_oracle_t - reward_policy
    
    def _take_action_and_observe(self, pi_t, potential_rewards_t):
        """Take action according to policy and observe reward
        
        Args:
            pi_t: Current policy probabilities
            potential_rewards_t: Potential rewards for all actions
            
        Returns:
            Tuple of (action, reward)
        """
        if len(pi_t) != self.n_action:
            raise ValueError("pi_t must have length equal to n_action")
    
        at = np.random.choice(self.n_action, p=pi_t)
        rt = potential_rewards_t[at]
        return at, rt
    
    def _update_statistics(self, x_tilde_t, at, rt, pi_t):
        """Update sufficient statistics based on observed data
        
        Args:
            x_tilde_t: Current predicted context
            at: Chosen action
            rt: Observed reward
            pi_t: Current policy probabilities
        Returns:
            Updated Vt and bt
        """
        imp_weight = (1/self.n_action) / pi_t[at]

        self.Vt[at,:,:] += np.outer(x_tilde_t, x_tilde_t)
        self.bt[at,:] += x_tilde_t * rt
        self.w_Vt[at,:,:] += imp_weight * (np.outer(x_tilde_t, x_tilde_t) - self.sigma_e * np.eye(self.dim))
        self.w_bt[at,:] += imp_weight * x_tilde_t * rt

        # Update Vt_inv
        x = x_tilde_t
        V_inv_x = self.Vt_inv[at,:,:] @ x
        denominator = 1 + x @ V_inv_x
        
        
        outer_update = np.outer(V_inv_x, V_inv_x) / denominator
        self.Vt_inv[at,:,:] = self.Vt_inv[at,:,:] - outer_update
        
        # Update w_Vt_inv
        
        self.w_Vt_inv[at,:,:] = np.linalg.pinv(self.w_Vt[at,:,:])
        
        self.w_theta_est[at,:] = self.w_Vt_inv[at,:,:] @ self.w_bt[at,:]
        self.theta_est[at,:] = self.Vt_inv[at,:,:] @ self.bt[at,:]
        return self.Vt, self.bt, self.w_Vt, self.w_bt, self.theta_est, self.w_theta_est

    def random_policy(self, x_tilde_t):
        """Random policy that returns uniform probabilities
        
        Args:
            x_tilde_t: Current predicted context
            
        Returns:
            Policy probabilities (uniform)
        """
        return np.ones(self.n_action) / self.n_action
    
    def ucb_policy(self, x_tilde_t):
        """UCB policy with predicted states
        
        Args:
            x_tilde_t: Current predicted context
            
        Returns:
            Policy probabilities based on UCB scores
        """
        # Get parameters from args
        C = 1.0
        
        # Compute UCBs for each action
        UCB_list = np.zeros(self.n_action)
        for a in range(self.n_action):
            # Use theta estimates from current state
            theta_a_hat = self.theta_est[a,:]
            mu_a = np.dot(theta_a_hat, x_tilde_t)
            # Use inverse for confidence calculation
            sigma_a2 = np.dot(x_tilde_t, np.matmul(self.Vt_inv[a,:,:], x_tilde_t))
            UCB_list[a] = mu_a + C * np.sqrt(sigma_a2)
                
        at_ucb = np.argmax(UCB_list)
        
        # Set policy probabilities
        pi_t = np.ones(self.n_action) * self.p0
        pi_t[at_ucb] = 1 - self.p0 * (self.n_action-1)
        
        return pi_t
    
    def ts_policy(self, x_tilde_t):
        """Thompson sampling policy with predicted states
        
        Args:
            x_tilde_t: Current predicted context
            
        Returns:
            Policy probabilities based on Thompson sampling
        """
        # Get parameters from args
        rho2 = self.args.sigma_eta ** 2 if hasattr(self.args, 'sigma_eta') else 1.0
        
        # Compute posterior for each action
        post_mean_t = np.zeros((self.n_action, self.dim))
        post_var_t = np.zeros((self.n_action, self.dim, self.dim))
        
        for a in range(self.n_action):
            # Use theta estimates from current state
            post_mean_t[a,:] = self.theta_est[a,:]
            # Use inverse for variance calculation
            post_var_t[a,:,:] = rho2 * self.Vt_inv[a,:,:]
        
        # Compute selection probabilities
        post_mean_w_x_tilde = np.dot(x_tilde_t, post_mean_t[0,:] - post_mean_t[1,:])
        post_var_w_x_tilde = np.dot(x_tilde_t, np.matmul(post_var_t[0,:,:] + post_var_t[1,:,:], x_tilde_t)) + 0.00001
        prob_0 = 1 - stats.norm.cdf(-post_mean_w_x_tilde / np.sqrt(post_var_w_x_tilde))
        pi_0 = prob_clip(prob_0, self.p0)
        pi_t = np.array([pi_0, 1 - pi_0])
        
        return pi_t
    
    def boltzmann_policy(self, x_tilde_t):
        """Boltzmann policy with predicted states
        
        Args:
            x_tilde_t: Current predicted context
            
        Returns:
            Policy probabilities based on Boltzmann exploration
        """
        # Get parameters from args
        gamma = self.args.gamma if hasattr(self.args, 'gamma') else 0
        
        # Compute values for each action
        est_mean_list = np.zeros(self.n_action)
        
        for a in range(self.n_action):
            # Use theta estimates from current state
            est_mean_list[a] = np.dot(self.theta_est[a,:], x_tilde_t)
                
        # Apply Boltzmann policy
        pi_t = softmax(est_mean_list, gamma)
        
        return pi_t
    
    def meb_policy(self, x_tilde_t):
        """MEB policy that takes into account measurement error
        
        Args:
            x_tilde_t: Current predicted context
            
        Returns:
            Policy probabilities based on MEB algorithm
        """
            
        # Find the best action based on the current MEB theta estimate
        at_meb = np.argmax(np.matmul(self.w_theta_est, x_tilde_t))
        
        # Set policy probabilities
        pi_t = np.ones(self.n_action) * self.p0
        pi_t[at_meb] = 1 - self.p0 * (self.n_action-1)
        
        return pi_t
        
    def run_bandit(self, policy='random', x_tilde_test=None, **kwargs):
        """Template function to run bandit algorithms
        
        Args:
            policy: Choice of algorithm policy ('random', 'ucb', 'ts', 'boltzmann', 'meb')
            x_tilde_test: Test context for policy evaluation
            **kwargs: Additional parameters specific to the chosen algorithm
            
        Returns:
            Dictionary containing history and results
        """
        # Set up policy function based on the choice
        if policy.lower() == 'random':
            policy_func = self.random_policy
        elif policy.lower() == 'ucb':
            policy_func = self.ucb_policy
        elif policy.lower() == 'ts':
            policy_func = self.ts_policy
        elif policy.lower() == 'boltzmann':
            policy_func = self.boltzmann_policy
        elif policy.lower() == 'meb':
            policy_func = self.meb_policy
        else:
            raise ValueError(f"Unknown policy: {policy}. Choose from 'random', 'ucb', 'ts', 'boltzmann', 'meb'")

        if self.x_list is None or self.x_tilde_list is None or self.potential_reward_list is None:
            raise ValueError("Context data not initialized. Call generate_potential_reward_w_xtilde first.")
            
        T = self.x_list.shape[0]
        d = self.dim
        
        # Initialize tracking variables for regret
        regret_list = np.zeros(T)
        regret = 0
        
        # Initialize algorithm state - will be used by the policy functions
        self._initialize_algorithm_state()
        
        # Create experiment logger
        logger = BanditExperimentLogger(
            x_list=self.x_list,
            x_tilde_list=self.x_tilde_list, 
            potential_reward_list=self.potential_reward_list,
            at_dag_list=self.at_dag_list
        )
        self.coverage = 0
        # Main algorithm loop
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]
            x_t = self.x_list[t,:]
            
            # Get policy for current state
            pi_t = policy_func(x_tilde_t)
            
            # Get test policy if requested
            pi_test = policy_func(x_tilde_test) if x_tilde_test is not None else None
            
            # Compute regret
            regret_t = self._compute_regret(x_t, pi_t)
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at, rt = self._take_action_and_observe(pi_t, self.potential_reward_list[t,:])
            
            # Update statistics based on the action and reward
            if policy.lower() != 'random':
                # skip if random policy
                self._update_statistics(x_tilde_t, at, rt, pi_t)
            
            # Update logger with this timestep
            if policy.lower() != 'meb':
                logger.update_step(t, self.w_theta_est, at, pi_t, pi_test)
            else:
                logger.update_step(t, self.theta_est, at, pi_t, pi_test)
            
            # Compute coverage if needed
            if (t+1) % self.coverage_freq == 0:
                var_est = estimate_variance(policy_func=policy_func, w_theta_est=self.w_theta_est.T, 
                                            args=self.args) / (t+1)
                self.coverage = compute_coverage(cur_theta_est=self.w_theta_est[0, 0], 
                                            var_est=var_est[0, 0, 0], 
                                            theta_true=self.theta[0, 0])
            logger.add_coverage(self.coverage)
        logger.policy_func = policy_func
        return logger.to_dict()