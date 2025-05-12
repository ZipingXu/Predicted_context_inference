from curses import raw
import scipy.stats as stats
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from utility import prob_clip, compute_coverage, weighted_theta_est, softmax

class CoverageTracker:
    def __init__(self, n_action, dim, env):
        self.n_action = n_action
        self.dim = dim
        self.env = env
        self.w_Vt = np.zeros((self.n_action, self.dim, self.dim))
        self.w_Vt_inv = np.zeros((self.n_action, self.dim, self.dim))
        self.w_bt = np.zeros((self.n_action, self.dim))

        self.w_theta_est = np.zeros((self.n_action, self.dim))
        
        l = 0.00001
        for a in range(self.n_action):
            self.w_Vt[a,:,:] = l * np.eye(self.dim)
            self.w_Vt_inv[a,:,:] = (1.0 / l) * np.eye(self.dim)  # Inverse of l * I is (1/l) * I

    def _update_statistics(self, x_tilde_t, at, rt, pi_t, t):
        """Update sufficient statistics based on observed data
        
        Args:
            x_tilde_t: Current predicted context, if args.mis is True, it is the true context
            at: Chosen action
            rt: Observed reward
            pi_t: Current policy probabilities
            t: Current timestep
        Returns:
            Updated Vt and bt
        """
        imp_weight = 1 / pi_t[at]
        if self.env.mis:
            self.w_Vt[at,:,:] += imp_weight * np.outer(x_tilde_t, x_tilde_t)
        else:
            self.w_Vt[at,:,:] += imp_weight * (np.outer(x_tilde_t, x_tilde_t) - self.env.sigma_e * np.eye(self.dim))
        self.w_bt[at,:] += imp_weight * x_tilde_t * rt
        
        # Update w_Vt_inv
    
        return self.w_Vt, self.w_bt, self.w_theta_est
    def get_coverage(self, policy_func, t):
        # compute this inverse only when needed
        for a in range(self.n_action):
            self.w_Vt_inv[a,:,:] = np.linalg.pinv(self.w_Vt[a,:,:])            
            self.w_theta_est[a,:] = self.w_Vt_inv[a,:,:] @ self.w_bt[a,:]
        if self.env.mis:
            theta_true = self.env.best_theta
        else:
                theta_true = self.env.theta
        var_est, avg_pa = self.estimate_variance(policy_func=policy_func) 
        var_est = var_est / (t+1)
        coverage = compute_coverage(cur_theta_est=self.w_theta_est[0, 0], 
                                            var_est=var_est[0, 0, 0], 
                                            theta_true=theta_true[0, 0], alpha=0.1)
        return coverage, avg_pa, var_est
    def estimate_variance(self, policy_func, n_true = 5000):
        ## estimate the asymptotic variance of the weighted estimator
        w_theta_est = self.w_theta_est.T
        d = w_theta_est.shape[0]
        n_action = w_theta_est.shape[1]

        x_list, x_tilde_list = self.env.generate_context(n_true)
        potential_reward_list, _ = self.env.generate_potential_rewards(x_list, x_tilde_list, noise = True)
            
        asy_var = np.zeros((n_action, d, d))
        avg_pa = np.zeros(n_action)
        if self.env.mis:
            Hat = np.zeros((n_action, d, d))
            for t in range(n_true):
                x_tilde_t = x_tilde_list[t, :]
                pa = policy_func(x_tilde_t)
                avg_pa += pa
                at = np.random.choice(range(n_action), p=pa)
                Hat[at, :, :] += (1/(pa[at]**2)) * np.outer(x_tilde_t, x_tilde_t) * (potential_reward_list[t, at] - np.dot(x_tilde_t, w_theta_est[:, at]))**2
            asy_var = Hat * self.env.sigma_s**(-2) / n_true
            avg_pa = avg_pa / n_true
        else:
            for t in range(n_true):
                x_tilde_t = x_tilde_list[t, :]
                x_t = x_list[t, :]
                for a in range(n_action):
                    pa = policy_func(x_tilde_t)[a]
                    avg_pa[a] += pa
                    ha_t = (np.outer(x_tilde_t, x_tilde_t) - self.env.sigma_e * np.eye(d)) @ w_theta_est[:, a] - np.outer(x_tilde_t, x_t) @ w_theta_est[:, a]
                    asy_var[a, :, :] += (1 / pa) * (np.outer(ha_t, ha_t) + (self.env.sigma_eta**2) * np.outer(x_tilde_t, x_tilde_t))
            # for a in range(n_action):
            #     asy_var[a, :, :] *= asy_var[a, :, :] / (pi_nd[a]**2)
            asy_var = asy_var * self.env.sigma_s**(-2) / n_true 
            avg_pa = avg_pa / n_true
        avg_pa = np.array(avg_pa)
        return asy_var, avg_pa

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
    avg_pa_list: np.ndarray = field(init=False)
    coverage_list: np.ndarray = field(init=False)
    var_est_list: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize arrays based on dimensions from input data."""
        T = self.x_list.shape[0]
        d = self.x_list.shape[1]
        n_action = self.potential_reward_list.shape[1]
        
        self.theta_est_list = np.zeros((n_action, T, d))
        self.at_list = np.zeros(T)
        self.pi_list = np.zeros((T, n_action))
        self.pi_list_test = np.zeros((T, n_action))
        self.avg_pa_list = np.zeros((T, n_action))
        self.coverage_list = np.zeros(T)
        self.var_est_list = np.zeros((T))
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
            "var_est_list": self.var_est_list,
            "policy_func": self.policy_func,
            "avg_pa_list": self.avg_pa_list
            
        }
    def log(self, dat, name, t):
        """Log additional data with a name.
        
        Args:
            dat: Data to log
            name: Name of the data to log
        """
        name = name + "_list"
        current_array = getattr(self, name)
        current_array[t] = dat
        
    
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
    def __init__(self, env, args=None):
        """Initialize a bandit model with an environment instance
        
        Args:
            env: Environment instance that provides context generation and reward computation
            args: Additional arguments for the bandit algorithm
        """
        self.env = env
        self.theta = env.theta
        self.sigma_eta = env.sigma_eta 
        self.sigma_e = env.sigma_e
        self.n_action = env.n_action
        self.dim = env.d
        self.args = args
        self.coverage_freq = args.coverage_freq
        
        self.x_list = env.x_list
        self.x_tilde_list = env.x_tilde_list
        self.potential_reward_list = env.potential_reward_list
        self.at_dag_list = env.at_dag_list

        self.p0 = args.p0
        
        self.avg_pa = [0, 0]

        # Algorithm hyperparameters
        self.l = args.l
        self.C = args.C
        self.gamma = args.gamma
        self.rho = args.rho
        
    def mean_reward(self, s, a):
        """Compute mean reward for state s and action a"""
        return self.env.mean_reward(s, a)

    def realized_reward(self, s, a):
        """Compute realized reward with noise for state s and action a"""
        return self.env.realized_reward(s, a)
        
    def _initialize_algorithm_state(self):
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
            self.Vt[a,:,:] = self.l * np.eye(d)
            self.Vt_inv[a,:,:] = (1.0 / self.l) * np.eye(d)  # Inverse of l * I is (1/l) * I
            self.w_Vt[a,:,:] = self.l * np.eye(d)
            self.w_Vt_inv[a,:,:] = (1.0 / self.l) * np.eye(d)  # Inverse of l * I is (1/l) * I
            
        return self.Vt, self.Vt_inv, self.bt, self.w_Vt, self.w_Vt_inv, self.w_bt
    
    def _compute_regret(self, x_t, pi_t):
        """Compute regret for the current timestep
        
        Args:
            x_t: Current context
            pi_t: Current policy probabilities
            
        Returns:
            Regret for the current timestep
        """
        return self.env.compute_regret(x_t, pi_t, self.p0)
    
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
    
    def _update_statistics(self, x_tilde_t, at, rt, pi_t, t):
        """Update sufficient statistics based on observed data
        
        Args:
            x_tilde_t: Current predicted context
            at: Chosen action
            rt: Observed reward
            pi_t: Current policy probabilities
            t: Current timestep
        Returns:
            Updated Vt and bt
        """
        # Update Vt_inv
        self.Vt[at,:,:] += np.outer(x_tilde_t, x_tilde_t)
        self.bt[at,:] += x_tilde_t * rt

        # x = x_tilde_t
        V_inv_x = self.Vt_inv[at,:,:] @ x_tilde_t
        denominator = 1 + x_tilde_t @ V_inv_x
        
        outer_update = np.outer(V_inv_x, V_inv_x) / denominator
        self.Vt_inv[at,:,:] = self.Vt_inv[at,:,:] - outer_update
        self.theta_est[at,:] = self.Vt_inv[at,:,:] @ self.bt[at,:]
        
        # Update w_Vt_inv
        # if self.args.env == "meb":
        imp_weight = 1 / pi_t[at]
        if t < 100: # use naive estimator for the first 100 timesteps
            imp_weight = 1/2
        # print(self.env.mis)
        if self.env.mis:
            self.w_Vt[at,:,:] += imp_weight * np.outer(x_tilde_t, x_tilde_t)
        else:
            self.w_Vt[at,:,:] += imp_weight * (np.outer(x_tilde_t, x_tilde_t) - self.sigma_e * np.eye(self.dim))
        self.w_bt[at,:] += imp_weight * x_tilde_t * rt
            
        self.w_Vt_inv[at,:,:] = np.linalg.pinv(self.w_Vt[at,:,:])
            
        self.w_theta_est[at,:] = self.w_Vt_inv[at,:,:] @ self.w_bt[at,:]

        
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
        C = self.C
        
        # Compute UCBs for each action
        UCB_list = np.zeros(self.n_action)
        for a in range(self.n_action):
            # Use theta estimates from current state
            theta_a_hat = self.theta_est[a,:]
            mu_a = np.dot(theta_a_hat, x_tilde_t)
            # Use inverse for confidence calculation
            sigma_a2 = x_tilde_t @ self.Vt_inv[a,:,:] @ x_tilde_t.reshape((self.dim, 1))
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
        rho2 = self.rho ** 2
        
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
        gamma = self.gamma
        
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
            x_tilde_test: for evaluating the policy at a test context
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
            
        T = self.args.T
        d = self.dim

        self.coverage_tracker = CoverageTracker(self.n_action, self.dim, self.env)
        
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
        self.var_est = np.zeros((self.n_action, self.dim, self.dim))
        self.avg_pa = [0, 0]
        
        # Main algorithm loop
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]

            # there is no x_t if mis is true
            x_t = self.x_list[t,:] if not self.args.mis else x_tilde_t
            
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
            self._update_statistics(x_tilde_t, at, rt, pi_t, t)
            self.coverage_tracker._update_statistics(x_tilde_t, at, rt, pi_t, t)
            
            # Update logger with this timestep
            if policy.lower() in ['meb', 'random']:
                logger.update_step(t, self.w_theta_est, at, pi_t, pi_test)
            else:
                logger.update_step(t, self.theta_est, at, pi_t, pi_test)
            
            # Compute coverage if needed
            if (t+1) % self.coverage_freq == 0:
                self.coverage, self.avg_pa, self.var_est = self.coverage_tracker.get_coverage(policy_func, t)
                # print(np.linalg.norm(self.env.best_theta[:, 0] - self.w_theta_est[0, :]))
                # else:
                    # self.coverage, self.avg_pa, self.var_est = self.coverage_tracker.get_coverage(policy_func, t, self.env.theta)
            logger.log(self.coverage, "coverage", t)
            logger.log(self.avg_pa, "avg_pa", t)
            logger.log(self.var_est[0, 0, 0], "var_est", t)
            
        logger.policy_func = policy_func
        return logger.to_dict()