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
            "coverage_list": self.coverage_list
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


class weighted_theta_est:
    def __init__(self, args):
        self.args = args
        self.pi_nd = args.pi_nd
        self.sigma_e = args.sigma_e
        self.n_action = args.n_action
        self.d = args.d
        self.initialize()
    def initialize(self):
        self.V_t = np.zeros((self.n_action, self.d, self.d))
        self.V_t_inv = np.zeros((self.n_action, self.d, self.d))
        self.W_t = np.zeros((self.n_action, self.d, self.d))
        self.b_t = np.zeros((self.n_action, self.d))
        
        # Initialize V_t and V_t_inv with a small identity matrix for numerical stability
        for a in range(self.n_action):
            # Use a small ridge term for numerical stability
            self.V_t[a,:,:] = 1e-6 * np.eye(self.d)
            self.V_t_inv[a,:,:] = 1e6 * np.eye(self.d)  # Inverse of 1e-6 * I is 1e6 * I
        
    def update(self, x_tilde_t, a_t, r_t, pi_t):
        imp_weight_at = self.pi_nd[a_t] / pi_t[a_t]
        for a in range(self.n_action):
            if a == a_t:
                # Calculate the rank-1 update term
                update_term = imp_weight_at * (np.outer(x_tilde_t, x_tilde_t) - self.sigma_e * np.eye(self.d))
                
                # Update V_t
                self.V_t[a,:,:] += update_term
                
                # Update V_t_inv using Sherman-Morrison formula
                self.update_V_t_inv(a, x_tilde_t, imp_weight_at)
                
                # Update b_t
                self.b_t[a,:] += imp_weight_at * x_tilde_t * r_t
                
    def update_V_t_inv(self, action, x_tilde_t, imp_weight):
        """Update the inverse of V_t using Sherman-Morrison formula for the specified action.
        
        The Sherman-Morrison formula states that if A^(-1) is known, then 
        (A + u*v^T)^(-1) = A^(-1) - (A^(-1)*u*v^T*A^(-1))/(1 + v^T*A^(-1)*u)
        
        Args:
            action: The action for which to update the inverse
            x_tilde_t: The context vector x_tilde
            imp_weight: The importance weight
        """
        # For numerical stability, if the matrix is close to singular, recompute the inverse directly
        if np.linalg.det(self.V_t[action,:,:]) < 1e-10:
            self.V_t_inv[action,:,:] = np.linalg.pinv(self.V_t[action,:,:])
            return
            
        # We need to update V_inv for the combined update:
        # V += imp_weight * (x_tilde * x_tilde^T - sigma_e * I)
        
        # Get current inverse and setup variables
        V_inv = self.V_t_inv[action,:,:]
        x = x_tilde_t
        
        # Step 1: Apply Sherman-Morrison for the outer product update: V + imp_weight * (x_tilde * x_tilde^T)
        V_inv_x = V_inv @ x
        denominator = 1 + imp_weight * (x @ V_inv_x)
        
        # Update for the outer product term
        if abs(denominator) > 1e-10:  # Check to avoid division by near-zero
            outer_update = imp_weight * np.outer(V_inv_x, V_inv_x) / denominator
            V_inv = V_inv - outer_update
        
        # Step 2: Handle the subtraction of the sigma_e term: V - imp_weight * sigma_e * I
        # For this update, we'll use direct computation since Sherman-Morrison
        # is for rank-1 updates and this is a full-rank update
        if self.sigma_e > 0:
            # The update V = V - imp_weight * sigma_e * I is difficult to apply incrementally
            # For small updates, we can approximate; for larger ones, recompute directly
            sigma_e_scaled = imp_weight * self.sigma_e
            
            if sigma_e_scaled / np.linalg.norm(self.V_t[action,:,:], 'fro') < 0.01:
                # For small updates relative to V, use approximation
                # (A - εI)^(-1) ≈ A^(-1) + εA^(-2) + ε²A^(-3) + ...
                # We use just the first-order approximation
                V_inv = V_inv + sigma_e_scaled * (V_inv @ V_inv)
            else:
                # For larger updates, recompute directly as it's more stable
                self.V_t_inv[action,:,:] = np.linalg.pinv(self.V_t[action,:,:])
                return
        
        # Store the updated inverse
        self.V_t_inv[action,:,:] = V_inv
            
    def get_theta_est(self):
        theta_st = np.zeros((self.n_action, self.d))
        for a in range(self.n_action):
            # Use the maintained inverse instead of recomputing it
            theta_st[a,:] = np.matmul(self.V_t_inv[a,:,:], self.b_t[a,:])
        return theta_st


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
        Vt = np.zeros((self.n_action, d, d))
        Vt_inv = np.zeros((self.n_action, d, d))
        bt = np.zeros((self.n_action, d))
        
        for a in range(self.n_action):
            Vt[a,:,:] = l * np.eye(d)
            Vt_inv[a,:,:] = (1.0 / l) * np.eye(d)  # Inverse of l * I is (1/l) * I
            
        return Vt, Vt_inv, bt
    
    def _compute_regret(self, x_t, pi_t, p_0=0.2):
        """Compute regret for the current timestep
        
        Args:
            x_t: Current context
            pi_t: Current policy probabilities
            p_0: Minimum selection probability
            
        Returns:
            Regret for the current timestep
        """
        if np.dot(x_t, self.theta[:,0]) > np.dot(x_t, self.theta[:,1]):
            reward_oracle_t = np.dot(x_t, self.theta[:,0]) * (1-p_0) + np.dot(x_t, self.theta[:,1]) * p_0
        else:
            reward_oracle_t = np.dot(x_t, self.theta[:,1]) * (1-p_0) + np.dot(x_t, self.theta[:,0]) * p_0
            
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
        at = np.random.binomial(1, pi_t[1])
        rt = potential_rewards_t[at]
        return at, rt
    
    def _update_statistics(self, Vt, bt, x_tilde_t, at, rt):
        """Update sufficient statistics based on observed data
        
        Args:
            Vt: Current V matrix
            bt: Current b vector
            x_tilde_t: Current predicted context
            at: Chosen action
            rt: Observed reward
            
        Returns:
            Updated Vt and bt
        """
        Vt[at,:,:] += np.outer(x_tilde_t, x_tilde_t)
        bt[at,:] += x_tilde_t * rt
        return Vt, bt
        
    def _update_statistics_incremental(self, Vt, Vt_inv, bt, x_tilde_t, at, rt):
        """Update sufficient statistics and their inverses using Sherman-Morrison formula
        
        Args:
            Vt: Current V matrix
            Vt_inv: Current inverse of V matrix
            bt: Current b vector
            x_tilde_t: Current predicted context
            at: Chosen action
            rt: Observed reward
            
        Returns:
            Updated Vt, Vt_inv, and bt
        """
        # Update Vt with the outer product
        Vt[at,:,:] += np.outer(x_tilde_t, x_tilde_t)
        
        # Update Vt_inv using Sherman-Morrison formula
        # (A + u*v^T)^(-1) = A^(-1) - (A^(-1)*u*v^T*A^(-1))/(1 + v^T*A^(-1)*u)
        # Here u = v = x_tilde_t
        V_inv = Vt_inv[at,:,:]
        x = x_tilde_t
        
        V_inv_x = V_inv @ x
        denominator = 1 + x @ V_inv_x
        
        if abs(denominator) > 1e-10:  # Check to avoid division by near-zero
            outer_update = np.outer(V_inv_x, V_inv_x) / denominator
            Vt_inv[at,:,:] = V_inv - outer_update
        else:
            # If numerically unstable, recompute the inverse directly
            Vt_inv[at,:,:] = np.linalg.pinv(Vt[at,:,:])
        
        # Update bt
        bt[at,:] += x_tilde_t * rt
        
        return Vt, Vt_inv, bt
    
    def random_policy(self, p_0=0.2, x_tilde_test=None):
        """Random policy"""
        if self.x_list is None or self.x_tilde_list is None or self.potential_reward_list is None:
            raise ValueError("Context data not initialized. Call generate_potential_reward_w_xtilde first.")
            
        T = self.x_list.shape[0]
        d = self.dim
        
        # Initialize tracking variables
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        regret = 0  # Initialize regret
        
        # Initialize weighted theta estimator for coverage calculation
        self.w_est_cal = weighted_theta_est(args=self.args)
        
        # Create experiment logger
        logger = BanditExperimentLogger(
            x_list=self.x_list,
            x_tilde_list=self.x_tilde_list, 
            potential_reward_list=self.potential_reward_list,
            at_dag_list=self.at_dag_list
        )
        
        # For random policy, we create a dummy theta_est that's all zeros
        # This allows the logger to work correctly without special handling
        theta_est_dummy = np.zeros((self.n_action, d))
        
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]
            x_t = self.x_list[t,:]
            
            # Set uniform random policy
            pi_t = np.ones(self.n_action) / self.n_action
            
            # Set test policy if requested
            pi_test = np.ones(self.n_action) / self.n_action if x_tilde_test is not None else None
            
            # Compute regret
            regret_t = self._compute_regret(x_t, pi_t, p_0)
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at = np.random.choice(self.n_action, p=pi_t)
            rt = self.potential_reward_list[t,at]
            
            # Update logger with this timestep
            logger.update_step(t, theta_est_dummy, at, pi_t, pi_test)
            
            # Update coverage calculation
            self.w_est_cal.update(x_tilde_t, at, rt, pi_t)
            
            # Compute coverage if needed
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = self.w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat=w_theta_est, w_theta_est=w_theta_est, args=self.args, p0=(1-1/self.n_action)) / (t+1)
                coverage = compute_coverage(cur_theta_est=w_theta_est[0, 0], var_est=var_est[0, 0, 0], theta_true=self.theta[0, 0])
                logger.add_coverage(coverage)

        return logger.to_dict()
    
    def TS(self, rho2=1., l=1., p_0=0.2, x_tilde_test=None):
        """Thompson sampling with predicted states
        
        Args:
            rho2: Known noise variance
            l: Prior variance parameter
            p_0: Minimum selection probability
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        if self.x_list is None or self.x_tilde_list is None or self.potential_reward_list is None:
            raise ValueError("Context data not initialized. Call generate_potential_reward_w_xtilde first.")
            
        if self.n_action != 2:
            return None
            
        T = self.x_list.shape[0]
        d = self.dim

        # Create experiment logger
        logger = BanditExperimentLogger(
            x_list=self.x_list,
            x_tilde_list=self.x_tilde_list,
            potential_reward_list=self.potential_reward_list,
            at_dag_list=self.at_dag_list
        )
        
        # Initialize weighted theta estimator for coverage calculation
        self.w_est_cal = weighted_theta_est(args=self.args)
        
        # Initialize tracking for algorithm-specific variables
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        
        # Initialize algorithm state
        regret = 0
        Vt, Vt_inv, bt = self._initialize_algorithm_state(l)
            
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]
            x_t = self.x_list[t,:]
            
            # Compute posterior for each action
            post_mean_t = np.zeros((self.n_action, d))
            post_var_t = np.zeros((self.n_action, d, d))
            for a in range(self.n_action):
                # Use precomputed inverse instead of calculating it
                post_mean = np.matmul(Vt_inv[a,:,:], bt[a,:])
                post_mean_t[a,:] = post_mean
                # Use precomputed inverse for variance calculation
                post_var_t[a,:,:] = rho2 * Vt_inv[a,:,:]
                estimation_err_list[t,a] = np.linalg.norm(post_mean - self.theta[:,a])
            
            # Compute selection probabilities
            post_mean_w_x_tilde = np.dot(x_tilde_t, post_mean_t[0,:] - post_mean_t[1,:])
            post_var_w_x_tilde = np.dot(x_tilde_t, np.matmul(post_var_t[0,:,:] + post_var_t[1,:,:], x_tilde_t)) + 0.00001
            prob_0 = 1 - stats.norm.cdf(-post_mean_w_x_tilde / np.sqrt(post_var_w_x_tilde))
            pi_0 = prob_clip(prob_0, p_0)
            pi_t = np.array([pi_0, 1 - pi_0])
            
            # Compute test policy probabilities if requested
            pi_test = None
            if x_tilde_test is not None:
                post_mean_w_x_tilde_test = np.dot(x_tilde_test, post_mean_t[0,:] - post_mean_t[1,:])
                post_var_w_x_tilde_test = np.dot(x_tilde_test, np.matmul(post_var_t[0,:,:] + post_var_t[1,:,:], x_tilde_test))+0.00001
                prob_0_test = 1 - stats.norm.cdf(-post_mean_w_x_tilde_test / np.sqrt(post_var_w_x_tilde_test))
                pi_0_test = prob_clip(prob_0_test, p_0)
                pi_test = np.array([pi_0_test, 1 - pi_0_test])
            
            # Compute regret
            regret_t = self._compute_regret(x_t, pi_t, p_0)
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at, rt = self._take_action_and_observe(pi_t, self.potential_reward_list[t,:])
            
            # Update logger with this timestep
            logger.update_step(t, post_mean_t, at, pi_t, pi_test)
            
            # Update sufficient statistics
            Vt, Vt_inv, bt = self._update_statistics_incremental(Vt, Vt_inv, bt, x_tilde_t, at, rt)

            # Update coverage calculation
            self.w_est_cal.update(x_tilde_t, at, rt, pi_t)
            
            # Compute coverage if needed
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = self.w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat=logger.theta_est_list[:, t, :].T, w_theta_est=w_theta_est, args=self.args) / (t+1)
                coverage = compute_coverage(cur_theta_est=w_theta_est[0, 0], var_est=var_est[0, 0, 0], theta_true=self.theta[0, 0])
                logger.add_coverage(coverage)

        return logger.to_dict()

    def Boltzmann(self, gamma=0, l=1., p_0=0.2, x_tilde_test=None):
        """Boltzmann with predicted states
        
        Args:
            gamma: Temperature parameter for softmax
            l: Ridge regression parameter
            p_0: Minimum selection probability
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        if self.x_list is None or self.x_tilde_list is None or self.potential_reward_list is None:
            raise ValueError("Context data not initialized. Call generate_potential_reward_w_xtilde first.")
            
        if self.n_action != 2:
            return None
            
        T = self.x_list.shape[0]
        d = self.dim

        # Create experiment logger
        logger = BanditExperimentLogger(
            x_list=self.x_list,
            x_tilde_list=self.x_tilde_list,
            potential_reward_list=self.potential_reward_list,
            at_dag_list=self.at_dag_list
        )
        
        # Initialize weighted theta estimator for coverage calculation
        self.w_est_cal = weighted_theta_est(args=self.args)
        
        # Initialize tracking for algorithm-specific variables
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        
        # Initialize algorithm state
        regret = 0
        Vt, Vt_inv, bt = self._initialize_algorithm_state(l)
            
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]
            x_t = self.x_list[t,:]
            
            # Compute values for each action
            est_mean_list = np.zeros(self.n_action)
            theta_est_t = np.zeros((self.n_action, d))
            for a in range(self.n_action):
                # Use precomputed inverse instead of calculating it
                theta_est_t[a,:] = np.matmul(Vt_inv[a,:,:], bt[a,:])
                est_mean_list[a] = np.dot(theta_est_t[a,:], x_tilde_t)
                estimation_err_list[t,a] = np.linalg.norm(theta_est_t[a,:] - self.theta[:,a])
                
            # Apply Boltzmann policy
            pi_t = softmax(est_mean_list, gamma)
            
            # Compute test policy probabilities if requested
            pi_test = None
            if x_tilde_test is not None:
                est_mean_list_test = np.zeros(self.n_action)
                for a in range(self.n_action):
                    est_mean_list_test[a] = np.dot(theta_est_t[a,:], x_tilde_test)
                pi_test = softmax(est_mean_list_test, gamma)
            
            # Compute regret
            regret_t = self._compute_regret(x_t, pi_t, p_0)
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at, rt = self._take_action_and_observe(pi_t, self.potential_reward_list[t,:])
            
            # Update logger with this timestep
            logger.update_step(t, theta_est_t, at, pi_t, pi_test)

            # Update coverage calculation
            self.w_est_cal.update(x_tilde_t, at, rt, pi_t)
            
            # Update sufficient statistics
            Vt, Vt_inv, bt = self._update_statistics_incremental(Vt, Vt_inv, bt, x_tilde_t, at, rt)

            # Compute coverage if needed
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = self.w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat=logger.theta_est_list[:, t, :].T, w_theta_est=w_theta_est, args=self.args, if_softmax=True) / (t+1)
                coverage = compute_coverage(cur_theta_est=w_theta_est[0, 0], var_est=var_est[0, 0, 0], theta_true=self.theta[0, 0])
                logger.add_coverage(coverage)

        return logger.to_dict()

    def UCB(self, C=1., l=1., p_0=0.2, x_tilde_test=None):
        """UCB with predicted states
        
        Args:
            C: UCB confidence width parameter
            l: Ridge regression parameter
            p_0: Minimum selection probability
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        if self.x_list is None or self.x_tilde_list is None or self.potential_reward_list is None:
            raise ValueError("Context data not initialized. Call generate_potential_reward_w_xtilde first.")
            
        if self.n_action != 2:
            return None
            
        T = self.x_list.shape[0]
        d = self.dim

        # Create experiment logger
        logger = BanditExperimentLogger(
            x_list=self.x_list,
            x_tilde_list=self.x_tilde_list,
            potential_reward_list=self.potential_reward_list,
            at_dag_list=self.at_dag_list
        )
        
        # Initialize weighted theta estimator for coverage calculation
        self.w_est_cal = weighted_theta_est(args=self.args)
        
        # Initialize tracking for algorithm-specific variables
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        
        # Initialize algorithm state
        regret = 0
        Vt, Vt_inv, bt = self._initialize_algorithm_state(l)
            
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]
            x_t = self.x_list[t,:]
            
            # Compute UCBs for each action
            UCB_list = np.zeros(self.n_action)
            theta_est_t = np.zeros((self.n_action, d))
            for a in range(self.n_action):
                # Use precomputed inverse instead of calculating it
                theta_a_hat = np.matmul(Vt_inv[a,:,:], bt[a,:])
                mu_a = np.dot(theta_a_hat, x_tilde_t)
                # Use precomputed inverse for confidence calculation
                sigma_a2 = np.dot(x_tilde_t, np.matmul(Vt_inv[a,:,:], x_tilde_t))
                UCB_list[a] = mu_a + C * np.sqrt(sigma_a2)
                theta_est_t[a,:] = theta_a_hat
                estimation_err_list[t,a] = np.linalg.norm(theta_a_hat - self.theta[:,a])
                
            at_ucb = np.argmax(UCB_list)
            
            # Set policy probabilities
            pi_t = np.ones(self.n_action) * p_0
            pi_t[at_ucb] = 1 - (self.n_action-1)*p_0
            
            # Compute test policy probabilities if requested
            pi_test = None
            if x_tilde_test is not None:
                UCB_list_test = np.zeros(self.n_action)
                for a in range(self.n_action):
                    # Use precomputed inverse here too
                    theta_a_hat = np.matmul(Vt_inv[a,:,:], bt[a,:])
                    mu_a_test = np.dot(theta_a_hat, x_tilde_test)
                    sigma_a2_test = np.dot(x_tilde_test, np.matmul(Vt_inv[a,:,:], x_tilde_test))
                    UCB_list_test[a] = mu_a_test + C * np.sqrt(sigma_a2_test)
                at_ucb_test = np.argmax(UCB_list_test)
                pi_test = np.ones(self.n_action) * p_0
                pi_test[at_ucb_test] = 1 - (self.n_action-1)*p_0
            
            # Compute regret
            regret_t = self._compute_regret(x_t, pi_t, p_0)
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at, rt = self._take_action_and_observe(pi_t, self.potential_reward_list[t,:])
            
            # Update logger with this timestep
            logger.update_step(t, theta_est_t, at, pi_t, pi_test)

            # Update coverage calculation
            self.w_est_cal.update(x_tilde_t, at, rt, pi_t)
            
            # Update sufficient statistics
            Vt, Vt_inv, bt = self._update_statistics_incremental(Vt, Vt_inv, bt, x_tilde_t, at, rt)

            # Compute coverage if needed
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = self.w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat=logger.theta_est_list[:, t, :].T, w_theta_est=w_theta_est, args=self.args) / (t+1)
                coverage = compute_coverage(cur_theta_est=w_theta_est[0, 0], var_est=var_est[0, 0, 0], theta_true=self.theta[0, 0])
                logger.add_coverage(coverage)

        return logger.to_dict()

    def MEB(self, Sigma_e_hat_list, ind_S, pi_nd_list, p_0=0.2, naive=False, x_tilde_test=None, lambda_=1.):
        """Online measurement error adjustment
        
        Args:
            Sigma_e_hat_list: Estimated measurement error covariance (T x d x d)
            ind_S: Binary vector indicating model update times
            pi_nd_list: Stabilizing policy (T x n_action)
            p_0: Minimum selection probability
            naive: Whether to use naive importance weights
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        if self.x_list is None or self.x_tilde_list is None or self.potential_reward_list is None:
            raise ValueError("Context data not initialized. Call generate_potential_reward_w_xtilde first.")
            
        T = self.x_list.shape[0]
        d = self.dim
        
        # Create experiment logger
        logger = BanditExperimentLogger(
            x_list=self.x_list,
            x_tilde_list=self.x_tilde_list,
            potential_reward_list=self.potential_reward_list,
            at_dag_list=self.at_dag_list
        )
        
        # Initialize tracking for algorithm-specific variables
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        
        # Initialize algorithm state
        s_t = -1
        V_t = np.zeros((self.n_action, d, d))
        V_t_inv = np.zeros((self.n_action, d, d))
        W_t = np.zeros((self.n_action, d, d))
        b_t = np.zeros((self.n_action, d))
        V_st = np.zeros((self.n_action, d, d))
        V_st_inv = np.zeros((self.n_action, d, d))
        W_st = np.zeros((self.n_action, d, d))
        b_st = np.zeros((self.n_action, d))
        theta_st = np.zeros((self.n_action, d))
        regret = 0
        
        # Initialize inverse matrices with small identity matrices
        for a in range(self.n_action):
            V_t_inv[a,:,:] = 1e6 * np.eye(d)  # Large value for empty matrix inverse
            V_st_inv[a,:,:] = 1e6 * np.eye(d)

        w_est_cal = weighted_theta_est(args=self.args)
        
        for t in range(T):
            x_tilde_t = self.x_tilde_list[t,:]
            x_t = self.x_list[t,:]
            
            # Set policy based on most recent theta estimate
            if s_t < 0:
                pi_t = np.ones(self.n_action) * 0.5
                pi_test = np.ones(self.n_action) * 0.5 if x_tilde_test is not None else None
            else:
                at_tilde = np.argmax(np.matmul(theta_st, x_tilde_t))
                pi_t = np.ones(self.n_action) * p_0
                pi_t[at_tilde] = 1 - p_0
                
                # Compute test policy if requested
                pi_test = None
                if x_tilde_test is not None:
                    at_tilde_test = np.argmax(np.matmul(theta_st, x_tilde_test))
                    pi_test = np.ones(self.n_action) * p_0
                    pi_test[at_tilde_test] = 1 - p_0
            
            # Compute regret
            regret_t = self._compute_regret(x_t, pi_t, p_0)
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at, rt = self._take_action_and_observe(pi_t, self.potential_reward_list[t,:])
            
            # Record current estimates
            for a in range(self.n_action):
                estimation_err_list[t,a] = np.linalg.norm(theta_st[a,:] - self.theta[:,a])
            
            # Update logger with this timestep
            logger.update_step(t, theta_st, at, pi_t, pi_test)
            
            # Update sufficient statistics
            imp_weight_at = pi_nd_list[t,at] / pi_t[at] if not naive else 1.0
            for a in range(self.n_action):
                if a == at:
                    # Update V_t using rank-1 update
                    update_term = imp_weight_at * (np.outer(x_tilde_t, x_tilde_t) - Sigma_e_hat_list[t,:,:])
                    V_t[a,:,:] += update_term
                    
                    # Update V_t_inv using Sherman-Morrison for the outer product term
                    if np.linalg.det(V_t[a,:,:]) > 1e-10:
                        V_inv = V_t_inv[a,:,:]
                        x = x_tilde_t
                        
                        # First handle the outer product part
                        V_inv_x = V_inv @ x
                        denominator = 1 + imp_weight_at * (x @ V_inv_x)
                        
                        if abs(denominator) > 1e-10:
                            outer_update = imp_weight_at * np.outer(V_inv_x, V_inv_x) / denominator
                            V_inv = V_inv - outer_update
                            
                        # For the sigma_e part, it's more complex due to full matrix subtraction
                        # For small updates, use approximation; for larger ones, recompute
                        sigma_scaled = imp_weight_at * np.linalg.norm(Sigma_e_hat_list[t,:,:], 'fro')
                        if sigma_scaled / np.linalg.norm(V_t[a,:,:], 'fro') < 0.01:
                            # Approximate update for small sigma_e
                            V_inv = V_inv + sigma_scaled * (V_inv @ V_inv)
                        else:
                            # Direct recomputation for larger sigma_e
                            V_t_inv[a,:,:] = np.linalg.pinv(V_t[a,:,:])
                            continue
                            
                        # Store the updated inverse
                        V_t_inv[a,:,:] = V_inv
                    else:
                        # If matrix is close to singular, recompute
                        V_t_inv[a,:,:] = np.linalg.pinv(V_t[a,:,:])
                    
                    # Update b_t
                    b_t[a,:] += imp_weight_at * x_tilde_t * rt
                
                # Update W_t for covariance adjustment
                if not naive:
                    W_t[a,:,:] += pi_nd_list[t,a] * Sigma_e_hat_list[t,:,:]
                elif a == at:
                    W_t[a,:,:] += Sigma_e_hat_list[t,:,:]
                    
            # Compute coverage
            w_est_cal.update(x_tilde_t, at, rt, pi_t)
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat=logger.theta_est_list[:, t, :].T, w_theta_est=w_theta_est, args=self.args) / (t+1)
                coverage = compute_coverage(cur_theta_est=w_theta_est[0, 0], var_est=var_est[0, 0, 0], theta_true=self.theta[0, 0])
                logger.add_coverage(coverage)
                
            # Update model if indicated
            if ind_S[t] == 1:
                s_t = t
                V_st = V_t.copy()
                V_st_inv = V_t_inv.copy()  # Store the current inverses
                # W_st = W_t.copy()
                b_st = b_t.copy()
                for a in range(self.n_action):
                    # Use regularized matrix for stability
                    # reg_matrix = V_st[a,:,:] + lambda_ * np.eye(d) * t**(1-t/T)
                    
                    # Now we have two options:
                    # 1. Use the Sherman-Morrison formula to update the inverse with the regularization term
                    # 2. Or recompute the inverse directly for this infrequent update
                    
                    # Since this is an infrequent operation and the regularization term changes,
                    # it's simpler and more stable to just recompute the inverse
                    # reg_matrix_inv = np.linalg.pinv(reg_matrix)
                    
                    # Compute parameter estimate
                    theta_st[a,:] = np.matmul(V_st_inv[a,:,:], b_st[a,:])

        return logger.to_dict()