import numpy as np
import scipy.stats as stats
from utility import prob_clip, compute_coverage, weighted_theta_est, estimate_variance


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
        self.W_t = np.zeros((self.n_action, self.d, self.d))
        self.b_t = np.zeros((self.n_action, self.d))
    def update(self, x_tilde_t, a_t, r_t, pi_t):
        imp_weight_at = self.pi_nd[a_t] / pi_t[a_t]
        for a in range(self.n_action):
            if a == a_t:
                self.V_t[a,:,:] += imp_weight_at * (np.outer(x_tilde_t, x_tilde_t) - self.sigma_e * np.eye(self.d))
                self.b_t[a,:] += imp_weight_at * x_tilde_t * r_t
            # self.W_t[a,:,:] += self.pi_nd[a] * self.sigma_e * np.eye(self.d)
    def get_theta_est(self):
        theta_st = np.zeros((self.n_action, self.d))
        for a in range(self.n_action):
            theta_st[a,:] = np.matmul(np.linalg.pinv(self.V_t[a,:,:]), self.b_t[a,:])
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

        return {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list
        }
    def random_policy(self, x_list, x_tilde_list, potential_reward_list, at_dag_list, x_tilde_test=None):
        """Random policy"""
        T = x_list.shape[0]
        pi_list = np.ones((T, self.n_action)) * 0.5
        at_list = np.random.choice(self.n_action, T)
        pi_list_test = np.ones((T, self.n_action)) * 0.5

        return {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "at_dag_list": at_dag_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test
        }
    def TS_w_predicted_state(self, x_list, x_tilde_list, potential_reward_list, rho2, l=1., p_0=0.2, x_tilde_test=None):
        """Thompson sampling with predicted states
        
        Args:
            x_list: Context list (T x d)
            x_tilde_list: Predicted context list (T x d) 
            potential_reward_list: Potential rewards (T x n_action)
            at_dag_list: Oracle actions (T)
            rho2: Known noise variance
            l: Prior variance parameter
            p_0: Minimum selection probability
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        if self.n_action != 2:
            return None
            
        T = x_list.shape[0]
        d = self.dim

        self.w_est_cal = weighted_theta_est(args = self.args)
        
        # Initialize tracking variables
        theta_est_list = np.zeros((self.n_action, T, d))
        pi_list = np.zeros((T, self.n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        pi_list_test = np.zeros((T, self.n_action))
        coverage_list = []

        history = {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test,
            "coverage_list": coverage_list
        }
        # Initialize algorithm state
        regret = 0
        Vt = np.zeros((self.n_action, d, d))
        bt = np.zeros((self.n_action, d))
        for a in range(self.n_action):
            Vt[a,:,:] = l * np.eye(d)
            
        for t in range(T):
            x_tilde_t = x_tilde_list[t,:]
            x_t = x_list[t,:]
            
            # Compute posterior for each action
            post_mean_t = np.zeros((self.n_action, d))
            post_var_t = np.zeros((self.n_action, d, d))
            for a in range(self.n_action):
                post_mean = np.matmul(np.linalg.inv(Vt[a,:,:]), bt[a,:])
                post_mean_t[a,:] = post_mean
                post_var_t[a,:,:] = rho2 * np.linalg.inv(Vt[a,:,:])
                theta_est_list[a,t,:] = post_mean
                estimation_err_list[t,a] = np.linalg.norm(post_mean - self.theta[:,a])
            
            # Compute selection probabilities
            post_mean_w_x_tilde = np.dot(x_tilde_t, post_mean_t[0,:] - post_mean_t[1,:])
            post_var_w_x_tilde = np.dot(x_tilde_t, np.matmul(post_var_t[0,:,:] + post_var_t[1,:,:], x_tilde_t)) + 0.00001
            prob_0 = 1 - stats.norm.cdf(-post_mean_w_x_tilde / np.sqrt(post_var_w_x_tilde))
            pi_0 = prob_clip(prob_0, p_0)
            pi_list[t,0] = pi_0
            pi_list[t,1] = 1 - pi_0
            
            # Compute regret
            if np.dot(x_t, self.theta[:,0]) > np.dot(x_t, self.theta[:,1]):
                regret_t = np.dot(x_t, self.theta[:,0] - self.theta[:,1]) * (1 - p_0 - pi_0)
            else:
                regret_t = np.dot(x_t, self.theta[:,1] - self.theta[:,0]) * (pi_0 - p_0)
            regret += regret_t
            regret_list[t] = regret
            
            # Evaluate test policy if requested
            if x_tilde_test is not None:
                post_mean_w_x_tilde_test = np.dot(x_tilde_test, post_mean_t[0,:] - post_mean_t[1,:])
                post_var_w_x_tilde_test = np.dot(x_tilde_test, np.matmul(post_var_t[0,:,:] + post_var_t[1,:,:], x_tilde_test))
                prob_0_test = 1 - stats.norm.cdf(-post_mean_w_x_tilde_test / np.sqrt(post_var_w_x_tilde_test))
                pi_0_test = prob_clip(prob_0_test, p_0)
                pi_list_test[t,0] = pi_0_test
                pi_list_test[t,1] = 1 - pi_0_test
            
            # Take action and observe reward
            at = np.random.binomial(1, 1 - pi_0)
            at_list[t] = at
            rt = potential_reward_list[t,at]
            
            # Update sufficient statistics
            Vt[at,:,:] += np.outer(x_tilde_t, x_tilde_t)
            bt[at,:] += x_tilde_t * rt

            self.w_est_cal.update(x_tilde_t, at, rt, pi_list[t,:])
            # Compute coverage
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = self.w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat = theta_est_list[:, t, :].T, args = self.args) / (t+1)
                coverage_list.append(compute_coverage(cur_theta_est = w_theta_est[0, 0], var_est = var_est[0, 0, 0], theta_true = self.theta[0, 0]))

        return {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test,
            "coverage_list": coverage_list
        }

    def UCB_w_predicted_state(self, x_list, x_tilde_list, potential_reward_list, C, l=1., p_0=0.2, x_tilde_test=None):
        """UCB with predicted states
        
        Args:
            x_list: Context list (T x d)
            x_tilde_list: Predicted context list (T x d)
            potential_reward_list: Potential rewards (T x n_action)
            at_dag_list: Oracle actions (T)
            C: UCB confidence width parameter
            l: Ridge regression parameter
            p_0: Minimum selection probability
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        if self.n_action != 2:
            return None
            
        T = x_list.shape[0]
        d = self.dim

        self.w_est_cal = weighted_theta_est(args = self.args)
        
        # Initialize tracking variables
        theta_est_list = np.zeros((self.n_action, T, d))
        pi_list = np.zeros((T, self.n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        pi_list_test = np.zeros((T, self.n_action))
        coverage_list = []
        history = {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test,
            "coverage_list": coverage_list
        }
        # Initialize algorithm state
        regret = 0
        Vt = np.zeros((self.n_action, d, d))
        bt = np.zeros((self.n_action, d))
        for a in range(self.n_action):
            Vt[a,:,:] = l * np.eye(d)
            
        for t in range(T):
            x_tilde_t = x_tilde_list[t,:]
            x_t = x_list[t,:]
            
            # Compute UCBs for each action
            UCB_list = np.zeros(self.n_action)
            for a in range(self.n_action):
                theta_a_hat = np.matmul(np.linalg.inv(Vt[a,:,:]), bt[a,:])
                mu_a = np.dot(theta_a_hat, x_tilde_t)
                sigma_a2 = np.dot(x_tilde_t, np.matmul(np.linalg.inv(Vt[a,:,:]), x_tilde_t))
                UCB_list[a] = mu_a + C * np.sqrt(sigma_a2)
                theta_est_list[a,t,:] = theta_a_hat
                estimation_err_list[t,a] = np.linalg.norm(theta_a_hat - self.theta[:,a])
                
            at_ucb = np.argmax(UCB_list)
            
            # Set policy probabilities
            for a in range(self.n_action):
                pi_list[t,a] = 1 - (self.n_action-1)*p_0 if a == at_ucb else p_0
                
            # Evaluate test policy if requested
            if x_tilde_test is not None:
                UCB_list_test = np.zeros(self.n_action)
                for a in range(self.n_action):
                    theta_a_hat = np.matmul(np.linalg.inv(Vt[a,:,:]), bt[a,:])
                    mu_a_test = np.dot(theta_a_hat, x_tilde_test)
                    sigma_a2_test = np.dot(x_tilde_test, np.matmul(np.linalg.inv(Vt[a,:,:]), x_tilde_test))
                    UCB_list_test[a] = mu_a_test + C * np.sqrt(sigma_a2_test)
                at_ucb_test = np.argmax(UCB_list_test)
                for a in range(self.n_action):
                    pi_list_test[t,a] = 1 - (self.n_action-1)*p_0 if a == at_ucb_test else p_0
            
            # Compute regret
            if np.dot(x_t, self.theta[:,0]) > np.dot(x_t, self.theta[:,1]):
                reward_oracle_t = np.dot(x_t, self.theta[:,0]) * (1-p_0) + np.dot(x_t, self.theta[:,1]) * p_0
            else:
                reward_oracle_t = np.dot(x_t, self.theta[:,1]) * (1-p_0) + np.dot(x_t, self.theta[:,0]) * p_0
            reward_policy = np.dot(x_t, self.theta[:,0]) * pi_list[t,0] + np.dot(x_t, self.theta[:,1]) * pi_list[t,1]
            regret_t = reward_oracle_t - reward_policy
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at = np.random.binomial(1, pi_list[t,1])
            at_list[t] = at
            rt = potential_reward_list[t,at]

            self.w_est_cal.update(x_tilde_t, at, rt, pi_list[t,:])
            
            # Update sufficient statistics
            Vt[at,:,:] += np.outer(x_tilde_t, x_tilde_t)
            bt[at,:] += x_tilde_t * rt

            # Compute coverage
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = self.w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat = theta_est_list[:, t, :].T, args = self.args) / (t+1)
                coverage_list.append(compute_coverage(cur_theta_est = w_theta_est[0, 0], var_est = var_est[0, 0, 0], theta_true = self.theta[0, 0]))

        return {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test,
            "coverage_list": coverage_list
        }

    def online_me_adjust_w_predicted_state(self, x_list, x_tilde_list, potential_reward_list, Sigma_e_hat_list, ind_S, pi_nd_list, p_0=0.2, naive=False, x_tilde_test=None, lambda_=1.):
        """Online measurement error adjustment
        
        Args:
            x_list: Context list (T x d)
            x_tilde_list: Predicted context list (T x d)
            potential_reward_list: Potential rewards (T x n_action)
            at_dag_list: Oracle actions (T)
            Sigma_e_hat_list: Estimated measurement error covariance (T x d x d)
            ind_S: Binary vector indicating model update times
            pi_nd_list: Stabilizing policy (T x n_action)
            p_0: Minimum selection probability
            naive: Whether to use naive importance weights
            x_tilde_test: Test context for policy evaluation
            
        Returns:
            Dictionary containing history and results
        """
        T = x_list.shape[0]
        d = self.dim
        
        # Initialize tracking variables
        theta_est_list = np.zeros((self.n_action, T, d))
        pi_list = np.zeros((T, self.n_action))
        at_list = np.zeros(T)
        estimation_err_list = np.zeros((T, self.n_action))
        regret_list = np.zeros(T)
        pi_list_test = np.zeros((T, self.n_action))
        coverage_list = []
        history = {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test,
            "coverage_list": coverage_list
        }
        # Initialize algorithm state
        s_t = -1
        V_t = np.zeros((self.n_action, d, d))
        W_t = np.zeros((self.n_action, d, d))
        b_t = np.zeros((self.n_action, d))
        V_st = np.zeros((self.n_action, d, d))
        W_st = np.zeros((self.n_action, d, d))
        b_st = np.zeros((self.n_action, d))
        theta_st = np.zeros((self.n_action, d))
        regret = 0

        w_est_cal = weighted_theta_est(args = self.args)
        
        for t in range(T):
            x_tilde_t = x_tilde_list[t,:]
            x_t = x_list[t,:]
            
            # Set policy based on most recent theta estimate
            if s_t < 0:
                pi_list[t,:] = 0.5
                if x_tilde_test is not None:
                    pi_list_test[t,:] = 0.5
            else:
                at_tilde = np.argmax(np.matmul(theta_st, x_tilde_t))
                pi_list[t,at_tilde] = 1 - p_0
                pi_list[t,1-at_tilde] = p_0
                if x_tilde_test is not None:
                    at_tilde_test = np.argmax(np.matmul(theta_st, x_tilde_test))
                    pi_list_test[t,at_tilde_test] = 1 - p_0
                    pi_list_test[t,1-at_tilde_test] = p_0
            
            # Compute regret
            if np.dot(x_t, self.theta[:,0]) > np.dot(x_t, self.theta[:,1]):
                reward_oracle_t = np.dot(x_t, self.theta[:,0]) * (1-p_0) + np.dot(x_t, self.theta[:,1]) * p_0
            else:
                reward_oracle_t = np.dot(x_t, self.theta[:,1]) * (1-p_0) + np.dot(x_t, self.theta[:,0]) * p_0
            reward_policy = np.dot(x_t, self.theta[:,0]) * pi_list[t,0] + np.dot(x_t, self.theta[:,1]) * pi_list[t,1]
            regret_t = reward_oracle_t - reward_policy
            regret += regret_t
            regret_list[t] = regret
            
            # Take action and observe reward
            at = np.random.binomial(1, pi_list[t,1])
            at_list[t] = at
            rt = potential_reward_list[t,at]
            
            # Record current estimates
            for a in range(self.n_action):
                theta_est_list[a,t,:] = theta_st[a,:]
                estimation_err_list[t,a] = np.linalg.norm(theta_st[a,:] - self.theta[:,a])
            
            # Update sufficient statistics
            imp_weight_at = pi_nd_list[t,at] / pi_list[t,at] if not naive else 1.0
            for a in range(self.n_action):
                if a == at:
                    V_t[a,:,:] += imp_weight_at * (np.outer(x_tilde_t, x_tilde_t) - Sigma_e_hat_list[t,:,:])
                    # V_t[a,:,:] += imp_weight_at * np.outer(x_tilde_t, x_tilde_t)
                    b_t[a,:] += imp_weight_at * x_tilde_t * rt
                if not naive:
                    W_t[a,:,:] += pi_nd_list[t,a] * Sigma_e_hat_list[t,:,:]
                elif a == at:
                    W_t[a,:,:] += Sigma_e_hat_list[t,:,:]
            # Compute coverage
            w_est_cal.update(x_tilde_t, at, rt, pi_list[t,:])
            if (t+1) % self.coverage_freq == 0:
                w_theta_est = w_est_cal.get_theta_est().T
                var_est = estimate_variance(theta_hat = w_theta_est, args = self.args) / (t+1)
                coverage_list.append(compute_coverage(cur_theta_est = w_theta_est[0, 0], var_est = var_est[0, 0, 0], theta_true = self.theta[0, 0]))
            # Update model if indicated
            if ind_S[t] == 1:
                s_t = t
                V_st = V_t.copy()
                W_st = W_t.copy()
                b_st = b_t.copy()
                for a in range(self.n_action):
                    # theta_st[a,:] = np.matmul(np.linalg.pinv(V_st[a,:,:] - W_st[a,:,:] + lambda_ * np.eye(d) * t**(1-t/T)), b_st[a,:])
                    theta_st[a,:] = np.matmul(np.linalg.pinv(V_st[a,:,:] + lambda_ * np.eye(d) * t**(1-t/T)), b_st[a,:])

        return {
            "x_list": x_list,
            "x_tilde_list": x_tilde_list,
            "potential_reward_list": potential_reward_list,
            "theta_est_list": theta_est_list,
            "at_list": at_list,
            "pi_list": pi_list,
            "pi_list_test": pi_list_test,
            "coverage_list": coverage_list
        }