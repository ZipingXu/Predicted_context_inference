import numpy as np
from scipy.stats import norm
# from context_generator import generate_x_tilde, generate_x_tilde_gaussian

## utility functions
def prob_clip(x, p_0):
    ## clip the probability value x to be between [p_0, 1-p_0]
    ## require 0<=p_0<=1/2
    if (x < p_0):
        return(p_0)
    if (x > (1-p_0)):
        return(1-p_0)
    return(x)

def softmax(x, gamma = 10):
    return(np.exp(x * gamma) / np.sum(np.exp(x * gamma)))

def compute_coverage(cur_theta_est, var_est, theta_true, alpha = 0.05):
    up_bound = cur_theta_est + np.sqrt(var_est) * norm.ppf(1-alpha/2)
    lower_bound = cur_theta_est - np.sqrt(var_est) * norm.ppf(1-alpha/2)      
    cover = 1 if (theta_true < up_bound and theta_true > lower_bound) else 0
    return(cover)


def estimate_variance(policy_func, w_theta_est, env, n_true = 1000):
    ## estimate the asymptotic variance of the weighted estimator
    d = w_theta_est.shape[0]
    n_action = w_theta_est.shape[1]

    x_list, x_tilde_list = env.generate_context(n_true)
    potential_reward_list, _ = env.generate_potential_rewards(x_list, x_tilde_list, noise = True)
        
    asy_var = np.zeros((n_action, d, d))
    avg_pa = np.zeros(n_action)
    # p0 = args.p0 if p0 is None else p0
    # calculate the asymptotic variance \Sigma_{\theta, a} according to Theorem 4.1
    if env.mis:
        Hat = np.zeros((n_action, d, d))
        for t in range(n_true):
            x_tilde_t = x_tilde_list[t, :]
            pa = policy_func(x_tilde_t)
            avg_pa += pa
            at = np.random.choice(range(n_action), p=pa)
            Hat[at, :, :] += (1/(pa[at])**2) * np.outer(x_tilde_t, x_tilde_t) * (potential_reward_list[t, at] - np.dot(x_tilde_t, w_theta_est[:, at]))**2
        asy_var = Hat * env.sigma_s**(-2) / n_true
        avg_pa = avg_pa / n_true
    else:
        for t in range(n_true):
            x_tilde_t = x_tilde_list[t, :]
            x_t = x_list[t, :]
            for a in range(n_action):
                pa = policy_func(x_tilde_t)[a]
                avg_pa[a] += pa
                ha_t = (np.outer(x_tilde_t, x_tilde_t) - env.sigma_e * np.eye(d)) @ w_theta_est[:, a] - np.outer(x_tilde_t, x_t) @ w_theta_est[:, a]
                asy_var[a, :, :] += (1 / pa) * (np.outer(ha_t, ha_t) + (env.sigma_eta**2) * np.outer(x_tilde_t, x_tilde_t))
        # for a in range(n_action):
        #     asy_var[a, :, :] *= asy_var[a, :, :] / (pi_nd[a]**2)
        asy_var = asy_var * env.sigma_s**(-2) / n_true 
        avg_pa = avg_pa / n_true
    return asy_var, avg_pa

def naive_theta_est(history, n_action=2):
    """Calculate naive theta estimates from experiment history.
    
    Args:
        history: Either a dictionary of experiment history or a result dictionary 
                from BanditExperimentLogger.to_dict()
        n_action: Number of actions
        
    Returns:
        Naive theta estimates (n_action x d)
    """
    x_list = history['x_list']
    x_tilde_list = history['x_tilde_list']
    d = x_list.shape[1]
    T = x_list.shape[0]

    at_list = history['at_list']
    potential_reward_list = history['potential_reward_list']
    
    # initialization
    Vt = np.zeros((n_action, d, d)) # Ut[a, :, :] = \sum_t W_t1_{A_t=a}(x_tilde_t * x_tilde_t^T - Sigma_e)
    bt = np.zeros((n_action, d)) # Vt[a, :, :] = \sum_t W_t1_{A_t=a} x_tilde_t * Y_t
    t = 0
    
    while (t<T):
        A_t = int(at_list[t])
        x_tilde_t = x_tilde_list[t, :]
        rt = potential_reward_list[t, A_t]
        
        ## update Ut for a = A_t
        Vt[A_t, :, :] = Vt[A_t, :, :] + (np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d))))
        
        ## update Vt for a = A_t
        bt[A_t, :] = bt[A_t, :] + x_tilde_t * rt
        
        ## update t
        t = t + 1
    
    ## compute weighted estimator theta_hat
    theta_est_batch = np.zeros((n_action, d))
    for a in range(n_action):
        theta_est_batch[a, :] = np.matmul(np.linalg.pinv(Vt[a, :, :]), bt[a, :])
    
    return theta_est_batch

def weighted_theta_est(history, pi_nd, sigma_e, n_action=2, stop_t=None):
    """Calculate weighted theta estimates from experiment history.
    
    Args:
        history: Either a dictionary of experiment history or a result dictionary 
                from BanditExperimentLogger.to_dict()
        pi_nd: (stationary) stabilizing policy, n_action dimensional
        sigma_e: Covariance matrix of the measurement error, d * d
        n_action: Number of actions
        stop_t: Optional stopping time (default: use all data)
        
    Returns:
        Weighted theta estimates (n_action x d)
    """
    if pi_nd is None:
        pi_nd = np.ones(n_action) / n_action
    ## extract information
    x_list = history['x_list']
    x_tilde_list = history['x_tilde_list']
    d = x_list.shape[1]
    T = x_list.shape[0]
    if stop_t is None:
        stop_t = T

    at_list = history['at_list']
    pi_list = history['pi_list']
    potential_reward_list = history['potential_reward_list']
    
    # initialization
    Vt = np.zeros((n_action, d, d)) # Ut[a, :, :] = \sum_t W_t1_{A_t=a}(x_tilde_t * x_tilde_t^T - Sigma_e)
    bt = np.zeros((n_action, d)) # Vt[a, :, :] = \sum_t W_t1_{A_t=a} x_tilde_t * Y_t
    
    # Initialize Ut_inv with small values for numerical stability
    for a in range(n_action):
        Vt[a, :, :] = 1e-6 * np.eye(d)
    
    t = 0
    
    while (t < stop_t):
        A_t = int(at_list[t])
        x_tilde_t = x_tilde_list[t, :]
        rt = potential_reward_list[t, A_t]
        
        ## extract importance weight for action a = A_t
        ipt_wt = pi_nd[A_t] / pi_list[t, A_t]
        
        ## update Ut for a = A_t using rank-1 update
        update_term = ipt_wt * (np.outer(x_tilde_t, x_tilde_t) - sigma_e * np.eye(d))
        Vt[A_t, :, :] += update_term

        ## update bt for a = A_t
        bt[A_t, :] += ipt_wt * x_tilde_t * rt
        
        ## update t
        t = t + 1
    
    ## compute weighted estimator theta_hat using maintained inverse
    theta_est_batch = np.zeros((n_action, d))
    for a in range(n_action):
        theta_est_batch[a, :] = np.matmul(np.linalg.pinv(Vt[a, :, :]), bt[a, :])
    
    return theta_est_batch