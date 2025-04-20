import numpy as np
from scipy.stats import norm
from context_generator import generate_x_tilde, generate_x_tilde_gaussian

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


def estimate_variance(theta_hat, w_theta_est, args, n_true = 1000, if_softmax = False, p0 = None):
    ## estimate the asymptotic variance of the weighted estimator
    d = theta_hat.shape[0]
    n_action = theta_hat.shape[1]
    if args.env == "random":
        x_list, x_tilde_list = generate_x_tilde_gaussian(n_true, args.sigma_s, args.sigma_e, d)
    else:
        x_list, x_tilde_list = generate_x_tilde(n_true)
        
    asy_var = np.zeros((n_action, d, d))
    p0 = args.p0 if p0 is None else p0
    # calculate the asymptotic variance \Sigma_{\theta, a} according to Theorem 4.1
    for t in range(n_true):
        x_tilde_t = x_tilde_list[t, :]
        x_t = x_list[t, :]
        max_r = np.max(np.dot(x_tilde_t, theta_hat))
        for a in range(n_action):
            if if_softmax:
                pa = softmax(np.dot(x_tilde_t, theta_hat), gamma = args.gamma)[a]
            else:
                pa = (1-p0) if np.dot(x_tilde_t, theta_hat[:, a]) == max_r else p0 / (n_action - 1)
            ha_t = (x_tilde_t.reshape((d, 1)) @ x_tilde_t.reshape((1, d)) - args.sigma_e) @ w_theta_est[:, a] - x_tilde_t.reshape((d, 1)) @ x_t.reshape((1, d)) @ w_theta_est[:, a]
            asy_var[a, :, :] += (1 / pa) * (ha_t.reshape((d, 1)) @ ha_t.reshape((1, d)) + args.sigma_eta * x_tilde_t.reshape((d, 1)) @ x_tilde_t.reshape((1, d)))
    # for a in range(n_action):
    #     asy_var[a, :, :] *= asy_var[a, :, :] / (pi_nd[a]**2)
    asy_var = asy_var * args.sigma_s**(-2)
    return asy_var / n_true

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
    pi_list = history['pi_list']
    potential_reward_list = history['potential_reward_list']
    
    # initialization
    Ut = np.zeros((n_action, d, d)) # Ut[a, :, :] = \sum_t W_t1_{A_t=a}(x_tilde_t * x_tilde_t^T - Sigma_e)
    Vt = np.zeros((n_action, d)) # Vt[a, :, :] = \sum_t W_t1_{A_t=a} x_tilde_t * Y_t
    t = 0
    
    while (t<T):
        A_t = int(at_list[t])
        x_tilde_t = x_tilde_list[t, :]
        rt = potential_reward_list[t, A_t]
        
        ## update Ut for a = A_t
        Ut[A_t, :, :] = Ut[A_t, :, :] + (np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d))))
        
        ## update Vt for a = A_t
        Vt[A_t, :] = Vt[A_t, :] + x_tilde_t * rt
        
        ## update t
        t = t + 1
    
    ## compute weighted estimator theta_hat
    theta_est_batch = np.zeros((n_action, d))
    for a in range(n_action):
        theta_est_batch[a, :] = np.matmul(np.linalg.pinv(Ut[a, :, :]), Vt[a, :])
    
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
    Ut = np.zeros((n_action, d, d)) # Ut[a, :, :] = \sum_t W_t1_{A_t=a}(x_tilde_t * x_tilde_t^T - Sigma_e)
    Ut_inv = np.zeros((n_action, d, d)) # Inverse of Ut for each action
    Vt = np.zeros((n_action, d)) # Vt[a, :, :] = \sum_t W_t1_{A_t=a} x_tilde_t * Y_t
    
    # Initialize Ut_inv with small values for numerical stability
    for a in range(n_action):
        Ut[a, :, :] = 1e-6 * np.eye(d)
        Ut_inv[a, :, :] = 1e6 * np.eye(d)
    
    t = 0
    
    while (t < stop_t):
        A_t = int(at_list[t])
        x_tilde_t = x_tilde_list[t, :]
        rt = potential_reward_list[t, A_t]
        
        ## extract importance weight for action a = A_t
        ipt_wt = pi_nd[A_t] / pi_list[t, A_t]
        
        ## update Ut for a = A_t using rank-1 update
        update_term = ipt_wt * (np.outer(x_tilde_t, x_tilde_t) - sigma_e)
        Ut[A_t, :, :] += update_term
        
        ## Update Ut_inv using Sherman-Morrison formula
        # For the outer product term
        U_inv = Ut_inv[A_t, :, :]
        x = x_tilde_t
        
        # Apply Sherman-Morrison for the outer product update
        U_inv_x = U_inv @ x
        denominator = 1 + ipt_wt * (x @ U_inv_x)
        
        if abs(denominator) > 1e-10:  # Check to avoid division by near-zero
            outer_update = ipt_wt * np.outer(U_inv_x, U_inv_x) / denominator
            U_inv -= outer_update
        
        # Handle the sigma_e term (approximation for small updates)
        if isinstance(sigma_e, (int, float)) and sigma_e > 0:
            sigma_e_scaled = ipt_wt * sigma_e
            if sigma_e_scaled / np.linalg.norm(Ut[A_t, :, :], 'fro') < 0.01:
                U_inv += sigma_e_scaled * (U_inv @ U_inv)
            else:
                # For larger updates, recompute directly
                Ut_inv[A_t, :, :] = np.linalg.pinv(Ut[A_t, :, :])
        elif isinstance(sigma_e, np.ndarray) and np.any(sigma_e > 0):
            # For matrix sigma_e, compute inverse directly as it's more complex
            Ut_inv[A_t, :, :] = np.linalg.pinv(Ut[A_t, :, :])
        else:
            # Store the updated inverse
            Ut_inv[A_t, :, :] = U_inv
        
        ## update Vt for a = A_t
        Vt[A_t, :] += ipt_wt * x_tilde_t * rt
        
        ## update t
        t = t + 1
    
    ## compute weighted estimator theta_hat using maintained inverse
    theta_est_batch = np.zeros((n_action, d))
    for a in range(n_action):
        theta_est_batch[a, :] = np.matmul(Ut_inv[a, :, :], Vt[a, :])
    
    return theta_est_batch