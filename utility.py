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


def estimate_variance(theta_hat, args, n_true = 1000, if_softmax = False):
    ## estimate the asymptotic variance of the weighted estimator
    d = theta_hat.shape[0]
    n_action = theta_hat.shape[1]
    if args.env == "random":
        x_list, x_tilde_list = generate_x_tilde_gaussian(n_true, args.sigma_s, args.sigma_e, d)
    else:
        x_list, x_tilde_list = generate_x_tilde(n_true)
        
    asy_var = np.zeros((n_action, d, d))

    for t in range(n_true):
        x_tilde_t = x_tilde_list[t, :]
        x_t = x_list[t, :]
        max_r = np.max(x_tilde_t * theta_hat)
        for a in range(n_action):
            if if_softmax:
                tmp_a = softmax(x_tilde_t * theta_hat[0, :], gamma = 3)[a]
            else:
                tmp_a = (1-args.p0) if x_tilde_t * theta_hat[0, a] == max_r else args.p0 / (n_action - 1)
            ha_t = (x_tilde_t.reshape((d, 1)) @ x_tilde_t.reshape((1, d)) - args.sigma_e) @ theta_hat[:, a] - x_tilde_t.reshape((d, 1)) @ x_t.reshape((1, d)) @ theta_hat[:, a]
            asy_var[a, :, :] += 1 / tmp_a * (ha_t.reshape((d, 1)) @ ha_t.reshape((1, d)) + args.sigma_eta * x_tilde_t.reshape((d, 1)) @ x_tilde_t.reshape((1, d)))
    # for a in range(n_action):
    #     asy_var[a, :, :] *= asy_var[a, :, :] / (pi_nd[a]**2)
    asy_var = asy_var * args.sigma_s**(-2)
    return asy_var / n_true

def naive_theta_est(history, n_action = 2):
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
    
    return_list = theta_est_batch
    return(return_list)

def weighted_theta_est(history, pi_nd, sigma_e, n_action = 2, stop_t = None):
    ## calculate the weighted estimator for thetas given a bandit history data, with iid potential outcome
    ## history MUST be a dictionary containing the following entries:
    ##         'x_list': list of true underlying context, T*d
    ##         'x_tilde_list': list of predicted context list, T*d
    ##         'at_list': list of selected actions, T dimensional
    ##         'pi_list': list of action selection probabilities, T * n_action
    ##         'potential_reward_list': list of all potential rewards, T * n_action
    ##         'theta_est_list': list of estimators of thetas during the online algorithm, n_action * T * d
    ## pi_nd: (stationary) stablizing policy, n_action dimensional
    ## Sigma_e: covariance matrix of the measurement error, d * d
    ## OUTPUT: theta_est_batch: weighted estimators of theta with the batch data, n_action * d

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
    Vt = np.zeros((n_action, d)) # Vt[a, :, :] = \sum_t W_t1_{A_t=a} x_tilde_t * Y_t
    t = 0
    
    while (t<stop_t):
        A_t = int(at_list[t])
        x_tilde_t = x_tilde_list[t, :]
        rt = potential_reward_list[t, A_t]
        
        ## extract importance weight for action a = A_t
        ipt_wt = pi_nd[A_t] / pi_list[t, A_t]
        
        ## update Ut for a = A_t
        Ut[A_t, :, :] = Ut[A_t, :, :] + ipt_wt * (np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d))) - sigma_e)
        
        ## update Vt for a = A_t
        Vt[A_t, :] = Vt[A_t, :] + ipt_wt * x_tilde_t * rt
        
        ## update t
        t = t + 1
    
    ## compute weighted estimator theta_hat
    theta_est_batch = np.zeros((n_action, d))
    for a in range(n_action):
        theta_est_batch[a, :] = np.matmul(np.linalg.pinv(Ut[a, :, :]), Vt[a, :])
    
    return_list = theta_est_batch
    return(return_list)