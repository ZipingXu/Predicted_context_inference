


import numpy as np

## utility functions
def prob_clip(x, p_0):
    ## clip the probability value x to be between [p_0, 1-p_0]
    ## require 0<=p_0<=1/2
    if (x < p_0):
        return(p_0)
    if (x > (1-p_0)):
        return(1-p_0)
    return(x)

def weighted_theta_est(history, pi_nd, Sigma_e, n_action = 2):
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
    
    ## extract information
    x_list = history['x_list']
    x_tilde_list = history['x_tilde_list']
    d = x_list.shape[1]
    T = x_list.shape[0]

    at_list = history['at_list']
    pi_list = history['pi_list']
    potential_reward_list = history['potential_reward_list']
    if ('theta_est_list' in list(history.keys())):
        theta_est_list = history['theta_est_list']
    else:
        theta_est_list = None
    
    # initialization
    Ut = np.zeros((n_action, d, d)) # Ut[a, :, :] = \sum_t W_t1_{A_t=a}(x_tilde_t * x_tilde_t^T - Sigma_e)
    Vt = np.zeros((n_action, d)) # Vt[a, :, :] = \sum_t W_t1_{A_t=a} x_tilde_t * Y_t
    t = 0
    
    while (t<T):
        A_t = int(at_list[t])
        x_tilde_t = x_tilde_list[t, :]
        rt = potential_reward_list[t, A_t]
        
        ## extract importance weight for action a = A_t
        ipt_wt = pi_nd[A_t] / pi_list[t, A_t]
        
        ## update Ut for a = A_t
        Ut[A_t, :, :] = Ut[A_t, :, :] + ipt_wt * (np.matmul(x_tilde_t.reshape((d, 1)), x_tilde_t.reshape((1, d))) - Sigma_e)
        
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