# import numpy as np

# # failure mode 1

# d = 1
# x_dict = [0, -1]  ## true context uniformly drawn from this distribution
# e_dict = {}
# e_dict[0] = {"e": [1 , -2], "p": [2/3, 1/3]}
# e_dict[-1] = {"e": [-1, 2], "p": [2/3, 1/3]}

# # variance of e_t
# sigma_e = (2./3 * 1 + 1./3 * 4)
# # variance of x_t
# sigma_s = np.sum((x_dict-(np.mean(x_dict, axis=0)))**2, axis=0)
# # sigma_s_gaussian = 1

# # following two functions are for generating x_tilde_list of dimension (T, d)

# def generate_x_tilde(T):
#     x_list = np.zeros((T, d)) 
#     x_tilde_list = np.zeros((T, d)) 
#     for t in range(T):
#         x_idx = np.random.randint(0, 2)
#         x_t = x_dict[x_idx]
#         e_list = e_dict[x_t]["e"]        ## list of possible values for e_t
#         e_prob = e_dict[x_t]["p"]
#         e_t = np.random.choice(e_list, p = e_prob)
#         x_tilde_t = x_t + e_t
#         x_list[t, :] = x_t
#         x_tilde_list[t, :] = x_tilde_t
#     return x_list, x_tilde_list

# def generate_x_tilde_gaussian(T, sigma_s, sigma_e, d):
#     x_list = np.random.multivariate_normal(np.zeros(d), np.array([[sigma_s]]), T)
#     x_tilde_list = x_list + np.random.multivariate_normal(np.zeros(d), np.array([[sigma_e]]), T)
#     return x_list, x_tilde_list