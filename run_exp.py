from linearBandit import LinBandit
from utility import *
from context_generator import generate_x_tilde, generate_x_tilde_gaussian, sigma_e, sigma_s, sigma_s_gaussian
import argparse
from tqdm import tqdm
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Run linear bandit experiments with inference')
    
    # Algorithm parameters
    parser.add_argument('-T', type=int, default=5000,
                        help='Number of rounds')
    parser.add_argument('--p0', type=float, default=0.1,
                        help='Minimum probability threshold (between 0 and 0.5)')
    parser.add_argument('--env', type=str, default = 'failure1', help='choice from [random, failure1, failure2]')
    
    # Problem parameters  (only used in random environment)
    parser.add_argument('--d', type=int, default=1, help='Dimension of context vectors')
    parser.add_argument('--sigma_eta', type=float, default=0, help='Noise variance in rewards')
    parser.add_argument('--sigma_e', type=float, default=0.1, help='Measurement error variance')
    parser.add_argument('--n_action', type=int, default=2, help='Number of actions')
    # Experiment settings
    parser.add_argument('--n_rep', type=int, default=50, help='Number of experiment repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./runs/', help='Path to save results')
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

# setup environment
if args.env not in ['random', 'failure1', 'failure2']:
    raise ValueError('Invalid environment')

if args.env == 'random':
    generate_x_tilde = lambda T: generate_x_tilde_gaussian(T, args.sigma_s, args.sigma_e, args.d)
    args.theta = np.random.normal(0, 1, args.d * args.n_action).reshape((args.d, args.n_action))
    args.sigma_s = sigma_s_gaussian
elif args.env == 'failure1':
    args.d = 1
    args.n_action = 2
    args.sigma_e = sigma_e
    args.sigma_s = sigma_s
    args.theta = np.array([3, 1]).reshape((args.d, args.n_action))
elif args.env == 'failure2':
    args.d = 1
    args.n_action = 2
    args.sigma_e = sigma_e
    args.sigma_s = sigma_s
    args.theta = np.array([-3, -1]).reshape((args.d, args.n_action))

# algorithm default parameters

runUCB = lambda x_list, x_tilde_list, bandit_inst, prwd: bandit_inst.UCB_w_predicted_state (
        x_list = x_list,
        x_tilde_list = x_tilde_list, 
        potential_reward_list = prwd['potential_reward_list'],  
        C = 1.0, l = 1., p_0 = args.p0, 
        x_tilde_test = np.array([-1.])
)
runTS = lambda x_list, x_tilde_list, bandit_inst, prwd: bandit_inst.TS_w_predicted_state(
        x_list = x_list,
        x_tilde_list = x_tilde_list, 
        potential_reward_list = prwd['potential_reward_list'], 
        rho2 = args.sigma_eta ** 2, l = 1., p_0 = args.p0, 
        x_tilde_test = np.array([-1.]))

ind_S = (np.arange(args.T) > 100)
pi_nd_list = 0.5 * np.ones((args.T, args.n_action))
pi_nd = np.array([0.5, 0.5])
Sigma_e_hat_list = np.zeros((args.T, args.d, args.d))
for t in range(args.T):
    Sigma_e_hat_list[t, :, :] = args.sigma_e * np.eye(args.d)
runMEB = lambda x_list, x_tilde_list, bandit_inst, pwrd: bandit_inst.online_me_adjust_w_predicted_state(
        x_list = x_list, 
        x_tilde_list = x_tilde_list, 
        potential_reward_list = pwrd['potential_reward_list'],
        Sigma_e_hat_list = Sigma_e_hat_list,
        ind_S = ind_S, 
        pi_nd_list = pi_nd_list, 
        p_0 = args.p0, 
        naive = False, 
        x_tilde_test = np.array([-1.])
)

alg_dict = {
    'UCB': runUCB,
    'TS': runTS,
    'MEB': runMEB
}

history_dict = {
    "theta_est":{
        "UCB": np.zeros((args.n_rep, args.n_action, args.T, args.d)),
        "TS": np.zeros((args.n_rep, args.n_action, args.T, args.d)),
        "MEB": np.zeros((args.n_rep, args.n_action, args.T, args.d))
    },
    "pi_list":{
        "UCB": np.zeros((args.n_rep, args.T, args.n_action)),
        "TS": np.zeros((args.n_rep, args.T, args.n_action)),
        "MEB": np.zeros((args.n_rep, args.T, args.n_action))
    },
    "theta_est_batch":{
        "UCB": np.zeros((args.n_rep, args.n_action, args.d)),
        "TS": np.zeros((args.n_rep, args.n_action, args.d)),
        "MEB": np.zeros((args.n_rep, args.n_action, args.d))
    },
    "theta_est_naive":{
        "UCB": np.zeros((args.n_rep, args.n_action, args.d)),
        "TS": np.zeros((args.n_rep, args.n_action, args.d)),
        "MEB": np.zeros((args.n_rep, args.n_action, args.d))
    }
}

for i in tqdm(range(args.n_rep), desc="Running experiments"):
    x_list, x_tilde_list = generate_x_tilde(args.T)
    Bandit_1 = LinBandit(theta = args.theta, sigma = args.sigma_eta)
    Bandit_info = Bandit_1.generate_potential_reward_w_xtilde(x_list = x_list, x_tilde_list = x_tilde_list)
    
    for alg in alg_dict.keys():
        history = alg_dict[alg](x_list, x_tilde_list, Bandit_1, Bandit_info)
        history_dict['theta_est'][alg][i, :, :, :] = history['theta_est_list']
        history_dict['pi_list'][alg][i, :, :] = history['pi_list_test']
        history_dict['theta_est_batch'][alg][i, :, :] = weighted_theta_est(
            history = history, 
            pi_nd = pi_nd,
            sigma_e = args.sigma_e
        )
        history_dict['theta_est_naive'][alg][i, :, :] = naive_theta_est(history)

# Save history dictionary to a pickle file
with open(f'{args.save_path}history_dict_{args.env}.pkl', 'wb') as f:
    pickle.dump((history_dict, args), f)
