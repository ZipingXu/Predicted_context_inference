from linearBandit import LinBandit
from utility import *
from context_generator import generate_x_tilde, generate_x_tilde_gaussian, sigma_e, sigma_s
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
    parser.add_argument('--gamma', type=float, default = 0, help='Temperature parameter for Boltzmann')
    
    # Problem parameters  (only used in random environment)
    parser.add_argument('--d', type=int, default=1, help='Dimension of context vectors')
    parser.add_argument('--sigma_eta', type=float, default=0, help='Noise variance in rewards')
    parser.add_argument('--sigma_e', type=float, default=0.1, help='Measurement error variance')
    parser.add_argument('--sigma_s', type=float, default=0.1, help='Context variance')
    parser.add_argument('--n_action', type=int, default=2, help='Number of actions')
    parser.add_argument('--lambda_', type=float, default=0, help='Regularization parameter')
    parser.add_argument('--coverage_freq', type=int, default=100, help='Frequency of coverage check')
    # Experiment settings
    parser.add_argument('--n_rep', type=int, default=50, help='Number of experiment repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./runs/', help='Path to save results')
    parser.add_argument('--name', type=str, default='', help='Name of the experiment')
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

args.pi_nd = np.ones((args.n_action)) / args.n_action

# setup environment
if args.env not in ['random', 'failure1', 'failure2']:
    raise ValueError('Invalid environment')

if args.env == 'random':
    generate_x_tilde = lambda T: generate_x_tilde_gaussian(T, args.sigma_s, args.sigma_e, args.d)
    args.theta = np.random.normal(0, 1, args.d * args.n_action).reshape((args.d, args.n_action))
    args.theta_all = []
    # args.theta = []
    # args.sigma_s = sigma_s
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

# Define algorithm runners with updated parameter lists
runUCB = lambda bandit_inst: bandit_inst.run_bandit(
        policy='ucb', 
        x_tilde_test = np.array([-1.])
)
runTS = lambda bandit_inst: bandit_inst.run_bandit(
        policy='ts',
        x_tilde_test = np.array([-1.]))
runBoltzmann = lambda bandit_inst: bandit_inst.run_bandit(
        policy='boltzmann', 
        x_tilde_test = np.array([-1.]))
runRandom = lambda bandit_inst: bandit_inst.run_bandit(
        policy='random',
        x_tilde_test = np.array([-1.]))

# Prepare data for MEB algorithm
ind_S = (np.arange(args.T) > 100)
pi_nd_list = 0.5 * np.ones((args.T, args.n_action))
pi_nd = np.array([0.5, 0.5])
Sigma_e_hat_list = np.zeros((args.T, args.d, args.d))
for t in range(args.T):
    Sigma_e_hat_list[t, :, :] = args.sigma_e * np.eye(args.d)
    
runMEB = lambda bandit_inst: bandit_inst.run_bandit(
        policy='meb',
        ind_S = ind_S, 
        pi_nd_list = pi_nd_list, 
        naive = False, 
        lambda_ = args.lambda_,
        x_tilde_test = np.array([-1.])
)

alg_dict = {
    'UCB': runUCB,
    'TS': runTS,
    'MEB': runMEB,
    'Boltzmann': runBoltzmann,
    'Random': runRandom
}

# Initialize history dictionary with empty structure
history_dict = {
    "theta_est": {},
    "pi_list": {},
    "theta_est_batch": {},
    "theta_est_naive": {},
    "coverage_list": {},
    "policy_func": {}
}

# Define algorithms
algorithms = ["UCB", "TS", "MEB", "Boltzmann", "Random"]

# Initialize arrays for each algorithm
for alg in algorithms:
    history_dict["theta_est"][alg] = np.zeros((args.n_rep, args.n_action, args.T // args.coverage_freq, args.d))
    history_dict["pi_list"][alg] = np.zeros((args.n_rep, args.T // args.coverage_freq, args.n_action))
    history_dict["theta_est_batch"][alg] = np.zeros((args.n_rep, args.n_action, args.d))
    history_dict["theta_est_naive"][alg] = np.zeros((args.n_rep, args.n_action, args.d))
    history_dict["coverage_list"][alg] = np.zeros((args.n_rep, args.T // args.coverage_freq))
    history_dict["policy_func"][alg] = []
for i in tqdm(range(args.n_rep), desc="Running experiments"):
    # Generate contexts
    x_list, x_tilde_list = generate_x_tilde(args.T)
    if args.env == 'random':
        args.theta = np.random.normal(0, 1, args.d * args.n_action).reshape((args.d, args.n_action))
        args.theta_all.append(args.theta)

    # Create bandit instance and initialize contexts
    Bandit_1 = LinBandit(theta=args.theta, sigma=args.sigma_eta, args=args)
    Bandit_1.generate_potential_reward_w_xtilde(x_list=x_list, x_tilde_list=x_tilde_list)
    
    for alg in alg_dict.keys():
        # Execute algorithm (now without passing contexts explicitly)
        result_dict = alg_dict[alg](Bandit_1)
        
        # Store results in history dictionary
        # Subsample theta_est_list to match coverage frequency
        history_dict['theta_est'][alg][i, :, :, :] = result_dict['theta_est_list'][:, ::args.coverage_freq, :]
        history_dict['pi_list'][alg][i, :, :] = result_dict['pi_list_test'][::args.coverage_freq, :]
        history_dict['coverage_list'][alg][i, :] = np.array(result_dict['coverage_list'])[::args.coverage_freq]
        history_dict['policy_func'][alg].append(result_dict['policy_func'])
        
        # Calculate and store weighted and naive theta estimates
        history_dict['theta_est_batch'][alg][i, :, :] = weighted_theta_est(
            history=result_dict, 
            pi_nd=pi_nd,
            sigma_e=args.sigma_e
        )
        history_dict['theta_est_naive'][alg][i, :, :] = naive_theta_est(result_dict)
        
# Save history dictionary to a pickle file
if args.env == 'random':
    name = f'{args.save_path}history_dict_random_{args.sigma_s}_{args.sigma_e}_{args.name}.pkl'
else:
    name = f'{args.save_path}history_dict_{args.env}_{args.name}.pkl'
with open(name, 'wb') as f:
    pickle.dump((history_dict, args), f)
