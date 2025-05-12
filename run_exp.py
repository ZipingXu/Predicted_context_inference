from linearBandit import LinBandit
from utility import *
from environment import Environment, EnvConfig
import argparse
from tqdm import tqdm
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Run linear bandit experiments with inference')
    
    # Algorithm parameters
    parser.add_argument('-T', type=int, default=5000,
                        help='Number of rounds')
    parser.add_argument('--p0', type=float, default=0.1,
                        help='Minimum probability threshold (between 0 and 0.5)')
    parser.add_argument('--env', type=str, default = 'failure1', help="choice from [random, failure1, failure2, logistic, neural_network, polynomial]")
    
    # Policy parameters
    parser.add_argument('--gamma', type=float, default = 0, help='Temperature parameter for Boltzmann')
    parser.add_argument('-C', type=float, default = 1.0, help='Hyperparameter for UCB')
    parser.add_argument('-l', type=float, default = 1.0, help='Regularization parameter for all algorithms')
    parser.add_argument('--rho', type=float, default = 1.0, help='Hyperparameter for TS')
    
    # Problem parameters  (only used in random environment)
    parser.add_argument('-d', type=int, default=1, help='Dimension of context vectors')
    parser.add_argument('--sigma_eta', type=float, default=0.1, help='Reward noise variance')
    parser.add_argument('--context_noise', type=str, default='Gaussian', help='Context noise type', choices=['Gaussian', 'Laplace'])
    parser.add_argument('--sigma_e', type=float, default=2, help='Context noise variance')
    parser.add_argument('--sigma_s', type=float, default=0.25, help='Context variance (only used in random environment)')
    parser.add_argument('--n_action', type=int, default=2, help='Number of actions')
    parser.add_argument('--coverage_freq', type=int, default=100, help='Frequency of coverage check')
    # Experiment settings
    parser.add_argument('--n_rep', type=int, default=50, help='Number of experiment repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./runs/', help='Path to save results')
    parser.add_argument('--name', type=str, default='', help='Name of the experiment')
    parser.add_argument('--print', action='store_true', help='Print results')

    # run a single algorithm
    parser.add_argument('--single_alg', type=str, default=None, help='Name of the algorithm to run')

    # experiment for general model misspecification
    parser.add_argument("--mis", action='store_true', help="Run the general model misspecification experiment")
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

args.pi_nd = np.ones((args.n_action)) / args.n_action

# setup environment
if args.env not in ['random', 'failure1', 'failure2', 'logistic', 'neural_network', 'polynomial']:
    raise ValueError('Invalid environment')

if not args.mis and args.env not in ['random', 'failure1', 'failure2']:
    raise ValueError('Predicted context inference experiment only supports random, failure1, failure2')

if args.mis:
    args.sigma_e = 0
# Create environment configuration
env_config = EnvConfig(
    d=args.d,
    n_action=args.n_action,
    sigma_eta=args.sigma_eta,
    sigma_e=args.sigma_e,
    sigma_s=args.sigma_s,
    env_type=args.env,
    mis=args.mis,
    context_noise=args.context_noise
)

# Define algorithm runners with updated parameter lists
runUCB = lambda bandit_inst: bandit_inst.run_bandit(
        policy='ucb', 
        x_tilde_test = np.zeros(args.d)-1)
runTS = lambda bandit_inst: bandit_inst.run_bandit(
        policy='ts',
        x_tilde_test = np.zeros(args.d)-1)
runBoltzmann = lambda bandit_inst: bandit_inst.run_bandit(
        policy='boltzmann', 
        x_tilde_test = np.zeros(args.d)-1)
runRandom = lambda bandit_inst: bandit_inst.run_bandit(
        policy='random',
        x_tilde_test = np.zeros(args.d)-1)

# Prepare data for MEB algorithm
pi_nd_list = 0.5 * np.ones((args.T, args.n_action))
pi_nd = np.array([0.5, 0.5])
Sigma_e_hat_list = np.zeros((args.T, args.d, args.d))
for t in range(args.T):
    Sigma_e_hat_list[t, :, :] = args.sigma_e * np.eye(args.d)
    
runMEB = lambda bandit_inst: bandit_inst.run_bandit(
        policy='meb',
        pi_nd_list = pi_nd_list, 
        naive = False, 
        x_tilde_test = np.zeros(args.d)-1
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
    "policy_func": {},
    "avg_pa_list": {},
    "var_est_list": {}
}

# Define algorithms
algorithms = ["UCB", "TS", "MEB", "Boltzmann", "Random"]

# Initialize arrays for each algorithm
# print(int(np.floor(args.T / args.coverage_freq)))
for alg in algorithms:
    history_dict["theta_est"][alg] = np.zeros((args.n_rep, args.n_action, int(np.ceil(args.T / args.coverage_freq)), args.d))
    history_dict["pi_list"][alg] = np.zeros((args.n_rep, int(np.ceil(args.T / args.coverage_freq)), args.n_action))
    history_dict["theta_est_batch"][alg] = np.zeros((args.n_rep, args.n_action, args.d))
    history_dict["theta_est_naive"][alg] = np.zeros((args.n_rep, args.n_action, args.d))
    history_dict["coverage_list"][alg] = np.zeros((args.n_rep, int(np.ceil(args.T / args.coverage_freq))))
    history_dict["avg_pa_list"][alg] = np.zeros((args.n_rep, int(np.ceil(args.T / args.coverage_freq)), args.n_action))
    history_dict["var_est_list"][alg] = np.zeros((args.n_rep, int(np.ceil(args.T / args.coverage_freq))))
    history_dict["policy_func"][alg] = []

# Store all thetas for environment
args.theta_all = []

env = Environment(env_config)
for i in tqdm(range(args.n_rep), desc="Running experiments"):
    # Create environment instance
    
    # Generate theta for this repetition
    # theta = env.generate_theta()
    env.initialize(args.T)
    args.theta_all.append(env.theta)
    
    # Create bandit instance and initialize contexts    
    Bandit_1 = LinBandit(env=env, args=args)
    
    for alg in alg_dict.keys():

        if args.single_alg is not None and alg != args.single_alg:
            continue
        # Execute algorithm
        result_dict = alg_dict[alg](Bandit_1)
        
        # Store results in history dictionary
        history_dict['theta_est'][alg][i, :, :, :] = result_dict['theta_est_list'][:, ::args.coverage_freq, :]
        history_dict['pi_list'][alg][i, :, :] = result_dict['pi_list_test'][::args.coverage_freq, :]
        history_dict['coverage_list'][alg][i, :] = np.array(result_dict['coverage_list'])[::args.coverage_freq]
        history_dict['avg_pa_list'][alg][i, :, :] = np.array(result_dict['avg_pa_list'])[::args.coverage_freq, :]
        history_dict['var_est_list'][alg][i, :] = np.array(result_dict['var_est_list'])[::args.coverage_freq]
        history_dict['policy_func'][alg].append(result_dict['policy_func'])
        
        # Calculate and store weighted and naive theta estimates
        history_dict['theta_est_batch'][alg][i, :, :] = weighted_theta_est(
            history=result_dict, 
            pi_nd=pi_nd,
            sigma_e=env.sigma_e
        )
        history_dict['theta_est_naive'][alg][i, :, :] = naive_theta_est(result_dict)

# I want to reduce the memory usage
# del history_dict['theta_est']
# del history_dict['pi_list']
# del history_dict['avg_pa_list']
        
# Save history dictionary to a pickle file
if args.env == 'random':
    name = f'{args.save_path}history_dict_random_{args.sigma_s}_{args.sigma_e}_{args.name}.pkl'
else:
    name = f'{args.save_path}history_dict_{args.env}_{args.name}.pkl'
with open(name, 'wb') as f:
    pickle.dump((history_dict, args), f)

if args.print and args.single_alg is not None:
    print("true theta \n", env.theta) if not args.mis else print("true theta \n", env.best_theta)
    print("weighted theta est \n", np.mean(history_dict['theta_est'][args.single_alg], axis=0)[:, 1, :])
    print("var est \n", np.mean(history_dict['var_est_list'][args.single_alg], axis=0)[-1])
    print("coverage \n", np.mean(history_dict['coverage_list'][args.single_alg], axis=0)[-1])
    print("weighted theta std \n", np.std(history_dict['theta_est'][args.single_alg], axis=0)[:, 1, :]**2)