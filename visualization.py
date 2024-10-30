import pickle
import numpy as np
import matplotlib.pyplot as plt
from utility import estimate_variance
import seaborn as sns

# Academic-style colors - commonly used in papers
# Blue similar to common scientific paper blues, Red similar to common scientific paper reds
PAPER_BLUE = '#0072B2'  # A clear, professional blue
PAPER_RED = '#D55E00'   # A clear, professional red
# Set default colors for matplotlib
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[PAPER_BLUE, PAPER_RED])

save_path = './results/'

with open('./runs/history_dict_failure1.pkl', 'rb') as f:
    history_dict, args = pickle.load(f)

print(args)

algorithms = ['UCB', 'TS', 'MEB']

def plot_theta_est(history_dict, args, diff=False, save = None):
    if not diff:
        fig, axs = plt.subplots(3, 1, figsize=(6, 9))
        for i_algorithm, algorithm in enumerate(algorithms):
            for i_experiment in range(args.n_rep):
                axs[i_algorithm].plot(np.arange(1, args.T+1), history_dict['theta_est'][algorithm][i_experiment, 0, :, 0], color=PAPER_BLUE, alpha=0.2)
                axs[i_algorithm].plot(np.arange(1, args.T+1), history_dict['theta_est'][algorithm][i_experiment, 1, :, 0], color=PAPER_RED, alpha=0.2)
            axs[i_algorithm].set_title(f'estimated theta of {algorithm}')
            legend = ['action 0', 'action 1']
            axs[i_algorithm].legend(legend)
            axs[i_algorithm].set_xlabel('T')
            axs[i_algorithm].axhline(y=args.theta[0,0], color="blue", linestyle='--')
            axs[i_algorithm].axhline(y=args.theta[0,1], color="red", linestyle='--')
            if algorithm == 'MEB':
                axs[i_algorithm].set_ylim(min(args.theta[0,0], args.theta[0,1])-1, max(args.theta[0,0], args.theta[0,1])+1)
        plt.tight_layout()
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
    else:
        fig, axs = plt.subplots(3, 1, figsize=(6, 9))
        for i_algorithm, algorithm in enumerate(algorithms):
            theta_est_diff = np.mean(history_dict['theta_est'][algorithm][:, 0, :, 0] - history_dict['theta_est'][algorithm][:, 1, :, 0], axis=0)
            axs[i_algorithm].plot(np.arange(1, args.T+1), theta_est_diff, color=PAPER_BLUE)
            axs[i_algorithm].set_title(f'Difference in theta of {algorithm}')
            axs[i_algorithm].set_xlabel('T')
            axs[i_algorithm].set_ylabel('Difference in theta')
            if args.env == 'failure1':
                axs[i_algorithm].set_ylim(-0.5, 3)
            else:
                axs[i_algorithm].set_ylim(-3, 0.5)
        plt.tight_layout()
        plt.show()

def plot_pi_list(history_dict, args, save = None):
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    for i_algorithm, algorithm in enumerate(algorithms):
        means_pi0 = np.mean(history_dict['pi_list'][algorithm][:, :, 0], axis=0)
        ses_pi0 = np.std(history_dict['pi_list'][algorithm][:, :, 0], axis=0)
        axs[i_algorithm].plot(np.arange(1, args.T+1), means_pi0, color=PAPER_BLUE, alpha=0.6)
        axs[i_algorithm].fill_between(np.arange(1, args.T+1), means_pi0 - ses_pi0, means_pi0 + ses_pi0, color=PAPER_BLUE, alpha=0.1)
        axs[i_algorithm].set_title(f'Sampling probability at x=1 of {algorithm}')
        axs[i_algorithm].set_xlabel('T')
        axs[i_algorithm].set_ylabel('Sampling probability')
        axs[i_algorithm].set_ylim(0, 1)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()

def plot_batch_est(history_dict, args, draw_a = 0, save = None):
    if args.env == 'random':
        raise ValueError('Is not implemented for random environment')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # First subplot: density plot for UCB batch estimates
    for i_algorithm, algorithm in enumerate(algorithms):
        theta_est_batch = history_dict['theta_est_batch'][algorithm][:, draw_a, 0]
        true_theta = args.theta[0, draw_a]
        sns.kdeplot(theta_est_batch, color=PAPER_BLUE, ax=axes[i_algorithm])
        # Add vertical line at x=-3
        axes[i_algorithm].axvline(x=true_theta, color=PAPER_RED, linestyle='--')

        # predicted
        theta_hat = np.mean(history_dict['theta_est'][algorithm][:, :, -1, :], axis=0).reshape((args.d, args.n_action))
        # print(theta_hat)
        variance = estimate_variance(theta_hat = theta_hat, args = args, n_true = 10000)
        variance = variance/args.T
        # print(algorithm, variance)
        x = np.linspace(true_theta-0.5,true_theta+0.5,1000)
        gaussian = 1/np.sqrt(2*variance[draw_a, 0, 0]*np.pi) * np.exp(-(x-true_theta)**2/2/variance[draw_a, 0, 0])
        axes[i_algorithm].plot(x, gaussian, color='green', linestyle=':', label='N(3,1)')
        axes[i_algorithm].set_title(f'{algorithm}')
        # axes[i_algorithm].legend()
    plt.suptitle('Density plot of weighted estimators')
    plt.tight_layout()
    plt.show()

def plot_naive_est(history_dict, args, draw_a = 0, save = None):
    if args.env == 'random':
        raise ValueError('Is not implemented for random environment')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # First subplot: density plot for UCB batch estimates
    for i_algorithm, algorithm in enumerate(algorithms):
        theta_est_naive = history_dict['theta_est_naive'][algorithm][:, draw_a, 0]
        true_theta = args.theta[0, draw_a]
        sns.kdeplot(theta_est_naive, color=PAPER_BLUE, ax=axes[i_algorithm])
        # Add vertical line at x=-3
        axes[i_algorithm].axvline(x=true_theta, color=PAPER_RED, linestyle='--')
        # axes[i_algorithm].plot(x, gaussian, color='green', linestyle=':', label='N(3,1)')
        axes[i_algorithm].set_title(f'{algorithm}')
        # axes[i_algorithm].legend()
    plt.suptitle('Density plot of naive estimators')
    plt.tight_layout()
    plt.show()