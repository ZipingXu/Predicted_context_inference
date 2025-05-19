import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class EnvConfig:
    """Configuration for the environment"""
    d: int = 1  # dimension of context
    n_action: int = 2  # number of actions
    sigma_eta: float = 0.0  # noise in reward
    sigma_e: float = 1.0  # noise in context observation
    sigma_s: float = 1.0  # variance of true context
    env_type: str = "random"  # type of environment: "random", "failure1", "failure2", "polynomial", or "neural_network"
    theta: Optional[np.ndarray] = None  # true theta parameters (d x n_action)
    mis: bool = False  # whether to use misspecified theta
    p0: float = 0.1  # minimum probability threshold
    context_noise: str = "Gaussian"  # noise type: "Gaussian" or "Laplace" or "Student-t"
    # algorithm specific parameters
    q: int = 4  # degree of polynomial for polynomial environment
    hidden_dim: int = 5  # hidden dimension for neural network
    df: int = 10  # degrees of freedom for Student-t noise
    def __post_init__(self):
        self.theta = None

class NeuralNetwork(nn.Module):
    """Two-layer neural network with ReLU activation"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class Environment:
    """Environment class for contextual bandit with measurement error"""
    
    def __init__(self, config: EnvConfig):
        """Initialize environment with configuration
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.d = config.d
        self.n_action = config.n_action
        self.theta = config.theta
        self.nn = None
        self.best_theta = None
        self.sigma_eta = config.sigma_eta
        self.sigma_e = config.sigma_e
        self.sigma_s = config.sigma_s
        self.mis = config.mis
        self.p0 = config.p0
        self.q = config.q
        self.hidden_dim = config.hidden_dim
        
        # For failure modes
        if config.env_type in ["failure1", "failure2"]:
            self.x_dict = [0, -1]  # true context uniformly drawn from this distribution
            self.e_dict = {
                0: {"e": [1, -2], "p": [2/3, 1/3]},
                -1: {"e": [-1, 2], "p": [2/3, 1/3]}
            }
            # variance of e_t
            self.sigma_e = (2./3 * 1**2 + 1./3 * 2**2) # 2
            # variance of x_t
            # self.sigma_s = np.mean((self.x_dict-(np.mean(self.x_dict, axis=0)))**2, axis=0) # 0.25
            if self.mis:
                self.sigma_s = np.mean(np.array(self.x_dict)**2, axis=0) + self.sigma_e
            else:
                self.sigma_s = np.mean(np.array(self.x_dict)**2, axis=0) # 0.5
            self.d = 1
            self.n_action = 2
    def initialize_nn(self):
        self.nn = NeuralNetwork(self.d, self.hidden_dim, self.n_action)
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.nn.layer1.weight)
        nn.init.xavier_uniform_(self.nn.layer2.weight)
        nn.init.zeros_(self.nn.layer1.bias)
        nn.init.zeros_(self.nn.layer2.bias)
    
    def initialize(self, T: int):
        """Initialize environment for running bandit algorithm
        
        Args:
            T: Number of timesteps
        """
        self.generate_theta()
        self.x_list, self.x_tilde_list = self.generate_context(T)
        self.potential_reward_list, self.at_dag_list = self.generate_potential_rewards(self.x_list, self.x_tilde_list)
    
    def generate_theta(self):
        """Generate theta parameters based on environment type"""
        if self.config.env_type == "random":
            if self.theta is None:
                self.theta = np.random.normal(0, 1, self.d * self.n_action).reshape((self.d, self.n_action))
        elif self.config.env_type == "failure1":
            self.theta = np.array([3, 1]).reshape((self.d, self.n_action))
        elif self.config.env_type == "failure2":
            self.theta = np.array([-3, -1]).reshape((self.d, self.n_action))
        elif self.config.env_type == "polynomial":
            # For polynomial, theta has shape (d*q, n_action)
            if self.theta is None:
                self.theta = np.random.normal(0, 1, self.d * self.q * self.n_action).reshape((self.d * self.q, self.n_action))
        elif self.config.env_type == "neural_network":
            # For neural network, theta is handled by the nn model
            if self.nn is None:
                self.initialize_nn()
        else:
            raise ValueError(f"Unknown environment type: {self.config.env_type}")
        
        # best theta is an estimate of the inference target
        self.best_theta = self.compute_best_linear_theta()
    
    def generate_context(self, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate true and noisy contexts for T timesteps
        
        Args:
            T: Number of timesteps
            
        Returns:
            Tuple of (true contexts, noisy contexts) each of shape (T, d)
        """
        if self.config.env_type in ["random", "neural_network", "polynomial"]:
            x_list = np.random.multivariate_normal(
                np.zeros(self.d), 
                np.eye(self.d)*self.sigma_s, 
                T
            )
            if self.config.context_noise == "Gaussian":
                x_tilde_list = x_list + np.random.multivariate_normal(
                    np.zeros(self.d), 
                    np.eye(self.d)*self.sigma_e, 
                    T
                )
            elif self.config.context_noise == "Laplace":
                std_t = np.std(np.random.laplace(0, self.sigma_e, 5000))
                x_tilde_list = x_list + np.random.laplace(0, self.sigma_e, (T, self.d)) * np.sqrt(self.sigma_e) / std_t
            else:
                raise ValueError(f"Unknown context noise type: {self.config.context_noise}")
        elif self.config.env_type in ["failure1", "failure2"]:
            x_list = np.zeros((T, self.d))
            x_tilde_list = np.zeros((T, self.d))
            
            for t in range(T):
                x_idx = np.random.randint(0, 2)
                x_t = self.x_dict[x_idx]
                e_list = self.e_dict[x_t]["e"]
                e_prob = self.e_dict[x_t]["p"]
                e_t = np.random.choice(e_list, p=e_prob)
                x_tilde_t = x_t + e_t
                x_list[t, :] = x_t
                x_tilde_list[t, :] = x_tilde_t
        else:
            raise ValueError(f"Unknown environment type: {self.config.env_type}")
        
        if self.mis and self.config.env_type in ["polynomial", "neural_network"]:
            x_tilde_list = x_list # x_tilde_list should be the true context for polynomial and neural network
        return x_list, x_tilde_list
    def generate_potential_rewards(self, x_list: np.ndarray, x_tilde_list: np.ndarray, noise = True):
        T = x_list.shape[0]
        potential_reward_list = np.zeros((T, self.n_action))
        at_dag_list = np.zeros(T)
        for t in range(T):
            x_t = x_list[t, :]
            # x_tilde_t = x_tilde_list[t, :]
            reward_list = []
            for a in range(self.n_action):
                if noise:
                    potential_reward_t = self.realized_reward(x_t, a)
                else:
                    potential_reward_t = self.mean_reward(x_t, a)
                reward_list.append(potential_reward_t)
            potential_reward_list[t, :] = np.array(reward_list)
            at_dag_list[t] = np.argmax(reward_list)
        return potential_reward_list, at_dag_list

    def _polynomial_features(self, x: np.ndarray) -> np.ndarray:
        """Generate polynomial features up to degree q
        
        Args:
            x: Input vector of shape (d,)
            
        Returns:
            Polynomial features of shape (d*q,)
        """
        features = []
        for i in range(1, self.q + 1):
            features.append(x ** i)
        return np.concatenate(features)

    def mean_reward(self, x: np.ndarray, a: int) -> float:
        """Compute mean reward for context x and action a
        
        Args:
            x: Context vector of shape (d,)
            a: Action index
            
        Returns:
            Mean reward
        """
        if a >= self.n_action:
            raise ValueError(f"Action {a} is out of range [0, {self.n_action-1}]")
            
        if self.config.env_type == "polynomial":
            # Generate polynomial features
            x_poly = self._polynomial_features(x)
            return np.dot(self.theta[:, a], x_poly)
        elif self.config.env_type == "neural_network":
            # Convert to torch tensor and compute through neural network
            x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = self.nn(x_tensor)
            return output[0, a].item()
        else:
            return np.dot(self.theta[:, a], x)
    
    def realized_reward(self, x: np.ndarray, a: int) -> float:
        """Compute realized reward with noise for context x and action a
        
        Args:
            x: Context vector of shape (d,)
            a: Action index
            
        Returns:
            Realized reward with noise
        """
        mu = self.mean_reward(x, a)
        return mu + np.random.normal(0, np.sqrt(self.sigma_eta))
    
    def compute_regret(self, x_t, pi_t, p0):
        """Compute regret for the current timestep
        
        Args:
            x_t: Current context
            pi_t: Current policy probabilities
            p0: Minimum selection probability
            
        Returns:
            Regret for the current timestep
        """
        if self.config.env_type == "polynomial":
            x_poly = self._polynomial_features(x_t)
            mean_rewards = np.matmul(x_poly.reshape((1, -1)), self.theta).reshape(self.n_action)
        elif self.config.env_type == "neural_network":
            x_tensor = torch.FloatTensor(x_t).unsqueeze(0)
            with torch.no_grad():
                mean_rewards = self.nn(x_tensor).squeeze().numpy()
        else:
            mean_rewards = np.matmul(x_t.reshape((1, self.d)), self.theta).reshape(self.n_action)
            
        best_action = np.argmax(mean_rewards)
        second_best = np.argsort(mean_rewards)[-2]
        
        reward_oracle_t = mean_rewards[best_action] * (1-p0) + mean_rewards[second_best] * p0
        reward_policy = np.sum(mean_rewards * pi_t)
        
        return reward_oracle_t - reward_policy
    def compute_best_linear_theta(self, n = 100000):
        if self.best_theta is not None:
            return self.best_theta
        x_list, x_tilde_list = self.generate_context(n)
        potential_reward_list, at_dag_list = self.generate_potential_rewards(x_list, x_tilde_list, noise = False)
        best_theta = np.zeros((self.d, self.n_action))
        XX = np.linalg.inv(np.dot(x_tilde_list.T, x_tilde_list))
        for a in range(self.n_action):
            best_theta[:, a] = np.dot(XX, np.dot(x_tilde_list.T, potential_reward_list[:, a]))
        self.best_theta = best_theta
        return best_theta
