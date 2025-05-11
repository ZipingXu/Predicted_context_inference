import unittest
import numpy as np
from run_exp import parse_args, Environment, EnvConfig, LinBandit

class TestRunExp(unittest.TestCase):
    def setUp(self):
        # Create minimal test arguments
        self.args = parse_args()
        self.args.T = 100  # Small number of rounds for testing
        self.args.n_rep = 2  # Small number of repetitions
        self.args.env = 'random'  # Use random environment
        self.args.d = 2  # Small dimension
        self.args.n_action = 2  # Binary actions
        self.args.coverage_freq = 20  # Check coverage every 20 steps
        
    def test_environment_creation(self):
        """Test that environment can be created and initialized"""
        env_config = EnvConfig(
            d=self.args.d,
            n_action=self.args.n_action,
            sigma_eta=self.args.sigma_eta,
            sigma_e=self.args.sigma_e,
            sigma_s=self.args.sigma_s,
            env_type=self.args.env,
            mis=self.args.mis
        )
        env = Environment(env_config)
        self.assertIsNotNone(env)
        
    def test_bandit_creation(self):
        """Test that bandit can be created and run"""
        env_config = EnvConfig(
            d=self.args.d,
            n_action=self.args.n_action,
            sigma_eta=self.args.sigma_eta,
            sigma_e=self.args.sigma_e,
            sigma_s=self.args.sigma_s,
            env_type=self.args.env,
            mis=self.args.mis
        )
        env = Environment(env_config)
        theta = env.generate_theta()
        
        bandit = LinBandit(env, self.args)
        self.assertIsNotNone(bandit)
        
    def test_algorithm_runs(self):
        """Test that all algorithms can run without errors"""
        env_config = EnvConfig(
            d=self.args.d,
            n_action=self.args.n_action,
            sigma_eta=self.args.sigma_eta,
            sigma_e=self.args.sigma_e,
            sigma_s=self.args.sigma_s,
            env_type=self.args.env,
            mis=self.args.mis
        )
        env = Environment(env_config)
        theta = env.generate_theta()
        
        bandit = LinBandit(env, self.args)
        
        # Test each algorithm
        algorithms = ['random', 'ucb', 'ts', 'boltzmann', 'meb']
        for alg in algorithms:
            result = bandit.run_bandit(policy=alg, x_tilde_test=np.array([-1.]))
            self.assertIsNotNone(result)
            self.assertIn('theta_est_list', result)
            self.assertIn('pi_list', result)
            self.assertIn('coverage_list', result)

if __name__ == '__main__':
    unittest.main()
