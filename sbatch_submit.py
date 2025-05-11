import os
import itertools

log_dir = "./logs/all_5000/"
# Default parameters
default_params = {
    "-l": 0.1,
    "--sigma_eta": 0.01,
    "--p0": 0.05,
    "--sigma_s": 0.1,  # Default for random failure type
    "--coverage_freq": 500,
    "--n_rep": 5000,
    "--seed": 42,
    "--save_path": "./runs/all_5000/",
    "-T": 2500,
    "--name": ""
}

# Define experiments for each failure type
experiments = {
    "failure1": [
        # Experiment 1: vary learning rate
        # {"-l": [0.01, 0.1, 1.0], "--env": "failure1"},
        # Experiment 2: vary sigma_eta
        {"--sigma_eta": [0.0, 0.01, 0.1], "--env": "failure1"},
        # Experiment 3: vary p0
        # {"--p0": [0.01, 0.05, 0.1, 0.2], "--env": "failure1"},
        # Experiment 4: vary gamma
        # {"--gamma": [1.0, 10.0, 100.0], "--env": "failure1"},
    ],
    "failure2": [
        # Experiment 1: vary learning rate
        # {"-l": [0.01, 0.1, 1.0], "--env": "failure2"},
        # Experiment 2: vary sigma_eta
        {"--sigma_eta": [0.0, 0.01, 0.1], "--env": "failure2"},
        # Experiment 3: vary p0
        # {"--p0": [0.01, 0.05, 0.1, 0.2], "--env": "failure2"},
        # Experiment 4: vary gamma
        # {"--gamma": [1.0, 10.0, 100.0], "--env": "failure2"},
    ],
    "random": [
        # Experiment 1: vary learning rate
        # {"-l": [0.01, 0.1, 1.0], "--env": "random"},
        # Experiment 2: vary sigma_eta
        {"--sigma_eta": [0.0, 0.01, 0.1], "--env": "random"},
        # Experiment 3: vary p0
        # {"--p0": [0.01, 0.05, 0.1, 0.2], "--env": "random"},
        # Experiment 4: vary sigma_s (only for random)
        # {"--sigma_s": [0.01, 0.1, 1.0], "--env": "random"},
        # Experiment 5: vary gamma
        # {"--gamma": [1.0, 10.0, 100.0], "--env": "random"},
    ]
}

def create_sbatch_command(params):
    """Create an sbatch command with the given parameters."""
    name = params["--name"]
    PART="murphy,shared"             # Partition names
    MEMO=40960                     # Memory required (40GB)
    TIME="48:00:00"                  # Time required (48 hours)
    EMAIL="zipingxu@fas.harvard.edu"
    out_file = f"{log_dir}/{params['--env']}_{name}.out"
    err_file = f"{log_dir}/{params['--env']}_{name}.err"
    
    # ORDP="sbatch --mem=$MEMO -n 1 -p $PART --time=$TIME --mail-user=$EMAIL"
    cmd = "sbatch -o %s -e %s --mem %s -p %s --time %s --mail-user %s sbatch_run_exp.sh "%(out_file, err_file, MEMO, PART, TIME, EMAIL)
    
    
    
    # Add all parameters to the command

    for key, value in params.items():
        cmd += f" {key} {value}"
    
    return cmd

def run_experiments():
    """Run all experiments for all failure types."""
    for failure_type, experiment_list in experiments.items():
        print(f"Running experiments for {failure_type}")
        
        for experiment in experiment_list:
            # Get the parameter to vary and its values
            param_to_vary = list(experiment.keys())[0]
            if param_to_vary.startswith("--failure"):
                param_to_vary = list(experiment.keys())[1]
            
            values = experiment[param_to_vary]
            fixed_params = {k: v for k, v in experiment.items() if k != param_to_vary}
            
            print(f"  Varying {param_to_vary} with values {values}")
            
            # Run each variation
            for value in values:
                # Start with default parameters
                run_params = default_params.copy()
                
                # Update with fixed parameters for this experiment
                run_params.update(fixed_params)
                
                # Set the parameter we're varying
                run_params[param_to_vary] = value

                run_params["--name"] = f"{param_to_vary.split('-')[-1]}_{value}"
                if os.path.exists(f"{log_dir}/{run_params['--env']}_{run_params['--name']}.out"):
                    print(f"    Skipping {run_params['--env']}_{run_params['--name']} because it already exists")
                    continue
                # Create and execute the sbatch command
                cmd = create_sbatch_command(run_params)
                print(f"    Running: {cmd}")
                os.system(cmd)

if __name__ == "__main__":
    run_experiments() 