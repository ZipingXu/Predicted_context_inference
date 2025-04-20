#!/bin/bash
# SLURM parameters
PART=murphy,shared             # Partition names
MEMO=40960                     # Memory required (40GB)
TIME=48:00:00                  # Time required (48 hours)

module load python/3.10.9-fasrc01

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Running with arguments: $@"

# Run the Python script with all arguments passed to this script
python run_exp.py "$@"

# Print job completion information
echo "Job finished at: $(date)"
echo "Exit code: $?" 