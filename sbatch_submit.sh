#!/bin/bash
# SLURM parameters
PART=murphy,shared             # Partition names
MEMO=40960                     # Memory required (40GB)
TIME=48:00:00                  # Time required (48 hours)
EMAIL="zipingxu@fas.harvard.edu"


module load python/3.10.9-fasrc01

RUN_SCRIPT="/n/home03/zipingxu/Predicted_context_inference/sbatch_run_exp.sh"
SCRIPT_DIR="/n/home03/zipingxu/Predicted_context_inference/logs"
ORDP="sbatch --mem=$MEMO -n 1 -p $PART --time=$TIME --mail-user=$EMAIL"

DATE="0420"

JOBN="failure1_${DATE}"
OUTF=$SCRIPT_DIR/"$JOBN.out"
ERRF=$SCRIPT_DIR/"$JOBN.err"
ORD="$ORDP -J $JOBN -o $OUTF -e $ERRF $RUN_SCRIPT --env failure1 --n_rep 1000 -T 2500 --p0 0.2 --lambda_ 0.1 --coverage_freq 500 --name $DATE"
$ORD

JOBN="failure2_${DATE}"
OUTF=$SCRIPT_DIR/"$JOBN.out"
ERRF=$SCRIPT_DIR/"$JOBN.err"
ORD="$ORDP -J $JOBN -o $OUTF -e $ERRF $RUN_SCRIPT --env failure2 --n_rep 1000 -T 2500 --p0 0.2 --lambda_ 0.1 --coverage_freq 500 --name $DATE"

$ORD

JOBN="random_${DATE}"
OUTF=$SCRIPT_DIR/"$JOBN.out"
ERRF=$SCRIPT_DIR/"$JOBN.err"
ORD="$ORDP -J $JOBN -o $OUTF -e $ERRF $RUN_SCRIPT --env random --n_rep 1000 -T 2500 --p0 0.2 --lambda_ 0.1 --coverage_freq 500 --name $DATE"

$ORD