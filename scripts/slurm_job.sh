#!/usr/bin/env bash
# Generic SLURM job: runs its arguments on one node with 4 GPUs.
# Override the defaults on the command line, e.g.
#   sbatch --account=MY_ACCOUNT --partition=boost_usr_prod scripts/slurm_job.sh python train.py ...
# Activate the Python environment before sbatch (the job inherits it) or below.
#SBATCH --job-name=globio
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=globio_%x_%j.out
set -euo pipefail
# source .venv/bin/activate        # or: conda activate globio
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SLURM_SUBMIT_DIR:-.}"
echo "$(date) $*"
"$@"
