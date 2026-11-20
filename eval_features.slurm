#!/bin/bash
#SBATCH --job-name=eval-features
#SBATCH --partition=gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# Evaluation features (eval_features.py) for one genre: the held-out clips plus every generated
# set (eval_gen, eval_gen_steps4/16/32/50, eval_gen_base50) -> eval_feats/<set>/<genre>/*.npz.
# Run once generate_eval.slurm has finished all of that genre's sets (it refuses unfinished
# folders). Existing feature files are kept, so resubmitting only does what's missing.
# Then compute the metrics on a CPU node with eval_metrics.py.
#
# Usage:  sbatch eval_features.slurm rock
#         sbatch eval_features.slurm rock --sets heldout_clips eval_gen     # only these sets
# Submit from the repo root on Hawk.

set -euo pipefail
GENRE="${1:?usage: sbatch eval_features.slurm <genre> [eval_features.py args]}"
shift
cd "$SLURM_SUBMIT_DIR"
./hawk_container.sh python eval_features.py --genre "$GENRE" --workers 6 "$@"
