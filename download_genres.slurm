#!/bin/bash
#SBATCH --job-name=download-genres
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=day-long

# Download balanced per-genre subsets (default: rock + hiphop, 3300 each:
# ~3,000 to train on + ~300 held out for evaluation) into
# genre_tracks/<genre>/. Resumable: resubmit the same command to continue where
# it left off; existing MP3s are never rewritten.
#
# Usage:  sbatch download_genres.slurm
# Smoke:  sbatch download_genres.slurm --per-genre 10 --out genre_tracks_smoke
# Extra args are appended after the defaults, so they override them.
# CONDA_ENV overrides the conda env to activate (default: musicllm).
# Submit from persistent storage -- node-local scratch is wiped between jobs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-musicllm}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CACHE="${TMPDIR:-/tmp}/hf_xet_cache"

python download_genres.py --genres rock hiphop --per-genre 3300 --out genre_tracks \
    --workers "${SLURM_CPUS_PER_TASK:-4}" --scratch "${TMPDIR:-/tmp}" "$@"
