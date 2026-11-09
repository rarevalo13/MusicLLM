#!/bin/bash
#SBATCH --job-name=download-genres
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
##SBATCH --partition=<cpu-partition>

# Download balanced per-genre subsets (default: rock + hiphop, 1500 each) into
# genre_tracks/<genre>/. Resumable: resubmit the same command to continue where
# it left off; existing MP3s are never rewritten.
#
# Usage:  sbatch download_genres.slurm
# Smoke:  sbatch download_genres.slurm --per-genre 10 --out genre_tracks_smoke
# Extra args are appended after the defaults, so they override them.
# Set VENV to the venv to activate (default: .venv in the submit dir).
# Submit from persistent storage -- node-local scratch is wiped between jobs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv}/bin/activate"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CACHE="${TMPDIR:-/tmp}/hf_xet_cache"

python download_genres.py --genres rock hiphop --per-genre 1500 --out genre_tracks \
    --workers "${SLURM_CPUS_PER_TASK:-4}" --scratch "${TMPDIR:-/tmp}" "$@"
