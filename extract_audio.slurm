#!/bin/bash
#SBATCH --job-name=extract-audio
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
##SBATCH --partition=<cpu-partition>

# Job array for extract_audio.py: each array task processes PER_TASK shards
# (default 4), because SLURM MaxArraySize is often 1001 and there are 2,272
# shards. Usage:
#   N=$(python extract_audio.py --list)
#   sbatch --array=0-$(( (N + 3) / 4 - 1 ))%20 extract_audio.slurm --out /persistent/path/audio_subset
# (= array 0-567; %20 limits concurrent tasks). Rerun failures with
#   sbatch --array=<ids> extract_audio.slurm --out ...
# If you change PER_TASK (--export=ALL,PER_TASK=8), change the array math too.
# Set VENV to the venv to activate (default: .venv in the submit dir).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv}/bin/activate"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CACHE="${TMPDIR:-/tmp}/hf_xet_cache"

PER_TASK=${PER_TASK:-4}
N=$(python extract_audio.py --list)
for (( i = SLURM_ARRAY_TASK_ID * PER_TASK; i < (SLURM_ARRAY_TASK_ID + 1) * PER_TASK && i < N; i++ )); do
    python extract_audio.py --shard-index "$i" --scratch "${TMPDIR:-/tmp}" "$@"
done
