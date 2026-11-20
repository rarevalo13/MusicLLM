#!/bin/bash
#SBATCH --job-name=generate-eval
#SBATCH --partition=gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

# Main evaluation set for one genre (generate_eval.py): for every held-out clip in
# heldout_clips/<genre>/, one clip with the LoRA off and one with it on, from the
# clip's caption with the same seed, into eval_gen/<genre>/{base,lora}/ + manifest.csv.
# Resumable: submit the same command again. One job per genre at a time (they share
# eval_gen/<genre>/manifest.csv). Uses runs/<genre>/*-step=6000.ckpt unless LORA is set.
#
# Usage:  sbatch generate_eval.slurm rock
#         sbatch generate_eval.slurm rock --rounds 1        # only each track's _00 clip
#         LORA=runs/rock/<file>.ckpt sbatch generate_eval.slurm rock
# Smoke:  sbatch -p gpu-short -t 20 generate_eval.slurm rock --limit 4 --out eval_gen_smoke
# Submit from the repo root on Hawk.

set -euo pipefail
GENRE="${1:?usage: sbatch generate_eval.slurm <genre> [generate_eval.py args]}"
shift
cd "$SLURM_SUBMIT_DIR"
if [ -z "${LORA:-}" ]; then
    shopt -s nullglob
    ckpts=(runs/"$GENRE"/*-step=6000.ckpt)
    # never expand "${ckpts[@]}" while it may be empty: bash 4.2 + set -u errors on it
    [ "${#ckpts[@]}" -eq 1 ] || { echo "expected one runs/$GENRE/*-step=6000.ckpt, found ${#ckpts[@]}: set LORA=<path>" >&2; exit 1; }
    LORA="${ckpts[0]}"
fi
./hawk_container.sh python generate_eval.py --genre "$GENRE" --lora "$LORA" "$@"
