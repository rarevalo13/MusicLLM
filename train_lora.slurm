#!/bin/bash
#SBATCH --job-name=train-lora
#SBATCH --partition=gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

# Train one genre's LoRA on small-music-base from <genre>_latents/ (made by
# make_latents.slurm). Checkpoints (LoRA weights only, ~41 MB) land every 1000
# steps in runs/<genre>/; CSV logs in runs/<genre>/lightning_logs/.
# Needs `./hawk_container.sh hf auth login` first: the base model's text encoder
# comes from the gated stabilityai/stable-audio-3-small-music repo.
#
# Usage:  sbatch train_lora.slurm rock
#         sbatch train_lora.slurm hiphop --steps 8000      # extra args override defaults
# Smoke:  sbatch -p gpu-short -t 30 train_lora.slurm rock --encoded_dir ~/MusicLLM_tests/latents200_gpu --steps 200 --checkpoint_every 200 --save_dir ../runs/smoke
# Resume: --lora_checkpoint ../runs/<genre>/<file>.ckpt --steps <remaining> --save_dir ../runs/<genre>_resume
#         (the step count restarts at 0 in the new run)
# Submit from the repo root on Hawk.

set -euo pipefail
GENRE="${1:?usage: sbatch train_lora.slurm <genre> [train_lora.py args]}"
shift
cd "$SLURM_SUBMIT_DIR/stable-audio-3"
../hawk_container.sh python scripts/train_lora.py \
    --model small-music-base --encoded_dir "../${GENRE}_latents" --save_dir "../runs/$GENRE" \
    --rank 16 --adapter_type dora-rows --exclude seconds_total --duration 30 \
    --batch_size 4 --steps 6000 --checkpoint_every 1000 --demo_every 1000000 \
    --num_workers 8 "$@"
