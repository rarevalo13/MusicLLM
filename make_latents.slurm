#!/bin/bash
#SBATCH --job-name=make-latents
#SBATCH --partition=gpu-day-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

# Cut one genre's training MP3s into 30 s clips, pre-encode them to latents and
# save the held-out clips as WAV for evaluation. One genre per job:
#   sbatch make_latents.slurm rock      sbatch make_latents.slurm hiphop
# Needs split_holdout.py first (genre_splits/<genre>/train|heldout).
#
# Outputs: <genre>_latents/ (.npy + .json per clip, plus silence.npy) for
# train_lora.slurm, and heldout_clips/<genre>/ (WAV + .txt captions) which is
# never trained on. Training clips are cut on node-local scratch (or the submit
# dir if scratch has <100 GB free) and deleted when the job ends.
# Refuses to run if <genre>_latents/ already has files: pre_encode_dataset.py
# numbers outputs from 0 on every run, so rerun into a clean directory.
# Submit from the repo root on Hawk.

set -euo pipefail
GENRE="${1:?usage: sbatch make_latents.slurm <genre>}"
cd "$SLURM_SUBMIT_DIR"
SPLIT="genre_splits/$GENRE"; OUT="${GENRE}_latents"
[ -d "$SPLIT/train" ] || { echo "no $SPLIT/train: run split_holdout.py first" >&2; exit 1; }
if [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT/ already has files: move or delete it first" >&2; exit 1; fi
SCRATCH="${TMPDIR:-/tmp}"
if [ "$(df -Pk "$SCRATCH" | awk 'NR==2 {print $4}')" -lt $((100 * 1024 * 1024)) ]; then SCRATCH="$PWD"; fi
CLIPS="$SCRATCH/${GENRE}_clips_$SLURM_JOB_ID"
trap 'rm -rf "$CLIPS"' EXIT
echo "$(date) training clips -> $CLIPS"; df -h "$SCRATCH" | tail -1
./hawk_container.sh python prepare_lora_data.py --audio "$SPLIT/train" --genre-tag "$GENRE" --out "$CLIPS"
./hawk_container.sh python prepare_lora_data.py --audio "$SPLIT/heldout" --genre-tag "$GENRE" --out "heldout_clips/$GENRE"
echo "$(date) encoding"
(cd stable-audio-3 && ../hawk_container.sh python scripts/pre_encode_dataset.py --model same-s \
    --data_dir "$CLIPS" --output_path "../$OUT" --sample_size 1327104)
n_wav=$(find "$CLIPS" -name '*.wav' | wc -l)
n_npy=$(find "$OUT" -name '*.npy' ! -name silence.npy | wc -l)
echo "$(date) $GENRE: $n_wav clips -> $n_npy latents in $OUT/"
[ "$n_wav" -eq "$n_npy" ] || { echo "MISMATCH: delete $OUT/ and resubmit" >&2; exit 1; }
