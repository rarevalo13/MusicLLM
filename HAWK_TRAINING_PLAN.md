# Training the rock and hip hop LoRAs on Hawk

This is the step-by-step for training one LoRA per genre on Hawk, from 3,000 JamendoMaxCaps tracks
each, plus the one-time Hugging Face setup. Everything runs on the cluster: no Colab, Google Drive
or external SSD needed. Run every command on Hawk from `~/MusicLLM` unless a step says otherwise.

## At a glance

| Step | What | Runs on | Time |
|---|---|---|---|
| 0 | One-time setup: update the code, Hugging Face login, finish the Python environment | login node | ~15 min |
| 1 | Training smoke test (proves training works, measures its speed) | `gpu-short` | ~10 min |
| 2 | Download 3,300 tracks per genre | CPU job (`day-long`) | hours (network-bound) |
| 3 | Set aside 300 tracks per genre for evaluation | login node | seconds |
| 4 | Cut 30 s clips and encode them, one job per genre | 2 GPU jobs (`gpu-day-long`) | ~1–2 h each |
| 5 | Train the two LoRAs, one job per genre | 2 GPU jobs (`gpu-day-long`) | from step 1's speed |
| 6 | Listen to base vs LoRA, copy the results to your Mac | `gpu-short` + Mac | ~15 min |
| 8 | Generate the evaluation set (base vs LoRA on every held-out clip) | 4 GPU jobs (`gpu-day-long`) | smoke-test seconds per clip × ~2,200 per genre |

Steps 1 and 2 can run at the same time; in steps 4 and 5 the rock and hip hop jobs run side by side.

**Disk use in your Hawk home:** MP3s ~17 GB, held-out clips ~12 GB, latents ~7.5 GB, checkpoints
~0.5 GB. The training clips (~60 GB per genre) only exist temporarily during step 4, normally on
the GPU node's local disk.

**Why it's set up this way**

- Hawk runs RHEL 7.9, whose system C library (glibc 2.17) is too old for PyTorch 2.7.1. Every step
  that uses Stable Audio therefore runs inside an Ubuntu 22.04 container through
  `./hawk_container.sh`. The container (`containers/ubuntu-22.04/`) and its Python environment
  (`stable-audio-3/.venv/`) already exist in your Hawk home; neither is in git.
- The download runs in your conda env `musicllm`, which already worked for the 10-track smoke test.
- **Held-out tracks:** 300 tracks per genre, from whole artists, are never trained on. They are the
  real-music reference for the FAD evaluation. That's why the download asks for 3,300 per genre: about
  3,000 are left for training.

## Step 0: one-time setup

### 0.1 Connect

```bash
ssh hawk
cd ~/MusicLLM
```

If `ssh hawk` hangs or times out, check your VPN or campus network connection first.

### 0.2 Update the code

```bash
git stash          # sets aside your local edit to download_genres.slurm; the new version includes it
git pull
git submodule update --init --recursive
```

The new `download_genres.slurm` activates the conda env `musicllm` itself, so the stashed edit is no
longer needed. Once step 2 runs, you can delete it with `git stash drop`.

### 0.3 Hugging Face

Training needs your Hugging Face account. The base model it trains on, `small-music-base`, loads its
text encoder from the gated `stabilityai/stable-audio-3-small-music` repository, and the comparison
in step 6 uses `small-music` itself. The SAME-S encoder (step 4) and the dataset aren't gated.

1. **Check that your account has access.** Log in at huggingface.co and open
   https://huggingface.co/stabilityai/stable-audio-3-small-music. If it says you have been granted
   access, you're done; `small-music` already runs on your Mac, so this is probably the case.
   Otherwise accept the Stability AI Community License form on that page (approval is automatic).
2. **Create a token.** Go to https://huggingface.co/settings/tokens, choose "Create new token", type
   **Read**, and name it, say, `hawk`. Copy the `hf_...` string. Don't paste it into any file,
   notebook or chat.
3. **Log in on Hawk:**
   ```bash
   ./hawk_container.sh hf auth login
   ```
   Paste the token when asked (nothing appears as you type). Answer `n` if it asks about adding the
   token as a git credential.
4. **Check:**
   ```bash
   ./hawk_container.sh hf auth whoami      # prints your Hugging Face username
   ```

The token is saved in `~/.cache/huggingface/token` on Hawk, and every command run through
`hawk_container.sh`, including the batch jobs, uses it from then on. To remove it:
`./hawk_container.sh hf auth logout`.

### 0.4 Finish the Python environment

```bash
./hawk_container.sh ~/.local/bin/uv pip install --python stable-audio-3/.venv/bin/python \
    pandas==2.3.3 pyarrow==25.0.1 matplotlib==3.10.8 wandb==0.31.0
```

- `pandas` and `pyarrow`: `prepare_lora_data.py` needs them to read the track metadata.
- `matplotlib` and `wandb`: `train_lora.py` imports them, although the project doesn't list them.
- If they are already installed, the command just reports that and changes nothing.
- Don't run `uv sync` in `stable-audio-3/` from now on: it removes packages that aren't in its lock
  file. If you ever do, run this command again.

### 0.5 Check the GPU

```bash
srun -p gpu-short --gres=gpu:1 -c 2 --mem=8G -t 10 ./hawk_container.sh \
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `2.7.1+cu126 True NVIDIA A100-PCIE-40GB`.

## Step 1: training smoke test

Before spending hours on data, check that training works on Hawk. This uses 200 clips that are
already encoded in `~/MusicLLM_tests/latents200_gpu`:

```bash
sbatch -p gpu-short -t 30 train_lora.slurm rock --encoded_dir ~/MusicLLM_tests/latents200_gpu \
    --steps 200 --checkpoint_every 200 --save_dir ../runs/smoke
squeue -u $USER                  # repeat until the job is gone
ls runs/smoke                    # expect epoch=...-step=200.ckpt (about 41 MB)
```

- The first run also downloads the text encoder, once.
- `sbatch` prints the job ID. Its log is `slurm-train-lora-<jobid>.out`; the end of the log shows
  any error.
- **Write down the training speed** that this command prints:
  ```bash
  tr '\r' '\n' < slurm-train-lora-<jobid>.out | grep -oE '[0-9.]+(it/s|s/it)' | tail -1
  ```
  Step 5 runs 6,000 steps, so its duration in hours is 6000 ÷ (it/s) ÷ 3600; if the speed is shown
  in s/it, it's 6000 × (s/it) ÷ 3600. For example, 1.0 it/s means about 1.7 hours.

## Step 2: download 3,300 tracks per genre

```bash
sbatch download_genres.slurm             # rock + hip hop, 3,300 each, into genre_tracks/
tail -f slurm-download-genres-<jobid>.out    # Ctrl-C stops watching, not the job
```

- Each log line is one dataset file: tracks kept, progress per genre, and GB downloaded so far.
  To check progress from the files themselves (works even if the log lags):
  `wc -l < genre_tracks/done_shards.txt; ls genre_tracks/rock | wc -l; ls genre_tracks/hiphop | wc -l`
- **Expected,** from a simulation of the script on the dataset's metadata: about 640–700 of the
  2,272 dataset files are scanned, so about 300–330 GB passes through the node's local disk, of which
  ~17 GB of MP3s are kept (rock ~9 GB, hip hop ~7.5 GB). Hip hop fills up last.
- **If the job stops** (the 1-day time limit, a node failure), submit the same command again: it
  picks up where it left off.
- **Done** when the log ends with `Done:` and `rock: 3300/3300` and `hiphop: 3300/3300`. Check:
  ```bash
  ls genre_tracks/rock | wc -l; ls genre_tracks/hiphop | wc -l      # 3300 each
  du -sh genre_tracks
  ```

## Step 3: set aside 300 tracks per genre for evaluation

```bash
./hawk_container.sh python split_holdout.py --tracks genre_tracks --per-genre 300
```

- It picks whole artists at random (fixed seed) until each genre has at least 300 held-out tracks,
  so no artist appears in both sets. It prints the training and held-out counts per genre.
- It creates `genre_splits/<genre>/train/` and `genre_splits/<genre>/heldout/` as folders of links
  to the MP3s in `genre_tracks/`; nothing is moved or copied. `train.csv` and `heldout.csv` list the
  tracks.
- Run it once, before step 4. Running it again keeps the same held-out artists. Don't change the
  split afterwards: the evaluation only means something if the held-out tracks were never trained on.

## Step 4: cut clips and encode them

```bash
sbatch make_latents.slurm rock
sbatch make_latents.slurm hiphop
```

Each job:
1. cuts the genre's training tracks into 30 s clips with captions (`prepare_lora_data.py`: up to 4
   per track, skipping the first 10 s), on the GPU node's local disk (or in `~/MusicLLM` if that disk
   has less than 100 GB free);
2. cuts the held-out tracks the same way into `heldout_clips/<genre>/`, kept as WAV for evaluation;
3. encodes the training clips with the SAME-S autoencoder into `<genre>_latents/` (one `.npy` and
   one `.json` per clip, plus `silence.npy`), which is what training reads;
4. checks that every clip was encoded, then deletes the temporary clips.

- **Expected:** about 11,000 clips per genre. Encoding took 0.27 s per clip on an A100 in testing,
  about 50 minutes per genre; cutting comes on top of that.
- **Done** when the log (`slurm-make-latents-<jobid>.out`) ends with a line like
  `rock: 11100 clips -> 11100 latents in rock_latents/`. Check:
  ```bash
  du -sh rock_latents hiphop_latents                 # about 3.7 GB each
  grep -o '"prompt": "[^"]*"' rock_latents/0000000000.json   # the first clip's caption
  ```
- **If a job fails,** it won't overwrite existing latents. Delete only that genre's latents, e.g.
  `rm -rf rock_latents`, and submit it again.

## Step 5: train

```bash
sbatch train_lora.slurm rock
sbatch train_lora.slurm hiphop
```

**Settings** (the defaults in `train_lora.slurm`): base model `small-music-base`, rank 16,
`dora-rows`, `--exclude seconds_total`, 30 s clips, batch size 4, 6,000 steps (about two passes over
~11,000 clips), a checkpoint every 1,000 steps. These match the Colab smoke test, apart from the
batch size and number of steps.

**Watching it:**
- `ls runs/rock` shows a new checkpoint every 1,000 steps, named like `epoch=2-step=6000.ckpt`
  (about 41 MB each, LoRA weights only).
- The loss for every step is in `runs/rock/lightning_logs/version_0/metrics.csv`.
- The time it takes follows from the speed you measured in step 1.

**If a job stops early,** continue from its last checkpoint for the remaining steps. The step counter
restarts at 0 and the optimizer starts fresh, so use a new folder:
```bash
sbatch train_lora.slurm rock --lora_checkpoint ../runs/rock/epoch=1-step=4000.ckpt \
    --steps 2000 --save_dir ../runs/rock_resume
```

**Other settings:** append any `train_lora.py` option, e.g. `--steps 8000`, or for the rank ablation
`--rank 32 --save_dir ../runs/rock_r32`. Use a new `--save_dir` for every run so checkpoints don't
mix. A run longer than a day needs `sbatch -p gpu-week-long -t 3-00:00:00 train_lora.slurm ...`.

## Step 6: listen and compare

**On Hawk,** generate three prompts per genre with the LoRA off and on, using the same seed (a few
minutes each). Use `ls runs/rock` for the exact checkpoint name:

```bash
srun -p gpu-short --gres=gpu:1 -c 4 --mem=16G -t 30 ./hawk_container.sh python compare_lora.py \
    --genre rock --lora runs/rock/epoch=2-step=6000.ckpt --out lora_comparison/rock
srun -p gpu-short --gres=gpu:1 -c 4 --mem=16G -t 30 ./hawk_container.sh python compare_lora.py \
    --genre hiphop --lora runs/hiphop/epoch=2-step=6000.ckpt --out lora_comparison/hiphop
```

Each writes `prompt0_base.wav`, `prompt0_lora.wav` and so on.

**On your Mac,** copy the comparisons and checkpoints, then try both LoRAs in the web demo:

```bash
cd ~/Developer/pythonprojects/MusicLLM
mkdir -p checkpoints
scp -r hawk:MusicLLM/lora_comparison ./lora_comparison_hawk
scp "hawk:MusicLLM/runs/rock/epoch=2-step=6000.ckpt" checkpoints/rock.ckpt
scp "hawk:MusicLLM/runs/hiphop/epoch=2-step=6000.ckpt" checkpoints/hiphop.ckpt
source stable-audio-3/.venv310-clean/bin/activate
python website/website.py --lora rock=checkpoints/rock.ckpt --lora hiphop=checkpoints/hiphop.ckpt
```

Then open http://127.0.0.1:8000. Checkpoints and WAV files are ignored by git.

## Step 7: clean up (optional)

- `rm -rf ~/MusicLLM_tests` removes the test files from the cluster setup (~0.1 GB). Do it after
  step 1, which uses them.
- Keep `genre_tracks/`, `genre_splits/`, `heldout_clips/`, the `*_latents/` folders and `runs/`
  until the project is graded: the evaluation and any retraining need them.
- `make_latents.slurm` deletes its own temporary clips. If a job was killed, remove any leftover
  `<genre>_clips_<jobid>/` folder from the repo.

## Step 8: generate the evaluation set

This makes the main evaluation set: for every held-out clip in `heldout_clips/<genre>/`, one
clip with the LoRA off and one with it on, from the clip's own caption with the same seed.

```bash
git pull     # picks up generate_eval.py + generate_eval.slurm
```

**Smoke test first** (a few minutes on `gpu-short`):

```bash
sbatch -p gpu-short -t 20 generate_eval.slurm rock --limit 4 --out eval_gen_smoke
```

When it finishes, check the log (`slurm-generate-eval-<jobid>.out`): the per-clip lines show the
seconds each generation took, and `eval_gen_smoke/rock/` should hold `manifest.csv` (8 rows) and 4
WAVs in each of `base/` and `lora/`. Then delete it: `rm -rf eval_gen_smoke`.

**Full run** — one job per genre, all four side by side:

```bash
for g in rock hiphop classical house; do sbatch generate_eval.slurm $g; done
```

Each job uses that genre's `runs/<genre>/*-step=6000.ckpt` automatically. **Watching it:**
`tail -f slurm-generate-eval-<jobid>.out` shows one line per clip, and
`ls eval_gen/rock/lora | wc -l` vs `ls heldout_clips/rock/*.wav | wc -l` shows the progress
(numbers should match when done). **If a job stops,** submit the same command again: existing
WAVs are skipped and it picks up where it left off. **Done** when the log ends with
`complete: every held-out clip has both files`.

**Expected:** about 1,100 clips per genre per condition at ~5.3 MB each — ~12 GB per genre,
~48 GB total. Time per genre ≈ (smoke-test seconds per clip) × ~2,200 generations. Leave the
WAVs on Hawk; the evaluation runs there too.

## After training

Not part of this plan: the metrics computed on the step 8 clips — FAD between generated clips
and `heldout_clips/<genre>/`, CLAP prompt adherence, the blind A/B listening test — plus latency
and VRAM measurements, and the ablations (rank 8/32, smaller training sets).

## Reference

### Partitions

| Partition | Nodes | Time limit |
|---|---|---|
| `short` (default) | CPU | 2 h |
| `day-long`, `week-long` | CPU | 1 day, 7 days |
| `gpu-short` | A100 | 2 h |
| `gpu-day-long`, `gpu-week-long` | A100 | 1 day, 7 days |

The `.slurm` files choose their partition; `sbatch -p <partition> -t <time> ...` overrides it.

### Job commands

```bash
squeue -u $USER                                        # your queued and running jobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # how a finished job went
scancel <jobid>                                        # stop a job
tail -f slurm-<name>-<jobid>.out                       # follow a job's log
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ssh hawk` hangs or times out | Check your VPN or campus network connection |
| `git pull` asks for a GitHub password or says "Permission denied (publickey)" | The repo is public, so switch to its HTTPS address: `git remote set-url origin https://github.com/rarevalo13/MusicLLM.git`, then pull again |
| `gated repo`, `401` or "Please log in" | Step 0.3 |
| `No module named 'pandas'` (or `pyarrow`, `matplotlib`, `wandb`) | Step 0.4 |
| CUDA not available, or "Found no NVIDIA driver" | The command didn't get a GPU: use a `gpu-*` partition with `--gres=gpu:1` (the `.slurm` files do) |
| `make_latents.slurm`: "already has files" | Delete that genre's `<genre>_latents/` and resubmit |
| Download job stopped | Submit it again; it resumes |

### What's where (in `~/MusicLLM` on Hawk)

| Path | Contents |
|---|---|
| `containers/ubuntu-22.04/` | The container |
| `stable-audio-3/.venv/` | Python environment (PyTorch 2.7.1 for CUDA 12.6) |
| `genre_tracks/` | Downloaded MP3s and `manifest.csv` |
| `genre_splits/` | Training and held-out track lists (links) |
| `heldout_clips/<genre>/` | Held-out 30 s clips for evaluation |
| `<genre>_latents/` | Encoded training clips |
| `runs/<genre>/` | Checkpoints and training logs |
| `lora_comparison/` | Base vs LoRA comparison WAVs |
| `eval_gen/<genre>/` | Evaluation set: `base/` + `lora/` WAVs and `manifest.csv` |
| `~/.cache/huggingface/` | Downloaded models and your Hugging Face token |

### How the container was built

Hawk's Singularity 3.5.3 can't pull current Docker images, so the container is a sandbox (a plain
directory) made from Ubuntu's official 22.04 base image, with CA certificates added. The Python
environment was installed with `uv sync --extra lora` in `stable-audio-3/`, plus the packages in
step 0.4. Singularity's `--nv` option alone didn't make the NVIDIA driver visible in this image, so
`hawk_container.sh` also binds the driver libraries into it.
