# AGENTS.md — MusicLLM

Context and instructions for AI coding agents working in this repository.

## Project overview

Final project for **CSIT 559 GenAI** (Montclair State). Goal: deploy, adapt and evaluate a
foundation model locally.

- **Application:** genre-specialized *instrumental* music generator (no vocals, no lyrics).
  Users describe music in text and get original tracks.
- **Core experiment:** LoRA fine-tune **Stable Audio 3 Small** on specific genres
  (rock, hip hop; classical and house added Nov 19) and show the adapted model beats the base model
  on those genres.
- **Deadline:** final submission + 15-min presentation due **Nov 30, 2026**
  (Thanksgiving Nov 26; aim to finish before it).
- **Required deliverables:** organized `code/` folder with source, setup instructions,
  dependency list and demo instructions; slides; final demo. The presentation must cover
  model choice, deployment setup (model/version, size, runtime, hardware, memory,
  latency, optimizations), adaptation method, example outputs, evaluation, limitations.

## Key decisions (don't change without asking the user)

- **Model:** Stable Audio 3 Small (~459M params, instrumental-only, official LoRA support,
  Stability AI Community License). Fallback: MusicGen-small via audiocraft.
- **Fine-tune with LoRA, not QLoRA, not from scratch.** The model already fits easily in GPU
  memory; quantization is covered as an inference/deployment experiment instead.
- **Dataset:** `amaai-lab/JamendoMaxCaps` on Hugging Face (~362k instrumental tracks, ~1 TB,
  metadata in JSONL files, audio in parquet shards, joined by track ID = MP3 filename).
- **Optional add-on:** small LLM (e.g. Qwen3-4B via Ollama) that rewrites casual requests
  into captions matching the training caption style. Ollama cannot run Stable Audio.

## Repository layout

| File | Purpose | Runs on |
|---|---|---|
| `explore_genres.py` | Downloads JamendoMaxCaps metadata only; prints genre/instrument/license counts; writes `jamendo_metadata.parquet` | Mac or cluster |
| `select_tracks.py` | `--genres X Y` → `selected_ids.txt` + `selected_metadata.csv` (instrumental, duration-filtered) | Mac |
| `download_sample.py` | Streams dataset, saves first N matching tracks (`--ids`); skips existing files; resumes via `<out>/stream_state.pkl` (`--restart` rescans from row 0) | Mac (small N) |
| `download_genres.py` | Balanced per-genre quotas (`--genres rock hiphop --per-genre 1500`) into `out/<genre>/`; skips tracks tagged with multiple requested genres; 60–600 s; `--max-per-artist 20`; walks shards in seeded random order; resumable via `manifest.csv` + `done_shards.txt` | Cluster |
| `download_genres.slurm` | Batch job wrapper for `download_genres.py` (rock + hiphop, 3,300 each; partition `day-long`; activates conda env `musicllm`, override with `CONDA_ENV`); extra args override its defaults; resubmit to resume | Cluster CPU nodes |
| `extract_audio.py` + `.slurm` | Full-scale extraction of `selected_ids.txt`: job array, 4 shards per task (2,272 shards → `--array=0-567`) | Cluster CPU nodes |
| `prepare_lora_data.py` | Cuts MP3s into 30 s stereo 44.1 kHz WAV chunks + matching `.txt` captions from Jamendo tags (`--genre-tag` also sets the default `--label` and `--out`; `--max-chunks`) | Mac or cluster |
| `split_holdout.py` | Holds out whole artists per genre (`--per-genre 300`, seeded) before clips are cut: `genre_splits/<genre>/{train,heldout}/` = relative symlinks into `genre_tracks/` + `train.csv`/`heldout.csv`. Never moves MP3s (`download_genres.py` drops manifest rows with missing MP3s and refills quotas); re-runs keep the held-out artists | Cluster (login node) |
| `make_latents.slurm` | GPU job per genre: cuts `genre_splits/<genre>/train` to node-local scratch and the held-out tracks to `heldout_clips/<genre>/` (WAV), pre-encodes with `same-s` (`--sample_size 1327104`) into `<genre>_latents/`, checks clip/latent counts; refuses a non-empty `<genre>_latents/` | Cluster GPU |
| `train_lora.slurm` | GPU job per genre: `train_lora.py` on `<genre>_latents/` (small-music-base, rank 16, dora-rows, batch 4, 6,000 steps, checkpoint every 1,000) → `runs/<genre>/`; extra args override | Cluster GPU |
| `hawk_container.sh` | Runs a command in the Ubuntu 22.04 Singularity sandbox (`containers/ubuntu-22.04`) with `stable-audio-3/.venv/bin` first on PATH; binds the NVIDIA driver libraries on GPU nodes | Cluster |
| `HAWK_TRAINING_PLAN.md` | The user's step-by-step for the full Hawk run, including the Hugging Face setup | — |
| `compare_lora.py` | Same prompts with LoRA strength 0 vs 1, same `seed=`, for A/B listening; `--genre rock\|hiphop\|classical\|house` picks the built-in prompts. On Hawk run from the repo root as `./hawk_container.sh python compare_lora.py`; with uv, from `stable-audio-3/` as `../compare_lora.py` | Cluster GPU |
| `stable_audio_test.py` | One-off text-to-audio test with `small-music` | Mac |
| `website/website.py` + `static/index.html` | Tyler's Flask web demo on `small-music`: prompt → WAV; `--lora NAME=PATH` (repeatable) adds a base/LoRA selector and same-seed side-by-side compare; seed box; serves `127.0.0.1:8000` (`--host`, `--port`) | Mac or cluster GPU |

**Status (Nov 8):** data scripts smoke-tested locally against the real dataset
(`download_sample.py` resume/skip; `download_genres.py` on a few shards: quotas, genre
exclusivity, resume, refill; `prepare_lora_data.py` rock + hip hop captions). The `.slurm`
files were only dry-run with a stub `python`. `train_lora.py` and `compare_lora.py` have only run
on Colab so far (Nov 15, below), not on the cluster. Verify on the cluster before scaling up.
**Nov 15:** 100-step rock smoke test on Colab (A100 80GB) in `lora_training.ipynb`:
`train_lora.py` on the 20 clips in `rock_lora_smoke/` (rank 16, `dora-rows`,
`--exclude seconds_total`, 10.7M trainable params) → `epoch=4-step=100.ckpt` (41 MB), then
`compare_lora.py` → 3 base/LoRA pairs. The checkpoint and comparisons are on Google Drive in
`MusicLLM_runs/lora_rock_smoke/`, not in the repo.
**Nov 17:** Tyler's website merged into main as a squash commit (authored by him), then fixed:
`small-music` defaults (8 steps, CFG 1), 120 s duration cap, seed box, LoRA selector + base
comparison, port 8000, complete `website/REQUIREMENTS.txt`. Its LoRA option has only been tested
with a fake model; try it next with the Nov 15 checkpoint.
**Nov 17 (later):** Hawk inspected over SSH (facts under Environments). Built an Ubuntu 22.04
Singularity sandbox and the Stable Audio env inside it. `pre_encode_dataset.py` on 200 clips on an
A100: 0.27 s per clip steady state with `--sample_size 1327104`; latents (256, 323) float32 with the
caption in each `.json`. `train_lora.py` got as far as loading the model, then stopped: `small-music-base` loads its
T5-Gemma text encoder from the gated `small-music` repo, so Hawk needs `hf auth login`.
**User decision: train both genres on Hawk, not Colab.** Added `split_holdout.py`,
`make_latents.slurm`, `train_lora.slurm`, `hawk_container.sh`; the user's walkthrough is
`HAWK_TRAINING_PLAN.md`.
**Nov 18: full Hawk run done** (the user followed the plan; jobs 64171–64192). Smoke test 200 steps:
2 min 53 s. Download (`day-long`, 8 CPUs): 733 shards, 344.4 GB in 52 min 50 s (~109 MB/s), more
than the simulation predicted → 3,300 rock (1,125 artists) + 3,300 hip hop (872 artists), 16 GB of
MP3s. Split: rock 2,995 train / 305 held out, hip hop 2,994 / 306. `make_latents.slurm` (~1 h per
genre, both on one node): cutting ~0.4–0.5 s per track, encoding ~0.22 s per clip; rock 11,236
training + 1,126 held-out clips, hip hop 10,836 + 1,139. The GPU node's `/tmp` had <100 GB free, so
the temporary clips (~117 GB together) went to the NFS home and were deleted afterwards.
`train_lora.slurm` (6,000 steps, batch 4, ~2.1–2.2 passes): ~1.75 it/s (7 clips/s), ~58 min per
genre; 10.7M trainable of 578M params; final `runs/<genre>/epoch=2-step=6000.ckpt` (43 MB).
`compare_lora.py` per genre: under a minute → `lora_comparison/<genre>/`. Total ≈4.2 A100-hours
(training 1.9). Hawk's SLURM 20.02 records no CPU/RAM/GPU usage (`jobacct_gather/none`), so peak
RAM and VRAM weren't measured.

## Environments

- **Local (macOS, M1 Pro, 16 GB):** the working env is `stable-audio-3/.venv310-clean`
  (Python 3.10; has `stable_audio_3`, datasets, pandas, pyarrow, soundfile) →
  `source stable-audio-3/.venv310-clean/bin/activate`. There is no root `.venv`.
  Data-script deps: `pip install -U huggingface_hub datasets pandas pyarrow numpy soundfile`.
  Port 5000 is taken by macOS AirPlay Receiver (ControlCenter), so the web demo defaults to
  8000. Flask isn't in `.venv310-clean` until you run `pip install -r website/REQUIREMENTS.txt`.
- **Stable Audio 3:** NOT on PyPI. Vendored as the git submodule `stable-audio-3/`
  (`git submodule update --init --recursive`). For training:
  ```bash
  cd stable-audio-3
  uv sync --extra lora      # lora extra (pytorch_lightning) required for training
  uv run hf auth login      # accept model licenses on Hugging Face first
  ```
  Run its tools with `uv run python ...` from inside `stable-audio-3/` (not on Hawk: see below).
- **Google Colab (fallback):** ran the Nov 15 smoke test and was briefly the planned training
  environment (with a paid Google Drive) until the Hawk container worked on Nov 17. Colab
  preinstalls matplotlib and wandb, which `train_lora.py` needs but the project doesn't list.
  Downloading on the Mac measured ~27 MB/s from Hugging Face on one stream.
  `pre_encode_dataset.py` uses CUDA or falls back to CPU (no MPS); on the Mac its DataLoader
  workers crash (`name 'Path' is not defined`, then endless retries) and the default
  `--sample_size` (~285 s) wastes memory: don't encode on the Mac.
- **Cluster: Hawk** (`ssh hawk`, user `arevalor`, repo `~/MusicLLM`); the training environment
  since Nov 17. Off campus, SSH can time out (it did from the Mac on Nov 17).
  - 48 CPU nodes (24 cores, 128 GB RAM, 340 GB local scratch): preprocessing, downloads.
  - 1 large-memory node (72 cores, 4 TB RAM): rarely needed.
  - 6 GPU nodes, each 4× A100 40GB PCIe (1.7 TB NVMe scratch): LoRA training.
    One GPU per run is plenty; use nodes for parallel experiments, not multi-node jobs.
    Driver 550.163.01 (CUDA 12.4).
  - 1 GPU node, 3× L40S 48GB: inference, evaluation, demo serving.
  - SLURM partitions: CPU `short` (default, 2 h), `day-long`, `week-long`, `month-long`,
    `interactive-cpu` (2 days); A100 `gpu-short` (2 h), `gpu-day-long`, `gpu-week-long`,
    `gpu-month-long`; also `l40s-*` and `highmem-*`. Compute nodes reach Hugging Face and PyPI.
  - **RHEL 7.9, glibc 2.17** (system Python 3.6): PyTorch 2.7.1 can't run natively. Stable Audio
    runs in an Ubuntu 22.04 Singularity sandbox, `containers/ubuntu-22.04` (not in git), via
    `./hawk_container.sh`. It was built from Ubuntu's base rootfs because Singularity 3.5.3 can't
    pull current OCI images and lacks mksquashfs; a CA bundle was added. There's no Apptainer.
    `--nv` alone doesn't expose the driver in the sandbox, so `hawk_container.sh` binds libcuda,
    libnvidia-ml, libnvidia-ptxjitcompiler and libnvidia-nvvm into `/usr/lib/x86_64-linux-gnu`.
  - Stable Audio env: `stable-audio-3/.venv` (`uv sync --extra lora` inside the container; uv's
    own Python 3.10; torch 2.7.1+cu126) plus pandas 2.3.3, pyarrow 25.0.1, matplotlib 3.10.8 and
    wandb 0.31.0 via `uv pip install` (HAWK_TRAINING_PLAN.md step 0.4). **Don't `uv sync` there:**
    it removes those. `uv` is at `~/.local/bin/uv`, not on PATH.
  - Download jobs use the conda env `musicllm` (Python 3.10, no torch). The old repo `.venv` is
    Python 3.7: unusable.
  - Home is NFS (`/nfs/storage1/home`, 26 TB shared, ~5.2 TB free on Nov 17). Node-local scratch is
    temporary: write results to persistent storage.

## Common commands

```bash
# Data
python explore_genres.py
python select_tracks.py --genres rock                 # selected_ids.txt = rock selection
python download_sample.py --n 30 --ids selected_ids.txt --out sample_tracks   # 29 rock tracks already there
python prepare_lora_data.py --audio sample_tracks --out rock_lora_data       # 30-track smoke-test data
python prepare_lora_data.py --audio genre_tracks/hiphop --genre-tag hiphop   # "Instrumental hip hop" → hiphop_lora_data/

# Hawk (submit from ~/MusicLLM; full walkthrough in HAWK_TRAINING_PLAN.md)
./hawk_container.sh hf auth login                                       # once
sbatch download_genres.slurm                                            # rock + hiphop, 3,300 each → genre_tracks/
./hawk_container.sh python split_holdout.py --tracks genre_tracks --per-genre 300
sbatch make_latents.slurm rock; sbatch make_latents.slurm hiphop        # → <genre>_latents/, heldout_clips/<genre>/
sbatch train_lora.slurm rock; sbatch train_lora.slurm hiphop            # → runs/<genre>/*.ckpt
srun -p gpu-short --gres=gpu:1 -c 4 --mem=16G -t 30 ./hawk_container.sh python compare_lora.py \
    --genre rock --lora runs/rock/<checkpoint>.ckpt --out lora_comparison/rock

# Training elsewhere (any CUDA machine with uv, inside stable-audio-3/)
uv run python scripts/train_lora.py \
    --model small-music-base --data_dir ../rock_lora_data \
    --rank 16 --adapter_type dora-rows --exclude seconds_total \
    --duration 30 --steps 1000 --save_dir ./lora_rock_test
uv run python ../compare_lora.py --lora lora_rock_test/<checkpoint>.ckpt

# Web demo (Mac: in .venv310-clean after pip install -r website/REQUIREMENTS.txt)
python website/website.py --lora rock=path/to/rock.ckpt          # http://127.0.0.1:8000
# Web demo (CUDA machine with uv, inside stable-audio-3/)
uv run --with flask==3.1.3 python ../website/website.py --lora rock=lora_rock_test/<checkpoint>.ckpt
```

## Stable Audio 3 API notes

```python
from stable_audio_3 import StableAudioModel
model = StableAudioModel.from_pretrained("small-music")   # post-trained: use for generation
model.load_lora(["path/to/lora.safetensors"])
model.set_lora_strength(1.0)                               # 0.0 disables without unloading
audio = model.generate(prompt="...", duration=30, seed=0)  # pass seed= for reproducible A/B
# audio shape: (1, channels, samples), 44.1 kHz stereo
sf.write("out.wav", audio.detach().cpu().squeeze(0).T.numpy(), 44100)
```

- `small-music` = post-trained checkpoint (fast, few steps). `small-music-base` = checkpoint
  intended for fine-tuning (train LoRAs on this; needs more steps if generating from it).
  LoRAs trained on `small-music-base` are applied to `small-music` at inference.
- `steps` = number of reverse (denoising) sampling steps at inference (default 8; use ~50 for
  `-base` models); `cfg_scale` = classifier-free guidance strength (default 1.0). Upstream docs say
  it has no effect on post-trained checkpoints, but the DiT applies CFG whenever `cfg_scale != 1.0`
  (`stable_audio_3/models/dit.py`), and on `small-music` it changes the output and slows generation:
  keep 1.0 there. Keep both fixed within any comparison.
- `generate()` re-seeds torch itself (`seed=-1` → random), so `torch.manual_seed()` before it
  does nothing: always pass `seed=`.
- `generate()` caps output at 120 s (`sample_size=5292032` samples at 44.1 kHz); a longer
  `duration` silently comes back as 120 s.
- Training data format: folder of audio files, each with a same-named `.txt` caption.
- `scripts/train_lora.py` flags: `--model` must be a `-base` model; output dir is `--save_dir`
  (upstream `docs/workflows/lora.md` shows `--output_dir`, which does not exist); `--duration`
  defaults to 380 s and pads every clip to that, so pass `--duration 30` for 30 s chunks.
  Defaults: `--steps 10000`, `--lr 1e-4`, `--batch_size 1`, `--checkpoint_every 500`,
  `--demo_every 500`, `--logger csv`. Checkpoints are Lightning `.ckpt` files holding only the
  LoRA weights + `lora_config`; `load_lora()` accepts `.ckpt` or `.safetensors`.
- `--exclude seconds_total` prevents conditioner hijacking on small datasets.
- Pre-encoding to latents is recommended for large datasets: for Small use
  `scripts/pre_encode_dataset.py --model same-s`, then train with `--encoded_dir`.
- `train_lora.py` requires an NVIDIA (CUDA) GPU; Small needs only ~2.5 GB VRAM.
  Medium requires Flash Attention 2 (not used here). An MLX LoRA trainer for Apple Silicon
  exists in `stable-audio-3/optimized/mlx/` (not part of the plan).

## Dataset facts (verified Nov 8, 2026)

- HF repo: 2,272 audio shards `data/train-XXXXX-of-02272.parquet` (1,061 GB total, median
  0.47 GB, ~160 tracks, 2 row groups each; schema `audio: struct<bytes, path>`) plus 1,170
  date-named metadata JSONL files and `final_caption30sec.jsonl` (787 MB; 6 GB of JSONL in
  total, all downloaded by `explore_genres.py`). Shards are ordered by track ID **as a string**, so
  neighbouring IDs (same artist/album) cluster in the same shard.
- `jamendo_metadata.parquet`: 394,198 tracks, all `vocal == "instrumental"`; 351,202 are 60–600 s.
- Tags: `rock` 28,963; `hiphop` 15,541 (`rap` 2,797 and `trap` 4,043 are separate tags);
  186 tracks carry both rock and hiphop.
- Eligible for `download_genres.py` (60–600 s, one requested genre only): rock 25,925 tracks /
  4,637 artists; hip hop 14,187 / 2,016 artists (one artist has 563 → per-artist cap matters).
- Full 1,500-per-genre job: hip hop is the bottleneck; estimated ~260–330 shards scanned
  (~120–155 GB through scratch), ~9 GB of MP3s kept.
- **Target: ~3,000 training tracks per genre plus ~300 held out (user decision, Nov 17), so the
  download asks for 3,300 per genre.** Simulating `download_genres.py` (shards = string-sorted IDs
  split evenly, `random.Random(0)` shard order, 20-per-artist cap): 3,300 per genre → ~636–691
  shards (~300–325 GB through scratch); MP3s ~9.2 GB rock (~1,090 artists) + ~7.5 GB hip hop
  (~865 artists). The simulation reproduces the real shard order (Hawk smoke-test log), but the real
  run met the 10-track quotas one shard later, so treat these numbers as estimates.
  With the default `--max-chunks 4`: ~12,200 rock / ~12,000 hip hop clips before the hold-out,
  ~11,100 / ~10,900 after; 5.29 MB of WAV and ~0.33 MB of fp32 `same-s` latent (256 × 323) per
  30 s clip, so ~3.7 GB of latents per genre. `--max-chunks 2` halves both.
  The trainer's latent loader scans subfolders, so several latent folders can be trained together
  under one `--encoded_dir`.
- **Extra genres: classical + house (user decision, Nov 19; house instead of the broad `electronic`
  tag).** Eligible with all four tags exclusive (60–600 s) → after the 20-per-artist cap: classical
  18,789 → 10,802 (2,161 artists), house 9,600 → 7,109 (1,910). Other tags after the cap: electronic
  ~39k, ambient 20,189, techno 6,962, jazz 5,619; metal 2,513 and lo-fi 1,240 are too few for 3,300.
  Download into `genre_tracks` with `--genres rock hiphop classical house`: full quotas are skipped,
  the manifest blocks duplicates and the 4-way exclusivity keeps rock/hip hop-tagged tracks out.
  House is the bottleneck: simulated 901 shards after the first run's 732 done shards, × 1.13 (first
  run: real 733 vs simulated 647) ≈ 1,020 shards / ~480 GB / ~75 min; ~1,070 house and ~835
  classical artists, ~12,600 / ~11,800 clips before the hold-out. 67% of house captions read
  "with electronic … influences". `READABLE` got 48 more joined tags on Nov 19 (every run-together
  English tag on ≥0.1% of eligible classical or house tracks: `acidhouse`, `progressivehouse`,
  `contemporarypiano`, …), changing 15% of house and 7% of classical captions; misspellings,
  non-English tags and artist/software names stay as they are.

## Rules and constraints

- **Never download the full ~1 TB dataset to the Mac.** Small samples locally; scale-out
  jobs on the cluster.
- **Keep downloads resumable and idempotent:** skip files that already exist; write the
  original MP3 bytes from parquet (never decode and re-encode).
- **Licensing:** JamendoMaxCaps tracks carry per-track Creative Commons licenses, many
  non-commercial → research/academic use only. Never scrape YouTube, Spotify or commercial
  catalogs. Cite the dataset in the presentation.
- **Captions must match prompt style:** training captions and inference prompts should use
  the same vocabulary (genre, influences, instruments, mood, tempo). Jamendo joins
  multi-word tags (`electricguitar`, `hiphop`); `prepare_lora_data.py` maps common ones
  to readable text.
- Ask the user before long-running or expensive actions (full downloads, multi-hour jobs,
  deleting data).

## Evaluation plan

- Fixed set of ~20 prompts generated with base vs LoRA (same seed, steps, cfg).
- FAD against held-out real tracks of the genre; CLAP score for prompt adherence.
- Blind A/B listening test with 10–15 classmates.
- Latency and peak VRAM for 30/60/120 s clips; steps sweep (4/8/16/32/50) for a
  latency-vs-quality curve; optional int8 quantization experiment.
- Ablations: LoRA rank (8/16/32), dataset size.
- Failure analysis: off-genre prompts, long-form structure, artifacts, overfitting
  (outputs copying training songs).
- Demo: `website/` (Flask) with prompt box, duration slider, seed, base/LoRA selector and
  same-seed side-by-side compare.

## Next steps

`HAWK_TRAINING_PLAN.md` steps 0–6 ran on Nov 18 (see Status). Next:

1. Listen to `lora_comparison/<genre>/`; try both LoRAs in the web demo (its first test with real
   checkpoints).
2. Classical + house on Hawk, same steps as the plan: `sbatch download_genres.slurm --genres rock
   hiphop classical house` → `split_holdout.py` (rock/hip hop keep their split) →
   `make_latents.slurm` and `train_lora.slurm` per genre → `compare_lora.py --genre classical|house`.
3. Evaluation pipeline (FAD vs `heldout_clips/<genre>/`, CLAP, blind A/B test); measure latency and
   peak VRAM (SLURM doesn't record it); ablations; finish slides before Thanksgiving.

## Open questions to verify

- Peak VRAM for training (batch 4) and generation: not recorded by Hawk's SLURM; measure with
  `torch.cuda.max_memory_allocated()` or by sampling `nvidia-smi` during a short run.
- Whether all Jamendo MP3s are 44.1 kHz: all 29 in `sample_tracks/` are 44.1 kHz stereo;
  check `prepare_lora_data.py` skip messages at scale (the trainer resamples on load anyway).
- JamendoMaxCaps captions: the HF repo has `final_caption30sec.jsonl` (787 MB). Its format hasn't
  been inspected; it might be a better caption source than the tag-built captions.
