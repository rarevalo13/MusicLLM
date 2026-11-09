# AGENTS.md — MusicLLM

Context and instructions for AI coding agents working in this repository.

## Project overview

Final project for **CSIT 559 GenAI** (Montclair State). Goal: deploy, adapt and evaluate a
foundation model locally.

- **Application:** genre-specialized *instrumental* music generator (no vocals, no lyrics).
  Users describe music in text and get original tracks.
- **Core experiment:** LoRA fine-tune **Stable Audio 3 Small** on specific genres
  (rock, hip hop) and show the adapted model beats the base model on those genres.
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
| `download_genres.slurm` | Batch job wrapper for `download_genres.py`; extra args override its defaults; resubmit to resume | Cluster CPU nodes |
| `extract_audio.py` + `.slurm` | Full-scale extraction of `selected_ids.txt`: job array, 4 shards per task (2,272 shards → `--array=0-567`) | Cluster CPU nodes |
| `prepare_lora_data.py` | Cuts MP3s into 30 s stereo 44.1 kHz WAV chunks + matching `.txt` captions from Jamendo tags (`--genre-tag` also sets the default `--label` and `--out`; `--max-chunks`) | Mac or cluster |
| `compare_lora.py` | Same prompts with LoRA strength 0 vs 1, same `seed=`, for A/B listening; run from `stable-audio-3/` as `../compare_lora.py` | Cluster GPU |
| `stable_audio_test.py` | One-off text-to-audio test with `small-music` | Mac |

**Status (Nov 8):** data scripts smoke-tested locally against the real dataset
(`download_sample.py` resume/skip; `download_genres.py` on a few shards: quotas, genre
exclusivity, resume, refill; `prepare_lora_data.py` rock + hip hop captions). The `.slurm`
files were only dry-run with a stub `python`; `train_lora.py` and `compare_lora.py` have
**not been run yet** (need a CUDA GPU). Verify on the cluster before scaling up.

## Environments

- **Local (macOS, M1 Pro, 16 GB):** the working env is `stable-audio-3/.venv310-clean`
  (Python 3.10; has `stable_audio_3`, datasets, pandas, pyarrow, soundfile) →
  `source stable-audio-3/.venv310-clean/bin/activate`. There is no root `.venv`.
  Data-script deps: `pip install -U huggingface_hub datasets pandas pyarrow numpy soundfile`.
- **Stable Audio 3:** NOT on PyPI. Vendored as the git submodule `stable-audio-3/`
  (`git submodule update --init --recursive`). For training:
  ```bash
  cd stable-audio-3
  uv sync --extra lora      # lora extra (pytorch_lightning) required for training
  uv run hf auth login      # accept model licenses on Hugging Face first
  ```
  Run its tools with `uv run python ...` from inside `stable-audio-3/`.
- **Cluster (department HPC):**
  - 48 CPU nodes (24 cores, 128 GB RAM, 340 GB local scratch): preprocessing, downloads.
  - 1 large-memory node (72 cores, 4 TB RAM): rarely needed.
  - 6 GPU nodes, each 4× A100 40GB PCIe (1.7 TB NVMe scratch): LoRA training.
    One GPU per run is plenty; use nodes for parallel experiments, not multi-node jobs.
  - 1 GPU node, 3× L40S 48GB: inference, evaluation, demo serving.
  - Scheduler assumed to be SLURM (unconfirmed). Use Apptainer or venv/conda, not Docker.
  - Node-local scratch is temporary: write results to persistent storage.

## Common commands

```bash
# Data
python explore_genres.py
python select_tracks.py --genres rock                 # selected_ids.txt = rock selection
python download_sample.py --n 30 --ids selected_ids.txt --out sample_tracks   # 29 rock tracks already there
python prepare_lora_data.py --audio sample_tracks --out rock_lora_data       # 30-track smoke-test data

# Cluster CPU jobs (submit from the repo root; set the partition in the .slurm files first)
sbatch download_genres.slurm --per-genre 10 --out genre_tracks_smoke   # smoke test first
sbatch download_genres.slurm                                           # rock + hiphop, 1500 each
python prepare_lora_data.py --audio genre_tracks/rock --genre-tag rock
python prepare_lora_data.py --audio genre_tracks/hiphop --genre-tag hiphop   # "Instrumental hip hop" → hiphop_lora_data/

# Training (cluster GPU node, inside stable-audio-3/)
uv run python scripts/train_lora.py \
    --model small-music-base --data_dir ../rock_lora_data \
    --rank 16 --adapter_type dora-rows --exclude seconds_total \
    --duration 30 --steps 1000 --save_dir ./lora_rock_test

# A/B comparison (inside stable-audio-3/)
uv run python ../compare_lora.py --lora lora_rock_test/<checkpoint>.ckpt
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
  `-base` models); `cfg_scale` = classifier-free guidance strength (default 1.0; **no effect on
  post-trained checkpoints**). Keep both fixed within any comparison.
- `generate()` re-seeds torch itself (`seed=-1` → random), so `torch.manual_seed()` before it
  does nothing: always pass `seed=`.
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
- Demo: Gradio app with prompt box, duration slider, base/LoRA/side-by-side toggle.

## Next steps

1. Commit + push the scripts; copy `jamendo_metadata.parquet` and `sample_tracks/` (both
   gitignored) to the cluster.
2. Cluster setup: confirm SLURM, partitions and outbound internet; data venv `.venv`;
   `cd stable-audio-3 && uv sync --extra lora`; accept the HF terms for
   `stabilityai/stable-audio-3-small-music` (gated, auto-approved; `small-music-base`, `SAME-S`
   and the dataset are not gated); `hf auth login`.
3. Run the 29-track rock LoRA test end to end on one A100
   (`prepare_lora_data.py` → `train_lora.py` 1000 steps → `compare_lora.py`).
4. `sbatch download_genres.slurm --per-genre 10 --out genre_tracks_smoke`, check
   `manifest.csv`, then submit the full 3,000-track job.
5. Train per-genre LoRAs (pre-encode with `same-s` first); run ablations.
6. Build the evaluation pipeline and Gradio demo; finish slides before Thanksgiving.

## Open questions to verify

- Whether the cluster uses SLURM, its partition names (fill in `##SBATCH --partition` in both
  `.slurm` files) and whether compute nodes have outbound internet (the download jobs need it;
  otherwise run them on a login/transfer node).
- Whether all Jamendo MP3s are 44.1 kHz: all 29 in `sample_tracks/` are 44.1 kHz stereo;
  check `prepare_lora_data.py` skip messages at scale (the trainer resamples on load anyway).
- JamendoMaxCaps captions: the HF repo has `final_caption30sec.jsonl` (787 MB). Its format hasn't
  been inspected; it might be a better caption source than the tag-built captions.
