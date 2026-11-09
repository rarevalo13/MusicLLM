# MusicLLM

A genre-specialized **instrumental** music generator, built for CSIT 559 (GenAI) at Montclair State.
We LoRA fine-tune [Stable Audio 3 Small](https://huggingface.co/stabilityai/stable-audio-3-small-music)
on rock and hip hop tracks from [JamendoMaxCaps](https://huggingface.co/datasets/amaai-lab/JamendoMaxCaps)
and compare the fine-tuned model with the original.

## Pipeline

```text
explore_genres.py ──> jamendo_metadata.parquet  (tags for every track, no audio)
  ├─ select_tracks.py ──> selected_ids.txt ─┬─ download_sample.py         (Mac: a few MP3s)
  │                                         └─ extract_audio.py + .slurm  (cluster: every shard)
  └─ download_genres.py + .slurm ──> genre_tracks/<genre>/*.mp3           (cluster: N per genre)

prepare_lora_data.py ──> <genre>_lora_data/  (30 s WAV clips + .txt captions)
stable-audio-3/scripts/train_lora.py ──> LoRA checkpoint (.ckpt)          (cluster GPU)
compare_lora.py ──> lora_comparison/         (same prompts, LoRA off vs on)
```

| Script | Purpose | Runs on |
|---|---|---|
| `explore_genres.py` | Download the dataset metadata and print genre counts | Mac or cluster |
| `select_tracks.py` | List the track IDs for one or more genres | Mac |
| `download_sample.py` | Stream a few MP3s for local tests | Mac |
| `download_genres.py` + `.slurm` | Download a balanced number of tracks per genre | Cluster |
| `extract_audio.py` + `.slurm` | Extract every selected track, one shard at a time | Cluster |
| `prepare_lora_data.py` | Cut MP3s into captioned 30 s training clips | Mac or cluster |
| `compare_lora.py` | Generate the same prompts with the LoRA off and on | Cluster GPU |
| `stable_audio_test.py` | One-off text-to-music test | Mac |

## Setup

### Local (macOS)

Use Python 3.10 (pinned in `stable-audio-3/.python-version`).

```bash
git submodule update --init --recursive        # fetches stable-audio-3/
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./stable-audio-3      # Stable Audio 3 (+ torch, soundfile, huggingface_hub)
python -m pip install datasets pandas pyarrow  # used by the data scripts
hf auth login
```

If the environment in `stable-audio-3/.venv310-clean` already exists, activate it instead:
`source stable-audio-3/.venv310-clean/bin/activate`.

`stabilityai/stable-audio-3-small-music` is gated: accept its terms on Hugging Face with the account
you logged in with (approval is automatic). The base model `stable-audio-3-small-music-base`, the
SAME-S autoencoder and the dataset are not gated.

### Cluster

```bash
git clone --recurse-submodules git@github.com:rarevalo13/MusicLLM.git && cd MusicLLM
python3 -m venv .venv          # the .slurm files activate ./.venv (override with VENV=...)
.venv/bin/pip install -U huggingface_hub datasets pandas pyarrow numpy soundfile
cd stable-audio-3 && uv sync --extra lora     # training environment (needs uv)
uv run hf auth login
```

- `uv sync` installs PyTorch built for CUDA 12.6. For an older GPU driver, see "CUDA Version" in
  `stable-audio-3/README.md`.
- Set your partition on the `##SBATCH --partition=<cpu-partition>` line of both `.slurm` files.
- `jamendo_metadata.parquet` and `sample_tracks/` are ignored by git, so copy them to the cluster
  yourself.

## Quick start (local)

```bash
python stable_audio_test.py                               # generate 30 s of music -> neo_soul.wav
python explore_genres.py                                  # ~6 GB of metadata -> jamendo_metadata.parquet
python select_tracks.py --genres rock                     # -> selected_ids.txt
python download_sample.py --n 30 --ids selected_ids.txt   # -> sample_tracks/*.mp3
python prepare_lora_data.py --audio sample_tracks --out rock_lora_data
```

Then train on the cluster (see [Training](#training-with-stable-audio-3scriptstrain_lorapy)).

## Script reference

### `explore_genres.py`

Downloads the dataset's metadata (no audio) and summarizes it. No options.

- Downloads every `.jsonl` file in the dataset repo (1,171 files, about 6 GB, kept in the
  Hugging Face cache) and keeps the records that have a `musicinfo` field.
- `flatten(r)` turns one record into a row: `id, name, artist, duration, license, vocal, speed,
  genres, instruments, moods`.
- Writes `jamendo_metadata.parquet` (394,198 tracks) and prints the top 40 genres, the top 20
  instruments and the license mix.

```bash
python explore_genres.py > genres.txt     # genres.txt in this repo is that printout
```

### `select_tracks.py`

Lists the track IDs for one or more genres. Reads `jamendo_metadata.parquet` from the current folder.

- Keeps instrumental tracks tagged with any of `--genres` (case-insensitive) that last between
  `--min-sec` and `--max-sec`.
- Writes `selected_ids.txt` (one ID per line) and `selected_metadata.csv`, replacing earlier ones,
  and prints the track count, total hours and license mix.

```bash
python select_tracks.py --genres rock                          # 27,289 tracks (the current selected_ids.txt)
python select_tracks.py --genres hiphop --min-sec 60 --limit 500
```

Options: `--genres` (required), `--min-sec` (30), `--max-sec` (600), `--limit` (random sample of
this size, seed 0).

### `download_sample.py`

Streams the dataset and saves the first `--n` matching tracks as MP3s without downloading whole
shards to disk. Use it for small local samples.

- With `--ids`, only tracks listed in that file are saved. Tracks arrive in dataset order, so a
  rare genre takes longer to find.
- MP3s already in `--out` count toward `--n` and are never rewritten. The stream position is saved
  in `<out>/stream_state.pkl`, so the next run continues where the last one stopped.
- `save_state()` writes `stream_state.pkl` after every saved track and every 500 rows scanned.
  `write_ids()` writes all saved IDs to `<out>/downloaded_ids.txt`.

```bash
python download_sample.py --n 30 --ids selected_ids.txt   # 30 rock tracks -> sample_tracks/
python download_sample.py --n 40 --ids selected_ids.txt   # adds 10 more, resuming the stream
python download_sample.py --n 5 --out any_tracks          # first 5 tracks of any genre
```

Options: `--n` (100), `--ids` (none: any track), `--out` (`sample_tracks`), `--restart` (scan from
the first row again; existing files are still skipped). Use a new `--out` or `--restart` when you
change `--ids`, because the saved position belongs to the old filter.

### `download_genres.py` + `download_genres.slurm`

Downloads a balanced training set: `--per-genre` tracks for each genre. Meant for the cluster,
because it downloads whole shards (about 0.47 GB each).

- Candidates come from `jamendo_metadata.parquet`: instrumental, `--min-sec` to `--max-sec` long,
  and tagged with exactly one of the requested genres (a track tagged both `rock` and `hiphop` is
  skipped).
- Walks the 2,272 shards in a shuffled order (`--seed`). Shards are sorted by track ID and
  neighbouring IDs often belong to one artist, so shuffling spreads the sample across the catalog.
  Each shard is downloaded to `--scratch`, scanned and deleted; `--workers` shards download in
  parallel.
- Saves `<out>/<genre>/<id>.mp3` until every genre reaches `--per-genre`, keeping at most
  `--max-per-artist` tracks per artist per genre.
- Tracks progress in `<out>/manifest.csv` (one row per saved track: `id, genre, artist, name,
  duration, license, genres, shard`) and `<out>/done_shards.txt`. Re-running the same command
  resumes. MP3s are never rewritten; if you delete one, the next run replaces it with another track.
- `status()` formats the progress text, e.g. `rock 123/1500, hiphop 45/1500`.

```bash
python download_genres.py --genres rock hiphop --per-genre 2 --out genre_tracks_test   # local test, a few shards
sbatch download_genres.slurm --per-genre 10 --out genre_tracks_smoke                   # cluster smoke test
sbatch download_genres.slurm                                                           # full run: rock + hiphop, 1,500 each
```

Options: `--genres` (required), `--per-genre` (1500), `--out` (`genre_tracks`), `--meta`
(`jamendo_metadata.parquet`), `--min-sec` (60), `--max-sec` (600), `--max-per-artist` (20, `0` = no
cap), `--seed` (0), `--workers` (4), `--scratch` (`$TMPDIR` or `/tmp`).

The `.slurm` job (8 CPUs, 16 GB, 12 h) runs `--genres rock hiphop --per-genre 1500 --out
genre_tracks` with one worker per CPU and `$TMPDIR` as scratch. Arguments after the script name
override those defaults. If the job fails or times out, submit the same command again. The log is
`slurm-download-genres-<jobid>.out`. Rough estimate for the full run: 260–330 shards (120–155 GB)
pass through scratch and about 9 GB of MP3s are kept.

### `extract_audio.py` + `extract_audio.slurm`

Extracts the tracks listed in an ID file (default `selected_ids.txt`), one shard per call. Covering
a whole selection means reading all 2,272 shards (about 1 TB), so do that on the cluster.

- `--list` prints the number of shards.
- `--shard-index i` downloads shard `i` (in file name order) to `<scratch>/jmc_shard_<i>`, writes
  `<out>/<id>.mp3` for every listed ID it contains, then deletes the shard. Re-running a shard
  rewrites the same files, so retries are safe.

```bash
python extract_audio.py --list                              # 2272
python extract_audio.py --shard-index 0 --out audio_subset  # one shard, ~0.5 GB download
sbatch --array=0-567%20 extract_audio.slurm --out /persistent/path/audio_subset
```

Options: `--list`, `--shard-index`, `--ids` (`selected_ids.txt`), `--out` (`audio_subset`),
`--scratch` (`$TMPDIR` or `/tmp`).

The `.slurm` job array gives each task 4 shards (`PER_TASK`) because SLURM arrays are often capped
at 1,001 tasks: `--array=0-567` covers all 2,272 shards and `%20` runs at most 20 tasks at once.
Arguments after the script name go to `extract_audio.py`. Retry failed tasks with
`sbatch --array=<task ids> extract_audio.slurm <same arguments>`.

### `prepare_lora_data.py`

Turns a folder of `<id>.mp3` files into the training format of `train_lora.py`: pairs of
`clip.wav` and `clip.txt` (its caption).

- Looks up each track in `--meta`. Skips tracks without metadata, files that don't decode and audio
  that isn't 44.1 kHz; mono becomes stereo.
- Cuts `--chunk-sec` (30 s) chunks, skipping the first `--skip-intro-sec` (10 s) when the track is
  long enough, dropping near-silent chunks and keeping at most `--max-chunks` per track. Each chunk
  is written as `<id>_00.wav` (16-bit stereo, about 5.3 MB) with its caption in `<id>_00.txt`.
- `caption(r)` builds the caption for one metadata row:
  `<label> with <genre> and <genre> influences, featuring <up to 4 instruments>, <up to 3 moods> mood, <tempo> tempo`.
  Parts without tags are left out, and the genre being trained isn't repeated as an influence.
- `tidy(tags, n)` cleans a tag list: drops `instrumental`, `electroamplified` and `computer`, turns
  joined Jamendo tags into words (`electricguitar` → "electric guitar", `hiphop` → "hip hop",
  `lofi` → "lo-fi") and keeps the first `n`. Tempo tags become "very slow", "slow", "medium",
  "fast" or "very fast".

```bash
python prepare_lora_data.py --audio sample_tracks --out rock_lora_data       # 29 tracks -> up to 116 clips
python prepare_lora_data.py --audio genre_tracks/hiphop --genre-tag hiphop   # "Instrumental hip hop" -> hiphop_lora_data/
```

Example caption: `Instrumental rock with metal influences, energetic, powerful mood, medium tempo`.
Moods come from Jamendo's free-form tags, so some captions contain noise such as "pantera mood".

Options: `--audio` (required), `--meta` (`jamendo_metadata.parquet`), `--genre-tag` (`rock`),
`--label` (`Instrumental <genre>`), `--out` (`<genre-tag>_lora_data`), `--chunk-sec` (30),
`--max-chunks` (4), `--skip-intro-sec` (10).

### Training with `stable-audio-3/scripts/train_lora.py`

This script ships with Stable Audio 3. It trains a LoRA on a base checkpoint from a folder of audio
files with `.txt` captions and needs an NVIDIA GPU (Small uses about 2.5 GB of VRAM). Run it inside
`stable-audio-3/`:

```bash
cd stable-audio-3
uv run python scripts/train_lora.py --model small-music-base --data_dir ../rock_lora_data \
    --rank 16 --adapter_type dora-rows --exclude seconds_total \
    --duration 30 --steps 1000 --save_dir ./lora_rock_test
```

- `--model small-music-base`: the default is `medium-base`.
- `--duration 30`: the default of 380 s pads every 30 s clip to 380 s.
- `--exclude seconds_total`: keeps the LoRA off the duration conditioner (recommended upstream for
  small datasets).
- `--save_dir`: the upstream docs show `--output_dir`, which the script doesn't accept.
- Other defaults: `--steps 10000`, `--lr 1e-4`, `--batch_size 1`, a checkpoint and a demo every 500
  steps, CSV logs under `--save_dir`. Checkpoints are `.ckpt` files that contain only the LoRA weights.
- For large datasets, encode the audio once and train on the latents:
  `uv run python scripts/pre_encode_dataset.py --model same-s --data_dir ../rock_lora_data --output_path ../rock_latents`,
  then pass `--encoded_dir ../rock_latents` instead of `--data_dir`.

### `compare_lora.py`

Generates three built-in rock prompts twice, with the LoRA off (strength 0) and on (strength 1),
using the same seed so the LoRA is the only difference. The LoRA is trained on `small-music-base`
and applied to the post-trained `small-music`.

```bash
cd stable-audio-3
uv run python ../compare_lora.py --lora lora_rock_test/<checkpoint>.ckpt
```

Writes `lora_comparison/prompt0_base.wav`, `prompt0_lora.wav` and so on, in the folder you run it
from. Edit `PROMPTS` in the script to test other genres.

Options: `--lora` (required, `.ckpt` or `.safetensors`), `--model` (`small-music`), `--duration`
(30), `--out` (`lora_comparison`).

### `stable_audio_test.py`

Generates 30 s with `small-music` from the prompt written in the script and saves `neo_soul.wav`.
Edit the prompt and file name in the script.

```bash
python stable_audio_test.py
```

The first run downloads the model weights. Warnings that Flash Attention is disabled are harmless.
With `small-music`, `cfg_scale` has no effect and raising `steps` above the default 8 doesn't
necessarily help (the script uses 50). Without a script:
`stable-audio --model small-music -p "lo-fi hip hop beat, 90 BPM" --duration 30 -o beat.wav`.

## Files and folders

| Path | Contents | Ignored by git |
|---|---|---|
| `jamendo_metadata.parquet` | Output of `explore_genres.py` | yes |
| `genres.txt` | Printout of `explore_genres.py` | no |
| `selected_ids.txt`, `selected_metadata.csv` | `select_tracks.py --genres rock` (27,289 tracks) | no |
| `sample_tracks/` | 29 rock MP3s from `download_sample.py` | yes |
| `genre_tracks*/`, `*_tracks/`, `audio_subset/` | Downloaded MP3s | yes |
| `*_lora_data/` | Training clips from `prepare_lora_data.py` | yes |
| `lora_comparison/`, `*.wav` | Generated audio | yes |
| `slurm-*.out` | SLURM job logs | yes |
| `stable-audio-3/` | Stable Audio 3 code | git submodule |

`stable-audio-3/` is a separate repository pinned to a specific commit. Clone this project with
`git clone --recurse-submodules`, or run `git submodule update --init --recursive` in an existing
clone.

## Dataset and licenses

- JamendoMaxCaps has about 362k instrumental tracks in 2,272 parquet shards (about 1 TB), with
  metadata in JSONL files. A track's ID is its MP3 file name.
- Each track has its own Creative Commons license, many of them non-commercial, so use the data for
  coursework and research only and cite the dataset. `selected_metadata.csv` and `manifest.csv`
  record each track's license.
- Don't download the full dataset to a laptop.
- Stable Audio 3 is released under the [Stability AI Community License](https://stability.ai/license).
