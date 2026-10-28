# MusicLLM

Experiments with Stable Audio 3 music generation and genre-based selection of JamendoMaxCaps tracks.

## Setup

Use Python 3.10 for the pinned Stable Audio dependencies. From the project root, with Python 3.10 installed:

```bash
git submodule update --init --recursive
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./stable-audio-3
python -m pip install pandas pyarrow
hf auth login
```

If you already have the working environment in `stable-audio-3/.venv310-clean`, activate that instead of creating another:

```bash
source stable-audio-3/.venv310-clean/bin/activate
```

Request access to the gated model using the same Hugging Face account:
https://huggingface.co/stabilityai/stable-audio-3-small-music

## Generate music

```bash
python stable_audio_test.py
```

Edit the prompt and generation settings in the script. It currently generates 30 seconds of audio and saves `festival_house.wav`. The first run downloads model weights. Flash Attention is optional; warnings that it is disabled are not fatal.

## Explore and select Jamendo tracks

Run these commands from the project root:

```bash
python explore_genres.py
python select_tracks.py --genres lounge chillout --min-sec 60 --max-sec 420 --limit 100
python extract_audio.py --list
python extract_audio.py --shard-index 0 --out audio_subset
```

- `explore_genres.py`: downloads metadata, prints genre statistics, and creates `jamendo_metadata.parquet`.
- `select_tracks.py`: selects instrumental tracks matching any requested genre and writes `selected_ids.txt` and `selected_metadata.csv`.
- `extract_audio.py`: downloads one audio shard, extracts selected tracks into MP3 files, and removes its temporary shard directory. Repeat with other shard indices as needed; index 0 alone may not contain your selected tracks.

Audio shards can be large. Check available disk space and review dataset and individual track licenses before use.

## Stable Audio submodule

`stable-audio-3/` is a separate repository pinned to a specific commit. When cloning this project, use:

```bash
git clone --recurse-submodules <repository-url>
```

For an existing clone, run `git submodule update --init --recursive`.

Virtual environments, generated WAV files, and local `.env` files are ignored. Downloaded datasets and extracted MP3 files are not currently ignored; avoid staging them unintentionally.
