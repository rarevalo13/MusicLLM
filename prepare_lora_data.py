"""Turn downloaded MP3s + Jamendo metadata into a Stable Audio 3 LoRA training folder.

Each track is cut into 30-second WAV chunks, and each chunk gets a matching .txt
caption (clip.wav + clip.txt), which is the format scripts/train_lora.py expects.

Usage:
  python prepare_lora_data.py --audio rock_tracks --meta jamendo_metadata.parquet --out rock_lora_data
  python prepare_lora_data.py --audio genre_tracks/hiphop --genre-tag hiphop --max-chunks 4
Needs: pip install soundfile pandas pyarrow numpy
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--audio", required=True, help="folder of downloaded <id>.mp3 files")
ap.add_argument("--meta", default="jamendo_metadata.parquet")
ap.add_argument("--genre-tag", default="rock", help="Jamendo genre tag these tracks were downloaded for")
ap.add_argument("--label", default=None, help="caption prefix (default: 'Instrumental <genre>')")
ap.add_argument("--out", default=None, help="default: <genre-tag>_lora_data")
ap.add_argument("--chunk-sec", type=float, default=30.0)
ap.add_argument("--max-chunks", type=int, default=4, help="per track, so no song dominates")
ap.add_argument("--skip-intro-sec", type=float, default=10.0)
args = ap.parse_args()

# Jamendo writes multi-word tags as one word; make them readable for the text encoder
READABLE = {
    "electricguitar": "electric guitar", "acousticguitar": "acoustic guitar",
    "drummachine": "drum machine", "electricpiano": "electric piano",
    "classicalguitar": "classical guitar", "doublebass": "double bass",
    "electricbass": "electric bass", "acousticbass": "acoustic bass",
    "slideguitar": "slide guitar", "pipeorgan": "pipe organ",
    "hiphop": "hip hop", "alternativehiphop": "alternative hip hop",
    "triphop": "trip hop", "gangstarap": "gangsta rap",
    "lofi": "lo-fi", "rnb": "R&B", "filmscore": "film score",
    "newage": "new age", "darkambient": "dark ambient",
    "drumnbass": "drum and bass", "alternativerock": "alternative rock",
    "indierock": "indie rock", "poprock": "pop rock",
    "progressiverock": "progressive rock", "postrock": "post-rock",
    "hardrock": "hard rock", "classicrock": "classic rock",
    "bluesrock": "blues rock", "psychedelicrock": "psychedelic rock",
    "heavymetal": "heavy metal", "deathmetal": "death metal",
    "synthpop": "synth pop", "indiepop": "indie pop", "dreampop": "dream pop",
    "easylistening": "easy listening", "jazzfusion": "jazz fusion",
    "deephouse": "deep house", "avantgarde": "avant-garde",
}

DROP = {"instrumental", "electroamplified", "computer"}
TEMPO = {"verylow": "very slow", "low": "slow", "medium": "medium",
         "high": "fast", "veryhigh": "very fast"}

args.genre_tag = args.genre_tag.lower()
args.label = args.label or f"Instrumental {READABLE.get(args.genre_tag, args.genre_tag)}"
args.out = args.out or f"{args.genre_tag}_lora_data"


def tidy(tags, n):
    return [READABLE.get(t, t) for t in list(tags) if t not in DROP][:n]


def caption(r):
    extra = [READABLE.get(t, t) for t in list(r.genres) if t != args.genre_tag and t not in DROP][:2]
    parts = [args.label + (f" with {' and '.join(extra)} influences" if extra else "")]
    inst, moods = tidy(r.instruments, 4), tidy(r.moods, 3)
    if inst:
        parts.append("featuring " + ", ".join(inst))
    if moods:
        parts.append(", ".join(moods) + " mood")
    if r.speed in TEMPO:
        parts.append(f"{TEMPO[r.speed]} tempo")
    return ", ".join(parts)


meta = pd.read_parquet(args.meta).set_index("id")
os.makedirs(args.out, exist_ok=True)
n_clips = 0

for mp3 in sorted(glob.glob(os.path.join(args.audio, "*.mp3"))):
    tid = os.path.splitext(os.path.basename(mp3))[0]
    if tid not in meta.index:
        print(f"skip {tid}: no metadata")
        continue
    try:
        audio, sr = sf.read(mp3, always_2d=True)
    except Exception as e:
        print(f"skip {tid}: could not decode ({e})")
        continue
    if sr != 44100:
        print(f"skip {tid}: sample rate {sr}, expected 44100")
        continue
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)  # mono -> stereo

    cap = caption(meta.loc[tid])
    size = int(args.chunk_sec * sr)
    start = int(args.skip_intro_sec * sr) if len(audio) > size + args.skip_intro_sec * sr else 0
    kept = 0
    for s in range(start, len(audio) - size + 1, size):
        chunk = audio[s:s + size]
        if np.sqrt(np.mean(chunk ** 2)) < 1e-3:  # skip near-silent chunks
            continue
        name = f"{tid}_{kept:02d}"
        sf.write(os.path.join(args.out, name + ".wav"), chunk, sr)
        with open(os.path.join(args.out, name + ".txt"), "w") as fh:
            fh.write(cap)
        kept += 1
        if kept >= args.max_chunks:
            break
    n_clips += kept
    print(f"{tid}: {kept} clips | {cap}")

print(f"\nDone: {n_clips} clips ({n_clips * args.chunk_sec / 60:.0f} min of audio) in {args.out}/")
