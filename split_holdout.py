"""Split each genre's downloaded MP3s into training and held-out sets by artist.

Whole artists are held out (no artist appears in both splits) without moving any
MP3s: genre_splits/<genre>/train|heldout/<id>.mp3 are relative symlinks into
<tracks>/<genre>/, plus train.csv/heldout.csv with the manifest rows. Re-running
reuses the group keys in an existing heldout.csv, so the split never reshuffles.

Usage:
  python split_holdout.py --tracks genre_tracks --per-genre 300
Run before make_latents.slurm; stdlib only.
"""
import argparse
import csv
import os
import random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--tracks", default="genre_tracks", help="download_genres.py --out folder")
ap.add_argument("--out", default="genre_splits")
ap.add_argument("--per-genre", type=int, default=300, help="minimum held-out tracks per genre")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

manifest_path = os.path.join(args.tracks, "manifest.csv")
if not os.path.exists(manifest_path):
    raise SystemExit(f"no manifest: {manifest_path} (run download_genres.py first)")
with open(manifest_path, newline="") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames or []
    rows = [r for r in reader
            if os.path.exists(os.path.join(args.tracks, r["genre"], r["id"] + ".mp3"))]
if not rows:
    raise SystemExit(f"no manifest rows with an existing MP3 under {args.tracks}")


def group_key(r):
    return r["artist"] or "track:" + r["id"]


splits = {}
for genre in sorted({r["genre"] for r in rows}):
    grows = [r for r in rows if r["genre"] == genre]
    groups = defaultdict(list)
    for r in grows:
        groups[group_key(r)].append(r)
    held_path = os.path.join(args.out, genre, "heldout.csv")
    if os.path.exists(held_path):
        with open(held_path, newline="") as fh:
            held_keys = {group_key(r) for r in csv.DictReader(fh)}
    else:
        keys = sorted(groups)
        random.Random(args.seed).shuffle(keys)
        held_keys, n_held = set(), 0
        for k in keys:
            if n_held >= args.per_genre:
                break
            held_keys.add(k)
            n_held += len(groups[k])
    held = [r for r in grows if group_key(r) in held_keys]
    train = [r for r in grows if group_key(r) not in held_keys]
    if not train:
        raise SystemExit(
            f"{genre}: split would leave 0 training tracks "
            f"({len(grows)} tracks, {len(held)} held out)")
    splits[genre] = (train, held)

for genre, (train, held) in splits.items():
    for subset, subset_rows in (("train", train), ("heldout", held)):
        d = os.path.join(args.out, genre, subset)
        os.makedirs(d, exist_ok=True)
        for r in subset_rows:
            link = os.path.join(d, r["id"] + ".mp3")
            if os.path.lexists(link):
                continue
            target = os.path.join(args.tracks, genre, r["id"] + ".mp3")
            os.symlink(os.path.relpath(target, d), link)
    for name, subset_rows in (("train.csv", train), ("heldout.csv", held)):
        with open(os.path.join(args.out, genre, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(subset_rows)
    print(f"{genre}: train {len(train):,} tracks / {len({group_key(r) for r in train}):,} artists"
          f" | held out {len(held):,} tracks / {len({group_key(r) for r in held}):,} artists")
