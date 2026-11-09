"""Download balanced per-genre subsets of JamendoMaxCaps for LoRA training.

Keeps only tracks carrying exactly one of the requested genre tags (tracks tagged
with 2+ requested genres are skipped), then walks the parquet shards in a seeded
random order, saving MP3s into <out>/<genre>/ until each genre has --per-genre.
Each shard is downloaded to scratch, scanned, and deleted. Resumable: re-run or
resubmit the same command; manifest.csv + done_shards.txt track progress and
existing files are never rewritten.

Usage:
  python download_genres.py --genres rock hiphop --per-genre 10 --out genre_tracks_smoke   # smoke test
  python download_genres.py --genres rock hiphop --per-genre 1500 --out genre_tracks       # full run

Needs: pip install huggingface_hub pandas pyarrow
"""
import argparse
import csv
import os
import random
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "amaai-lab/JamendoMaxCaps"
COLS = ["id", "genre", "artist", "name", "duration", "license", "genres", "shard"]

ap = argparse.ArgumentParser()
ap.add_argument("--genres", nargs="+", required=True)
ap.add_argument("--per-genre", type=int, default=1500)
ap.add_argument("--out", default="genre_tracks")
ap.add_argument("--meta", default="jamendo_metadata.parquet")
ap.add_argument("--min-sec", type=float, default=60)
ap.add_argument("--max-sec", type=float, default=600)
ap.add_argument("--max-per-artist", type=int, default=20, help="per genre, 0 = no cap")
ap.add_argument("--seed", type=int, default=0, help="shard order")
ap.add_argument("--workers", type=int, default=4, help="shards downloaded in parallel ahead of processing")
ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
args = ap.parse_args()
args.genres = [g.lower() for g in args.genres]

wanted = set(args.genres)
meta = pd.read_parquet(args.meta)
vocal = meta["vocal"].fillna("instrumental") == "instrumental"
ok_dur = (meta["duration"] >= args.min_sec) & (meta["duration"] <= args.max_sec)
cand = {}
for r in meta[vocal & ok_dur].itertuples():
    hits = wanted & set(r.genres)
    if len(hits) == 1:
        cand[r.id] = (hits.pop(), r)
for g in args.genres:
    if not any(v[0] == g for v in cand.values()):
        raise SystemExit(f"genre '{g}' has 0 eligible tracks -- check spelling in genres.txt")
print(f"{len(cand):,} candidate tracks: " +
      ", ".join(f"{g} {sum(v[0] == g for v in cand.values()):,}" for g in args.genres))

os.makedirs(args.out, exist_ok=True)
manifest_path = os.path.join(args.out, "manifest.csv")
done_path = os.path.join(args.out, "done_shards.txt")
manifest, have = [], set()
if os.path.exists(manifest_path):
    with open(manifest_path, newline="") as fh:
        manifest = list(csv.DictReader(fh))
    kept = [r for r in manifest if os.path.exists(os.path.join(args.out, r["genre"], r["id"] + ".mp3"))]
    if len(kept) < len(manifest):
        print(f"dropped {len(manifest) - len(kept)} manifest rows whose MP3 is missing")
        with open(manifest_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, COLS)
            w.writeheader()
            w.writerows(kept)
    manifest = kept
    have = {r["id"] for r in manifest}
counts = Counter(r["genre"] for r in manifest if r["genre"] in wanted)
artists = Counter((r["genre"], r["artist"]) for r in manifest if r["genre"] in wanted)
done_shards = set(open(done_path).read().split()) if os.path.exists(manifest_path) and os.path.exists(done_path) else set()

mfh = open(manifest_path, "a", newline="")
writer = csv.DictWriter(mfh, COLS)
if mfh.tell() == 0:
    writer.writeheader()
    mfh.flush()

def status():
    return ", ".join(f"{g} {counts[g]}/{args.per_genre}" for g in args.genres)

if all(counts[g] >= args.per_genre for g in args.genres):
    print("all quotas already met:", status())
    raise SystemExit

shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
random.Random(args.seed).shuffle(shards)
todo = [s for s in shards if s not in done_shards]
os.makedirs(args.scratch, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="jmc_genres_", dir=args.scratch)
dfh = open(done_path, "a" if done_shards else "w")
saved_bytes, scanned_shards = 0, 0
ex = ThreadPoolExecutor(args.workers)
try:
    pending = {}
    next_submit = 0
    for i, shard in enumerate(todo):
        while next_submit < len(todo) and len(pending) < args.workers:
            pending[next_submit] = ex.submit(
                hf_hub_download, REPO, todo[next_submit], repo_type="dataset",
                local_dir=os.path.join(tmp, str(next_submit)))
            next_submit += 1
        local = pending.pop(i).result()
        saved_bytes += os.path.getsize(local)
        saved, quota_met = 0, False
        for batch in pq.ParquetFile(local).iter_batches(columns=["audio"], batch_size=32):
            for audio in batch.column("audio").to_pylist():
                tid = os.path.splitext(os.path.basename(audio["path"]))[0]
                c = cand.get(tid)
                if (c is None or not audio.get("bytes") or tid in have
                        or counts[c[0]] >= args.per_genre
                        or (c[1].artist and args.max_per_artist
                            and artists[(c[0], c[1].artist)] >= args.max_per_artist)):
                    continue
                g, r = c
                gdir = os.path.join(args.out, g)
                os.makedirs(gdir, exist_ok=True)
                dest = os.path.join(gdir, tid + ".mp3")
                if not os.path.exists(dest):
                    part = dest + ".part"
                    with open(part, "wb") as fh:
                        fh.write(audio["bytes"])  # original file, no re-encoding
                    os.replace(part, dest)
                row = {"id": tid, "genre": g, "artist": r.artist, "name": r.name,
                       "duration": r.duration, "license": r.license,
                       "genres": ",".join(r.genres), "shard": shard}
                writer.writerow(row)
                manifest.append(row)
                mfh.flush()
                have.add(tid)
                counts[g] += 1
                artists[(g, r.artist)] += 1
                saved += 1
            if all(counts[g] >= args.per_genre for g in args.genres):
                quota_met = True
                break
        scanned_shards += 1
        if quota_met:
            print(f"[{scanned_shards}/{len(todo)}] {shard}: +{saved} saved | {status()} | all quotas met, stopping")
            break
        dfh.write(shard + "\n")
        dfh.flush()
        shutil.rmtree(os.path.join(tmp, str(i)), ignore_errors=True)
        print(f"[{scanned_shards}/{len(todo)}] {shard}: +{saved} saved | {status()} | "
              f"{saved_bytes / 1e9:.1f} GB downloaded")
finally:
    ex.shutdown(wait=True, cancel_futures=True)
    shutil.rmtree(tmp, ignore_errors=True)
    mfh.close()
    dfh.close()

print("\nDone:")
for g in args.genres:
    na = len({r["artist"] for r in manifest if r["genre"] == g})
    print(f"  {g}: {counts[g]}/{args.per_genre} tracks, {na} distinct artists")
print(f"  shards scanned: {scanned_shards}, {saved_bytes / 1e9:.1f} GB downloaded this run")
if not all(counts[g] >= args.per_genre for g in args.genres):
    print("  WARNING: ran out of shards before all quotas were met")
