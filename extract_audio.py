"""Step 3: Pull only the selected tracks out of ONE parquet shard, then delete the shard.

Run one task per shard (see extract_audio.slurm), so no node ever holds the full ~1 TB.

Usage:  python extract_audio.py --list                       # print number of shards
        python extract_audio.py --shard-index 0 --out /persistent/path/audio_subset
"""
import argparse
import os
import shutil

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "amaai-lab/JamendoMaxCaps"

ap = argparse.ArgumentParser()
ap.add_argument("--shard-index", type=int)
ap.add_argument("--list", action="store_true")
ap.add_argument("--ids", default="selected_ids.txt")
ap.add_argument("--out", default="audio_subset")
ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
args = ap.parse_args()

shards = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
if args.list:
    print(len(shards))
    raise SystemExit

keep = set(open(args.ids).read().split())
shard = shards[args.shard_index]
os.makedirs(args.out, exist_ok=True)

# local_dir writes a real file (not a cache symlink), so deleting it frees scratch space
tmp_dir = os.path.join(args.scratch, f"jmc_shard_{args.shard_index}")
local = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=tmp_dir)

saved = 0
for batch in pq.ParquetFile(local).iter_batches(columns=["audio"], batch_size=32):
    for audio in batch.column("audio").to_pylist():
        track_id = os.path.splitext(os.path.basename(audio["path"]))[0]
        if track_id in keep and audio.get("bytes"):
            # Write the original encoded bytes: lossless and much faster than
            # decoding to an array and re-encoding to MP3.
            with open(os.path.join(args.out, f"{track_id}.mp3"), "wb") as fh:
                fh.write(audio["bytes"])
            saved += 1

shutil.rmtree(tmp_dir, ignore_errors=True)
print(f"{shard}: saved {saved} tracks")
