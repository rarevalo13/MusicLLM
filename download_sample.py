"""Download a small sample (e.g. 100 tracks) from JamendoMaxCaps by streaming.

Streams rows one at a time and stops as soon as N tracks are saved,
so it never downloads whole shards to disk. Resumable: existing MP3s count
toward --n and are never rewritten, and <out>/stream_state.pkl restores the
stream position so a re-run continues where it stopped (--restart to scan
from the first row again).

Usage:
  # Any 100 tracks (fastest):
  python download_sample.py --n 100

  # 100 tracks from your genre selection (run select_tracks.py first):
  python download_sample.py --n 100 --ids selected_ids.txt

Needs: pip install datasets
"""
import argparse
import os
import pickle

from datasets import Audio, load_dataset

REPO = "amaai-lab/JamendoMaxCaps"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100, help="number of tracks to save")
ap.add_argument("--ids", default=None, help="optional file of track IDs to keep")
ap.add_argument("--out", default="sample_tracks")
ap.add_argument("--restart", action="store_true",
                help="ignore saved stream position and scan from the first row")
args = ap.parse_args()

keep = set(open(args.ids).read().split()) if args.ids else None
os.makedirs(args.out, exist_ok=True)

saved = [os.path.splitext(f)[0] for f in os.listdir(args.out) if f.endswith(".mp3")]
if keep is not None:
    saved = [t for t in saved if t in keep]
state_path = os.path.join(args.out, "stream_state.pkl")

def write_ids():
    with open(os.path.join(args.out, "downloaded_ids.txt"), "w") as fh:
        fh.write("\n".join(saved))

if len(saved) >= args.n:
    write_ids()
    print(f"already have {len(saved)} >= {args.n} tracks in {args.out}/, nothing to do")
    raise SystemExit

ds = load_dataset(REPO, split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))  # raw MP3 bytes, no decoding

scanned = 0
if not args.restart and os.path.exists(state_path):
    with open(state_path, "rb") as fh:
        state = pickle.load(fh)
    ds.load_state_dict(state["state"])
    scanned = state["scanned"]
    print(f"Resuming after row {scanned:,}")

def save_state():
    tmp = state_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"state": ds.state_dict(), "scanned": scanned}, fh)
    os.replace(tmp, state_path)

for row in ds:
    scanned += 1
    audio = row["audio"]
    track_id = os.path.splitext(os.path.basename(audio["path"]))[0]
    if track_id in saved:
        continue
    if (keep is None or track_id in keep) and audio.get("bytes"):
        dest = os.path.join(args.out, f"{track_id}.mp3")
        part = dest + ".part"
        with open(part, "wb") as fh:
            fh.write(audio["bytes"])  # original file, no re-encoding
        os.replace(part, dest)
        saved.append(track_id)
        save_state()
        print(f"saved {len(saved)}/{args.n}  (scanned {scanned:,} rows)")
        if len(saved) >= args.n:
            break
    elif scanned % 500 == 0:
        save_state()
        print(f"...scanned {scanned:,} rows, {len(saved)} matches so far")

write_ids()
print(f"\nDone: {len(saved)} tracks in {args.out}/ after scanning {scanned:,} rows")
