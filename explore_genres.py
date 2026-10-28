"""Step 1: Download JamendoMaxCaps metadata (small, no audio) and explore genres.

Usage:  python explore_genres.py
Output: jamendo_metadata.parquet  (one row per track, used by select_tracks.py)
Needs:  pip install huggingface_hub pandas pyarrow
"""
import collections
import json

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "amaai-lab/JamendoMaxCaps"

files = list_repo_files(REPO, repo_type="dataset")
jsonl_files = [f for f in files if f.endswith(".jsonl")]
if not jsonl_files:
    raise SystemExit("No .jsonl files found. Non-parquet files in repo:\n" +
                     "\n".join(f for f in files if not f.endswith(".parquet")))
print(f"Found {len(jsonl_files)} JSONL files")

records = []
for f in jsonl_files:
    path = hf_hub_download(REPO, f, repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if "musicinfo" in r:  # metadata rows; caption files have another shape
                    records.append(r)


def flatten(r):
    info = r["musicinfo"]
    tags = info.get("tags") or {}
    return {
        "id": str(r["id"]),
        "name": r.get("name"),
        "artist": r.get("artist_name"),
        "duration": r.get("duration"),
        "license": r.get("license_ccurl", ""),
        "vocal": info.get("vocalinstrumental"),
        "speed": info.get("speed"),
        "genres": tags.get("genres") or [],
        "instruments": tags.get("instruments") or [],
        "moods": tags.get("vartags") or [],
    }


df = pd.DataFrame([flatten(r) for r in records]).drop_duplicates("id")
df.to_parquet("jamendo_metadata.parquet")

print(f"\n{len(df):,} tracks with metadata")
print(f"{(df.genres.str.len() == 0).mean():.0%} have NO genre tags\n")

print("Top 40 genres (track counts):")
for genre, n in collections.Counter(g for gs in df.genres for g in gs).most_common(40):
    print(f"  {genre:<22}{n:>8,}")

print("\nTop 20 instruments:")
for inst, n in collections.Counter(i for xs in df.instruments for i in xs).most_common(20):
    print(f"  {inst:<22}{n:>8,}")

print("\nLicense types:")
print(df.license.str.extract(r"licenses/([^/]+)/")[0].value_counts().to_string())
