"""Step 2: Pick the track IDs for your target genre(s).

Usage:  python select_tracks.py --genres lounge chillout --min-sec 60 --max-sec 420
Output: selected_ids.txt (one track ID per line) and selected_metadata.csv
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--genres", nargs="+", required=True, help="genre tags to keep (any match)")
ap.add_argument("--min-sec", type=int, default=30)
ap.add_argument("--max-sec", type=int, default=600)
ap.add_argument("--limit", type=int, default=None, help="cap number of tracks")
args = ap.parse_args()

wanted = {g.lower() for g in args.genres}
df = pd.read_parquet("jamendo_metadata.parquet")

mask = df.genres.apply(lambda gs: bool(wanted & {g.lower() for g in gs}))
mask &= df.vocal.fillna("instrumental").eq("instrumental")  # belt and braces
mask &= df.duration.between(args.min_sec, args.max_sec)
sel = df[mask]
if args.limit:
    sel = sel.sample(n=min(args.limit, len(sel)), random_state=0)

sel.id.to_csv("selected_ids.txt", index=False, header=False)
sel.to_csv("selected_metadata.csv", index=False)

print(f"Selected {len(sel):,} tracks, about {sel.duration.sum() / 3600:.0f} hours of audio")
print("\nLicense mix of selection:")
print(sel.license.str.extract(r"licenses/([^/]+)/")[0].value_counts().to_string())
