"""Compute the base-vs-LoRA evaluation metrics from eval_features.py outputs (CPU, numpy only).

The held-out clips (condition real) are the reference. For every generated set and genre:
  per_clip.csv      CLAP score (caption vs audio); AST tag probabilities of the four genres and the
                    4-way AST prediction; CLAP zero-shot 4-genre probabilities and prediction; KL
                    between the AST tag distributions of a generated clip and the held-out clip it
                    was prompted from (same caption); level stats
  paired.csv        per metric: real, base and LoRA means and LoRA - base, with 95% bootstrap CIs
                    resampling tracks, a sign-flip permutation p-value over tracks, LoRA win rate
  distribution.csv  KAD (CLAP and AST embeddings, 95% bootstrap CIs over tracks) and FAD (CLAP)
                    against the held-out clips. Held-out artists are split into two folds; each fold
                    in turn is the reference and the other fold is evaluated as real (the floor) and
                    as the base and LoRA clips generated from its captions; the folds are averaged.
                    gap_closed = (base - lora) / (base - real)
  cross_genre.csv   KAD of every set/condition/genre against every genre's held-out clips
  confusion.csv     4-genre prediction counts (AST and CLAP zero-shot)
  ltas.npz, ltas_summary.csv  long-term average spectra; mean |dB| gap to real over 50 Hz-16 kHz
                    and the share of energy above 16 kHz
  folds.csv         the artist folds
KAD follows kadtk (Chung et al. 2025): Gaussian kernel, bandwidth = median pairwise distance of the
reference set, unbiased MMD^2, x100, except that within-set means skip pairs of clips from the same
track. Embeddings are per clip (mean over 10 s windows), so KAD/FAD
values aren't comparable with published frame-level numbers.

On Hawk: srun -p short -c 4 --mem=16G -t 60 ./hawk_container.sh python eval_metrics.py
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter

import numpy as np

GENRES = ["rock", "hiphop", "classical", "house"]
AST_NAMES = {"rock": "Rock music", "hiphop": "Hip hop music",
             "classical": "Classical music", "house": "House music"}
SETS = ["eval_gen", "eval_gen_steps4", "eval_gen_steps16", "eval_gen_steps32",
        "eval_gen_steps50", "eval_gen_base50"]
PER_CLIP = {"ids", "captions", "clap", "clap_score", "ast_emb", "ast_logits", "ast_prob",
            "ltas_db", "rms_db", "peak", "clip_frac", "side_db"}
EMBEDDINGS = {"clap": "clap", "ast": "ast_emb"}
HIGHER = {"clap_score": True, "ast_p_target": True, "ast_acc": True,
          "zs_p_target": True, "zs_acc": True, "kl_ast": False}
KAD_SCALE = 100

ap = argparse.ArgumentParser()
ap.add_argument("--genres", nargs="+", default=GENRES)
ap.add_argument("--sets", nargs="+", default=SETS)
ap.add_argument("--reference", default="heldout_clips", help="feature set name of the held-out clips")
ap.add_argument("--feats", default="eval_feats")
ap.add_argument("--splits", default="genre_splits", help="heldout.csv per genre gives each track's artist")
ap.add_argument("--out", default="eval_results")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--perm", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)


def load(name, genre, cond):
    path = os.path.join(args.feats, name, genre, cond + ".npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def subset(d, ids):
    pos = {c: i for i, c in enumerate(d["ids"])}
    idx = np.array([pos[c] for c in ids], dtype=int)
    return {k: (v[idx] if k in PER_CLIP else v) for k, v in d.items()}


def artist_map(genre):
    path = os.path.join(args.splits, genre, "heldout.csv")
    if not os.path.exists(path):
        print(f"warning: no {path}: folds split by track instead of artist")
        return {}
    with open(path, newline="") as f:
        return {r["id"]: r["artist"] or "track:" + r["id"] for r in csv.DictReader(f)}


def assign_folds(ids, artist_of):
    # whole artists go to one fold; shuffled with --seed, then greedily balanced by clip count
    tracks = np.array([c.rsplit("_", 1)[0] for c in ids])
    artists = np.array([artist_of.get(t, "track:" + t) for t in tracks])
    count = Counter(artists.tolist())
    keys = sorted(count)
    random.Random(args.seed).shuffle(keys)
    size, fold_of = [0, 0], {}
    for k in keys:
        f = int(size[1] < size[0])
        fold_of[k] = f
        size[f] += count[k]
    return tracks, artists, np.array([fold_of[a] for a in artists])


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def sqdist(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)


def median_dist(x):
    d = np.sqrt(sqdist(x, x)[np.triu_indices(len(x), 1)])
    k = (len(d) - 1) // 2  # the lower median, as torch.median returns in kadtk
    return np.partition(d, k)[k]


class KAD:
    """KAD of evaluation sets against one reference set, optionally for bootstrap resamples.

    Within-set kernel means skip pairs of clips from the same track (chunks of one song, or clips
    generated from one caption, aren't independent samples); with one clip per track this is
    exactly kadtk's unbiased estimator."""

    def __init__(self, ref, ref_tracks):
        self.ref = ref
        self.gamma = 1 / (2 * median_dist(ref) ** 2 + 1e-8)
        self.k_rr = np.exp(-self.gamma * sqdist(ref, ref))[ref_tracks[:, None] != ref_tracks[None, :]].mean()

    def __call__(self, x, tracks, w=None):
        # w: (B, n) count of each clip in B bootstrap resamples of the tracks
        w = np.ones((1, len(x))) if w is None else w
        other = (tracks[:, None] != tracks[None, :]).astype(np.float64)
        k_xx = np.exp(-self.gamma * sqdist(x, x)) * other
        k_xr = np.exp(-self.gamma * sqdist(x, self.ref)).mean(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            within = ((w @ k_xx) * w).sum(1) / ((w @ other) * w).sum(1)
        return KAD_SCALE * (self.k_rr + within - 2 * (w @ k_xr) / w.sum(1))


def fad(ref, x):
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    c1, c2 = np.cov(ref, rowvar=False), np.cov(x, rowvar=False)
    ev, v = np.linalg.eigh(c1)
    s1 = (v * np.sqrt(np.clip(ev, 0, None))) @ v.T
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(s1 @ c2 @ s1), 0, None)).sum()
    return float(((ref.mean(0) - x.mean(0)) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * tr_sqrt)


def boot_weights(tracks):
    # (boot, clips): how often each clip's track is drawn when resampling tracks with replacement
    uniq, inv = np.unique(tracks, return_inverse=True)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=args.boot)
    return counts[:, inv].astype(np.float64)


def ci(reps):
    # a resample with every clip from one track has no between-track pairs (KAD undefined): skip those
    with np.errstate(invalid="ignore", divide="ignore"):
        return tuple(np.nanpercentile(reps, [2.5, 97.5]))


def clip_metrics(d, genre, real=None):
    p4 = d["ast_prob"][:, [label_index[AST_NAMES[g]] for g in GENRES]].astype(np.float64)
    zs_p = softmax(zs_scale * d["clap"].astype(np.float64) @ zs_emb.T)
    gi = GENRES.index(genre)
    m = {"clap_score": d["clap_score"].astype(np.float64), "ast_p_target": p4[:, gi],
         "ast_acc": (p4.argmax(1) == gi).astype(np.float64), "zs_p_target": zs_p[:, gi],
         "zs_acc": (zs_p.argmax(1) == gi).astype(np.float64)}
    if real is not None:
        lp, lq = log_softmax(real["ast_logits"]), log_softmax(d["ast_logits"])
        m["kl_ast"] = (np.exp(lp) * (lp - lq)).sum(1)
    return m, p4, zs_p


def clip_rows(set_name, cond, genre, d, m, p4, zs_p, tracks, artists, folds):
    rows = []
    for i, c in enumerate(d["ids"]):
        rows.append({"set": set_name, "genre": genre, "condition": cond, "clip_id": c,
                     "track": tracks[i], "artist": artists[i], "fold": folds[i], "caption": d["captions"][i],
                     "clap_score": m["clap_score"][i],
                     **{f"ast_p_{g}": p4[i, j] for j, g in enumerate(GENRES)},
                     "ast_pred": GENRES[p4[i].argmax()],
                     **{f"zs_p_{g}": zs_p[i, j] for j, g in enumerate(GENRES)},
                     "zs_pred": GENRES[zs_p[i].argmax()],
                     "kl_ast": m["kl_ast"][i] if "kl_ast" in m else "",
                     **{k: d[k][i] for k in ("rms_db", "peak", "clip_frac", "side_db")}})
    return rows


def confusion_rows(set_name, cond, genre, p4, zs_p):
    rows = []
    for judge, p in (("ast", p4), ("clap_zs", zs_p)):
        count = Counter(p.argmax(1).tolist())
        rows += [{"set": set_name, "genre": genre, "condition": cond, "judge": judge,
                  "predicted": g, "count": count.get(j, 0)} for j, g in enumerate(GENRES)]
    return rows


def paired_row(set_name, genre, metric, real_v, base_v, lora_v, tracks):
    w = boot_weights(tracks)
    row = {"set": set_name, "genre": genre, "metric": metric, "higher_is_better": int(HIGHER[metric]),
           "n_clips": len(base_v), "n_tracks": len(np.unique(tracks))}
    for name, v in (("real", real_v), ("base", base_v), ("lora", lora_v), ("delta", None)):
        if name == "delta":
            v = lora_v - base_v
        if v is None:
            row.update({name: "", name + "_lo": "", name + "_hi": ""})
            continue
        row[name] = v.mean()
        row[name + "_lo"], row[name + "_hi"] = ci((w @ v) / w.sum(1))
    delta = lora_v - base_v
    s = np.bincount(np.unique(tracks, return_inverse=True)[1], weights=delta)
    signs = rng.choice([-1.0, 1.0], size=(args.perm, len(s)))
    row["p_perm"] = (1 + (np.abs(signs @ s) >= abs(s.sum()) - 1e-12).sum()) / (1 + args.perm)
    row["lora_win_rate"] = (delta > 0 if HIGHER[metric] else delta < 0).mean()
    return row


def distribution_rows(set_name, genre, R, folds, tracks, ev_sets, ev_folds, ev_tracks):
    # R/folds/tracks: all held-out clips (the reference halves); ev_sets: real/base/lora on the same clip ids
    rows = []
    for emb, key in EMBEDDINGS.items():
        pts, reps, fads = ({c: [] for c in ev_sets} for _ in range(3))
        n_ref, n_ev = [], []
        for f in (0, 1):
            ref, ev = R[key][folds == f], ev_folds == 1 - f
            kad = KAD(ref, tracks[folds == f])
            w = boot_weights(ev_tracks[ev])
            for c, d in ev_sets.items():
                pts[c].append(kad(d[key][ev], ev_tracks[ev])[0])
                reps[c].append(kad(d[key][ev], ev_tracks[ev], w))
                if emb == "clap":
                    fads[c].append(fad(ref, d[key][ev]))
            n_ref.append(len(ref))
            n_ev.append(int(ev.sum()))
        base = {"set": set_name, "genre": genre, "embedding": emb, "n_ref": np.mean(n_ref), "n_eval": np.mean(n_ev)}
        for metric, values, boot in (("kad", pts, reps), ("fad", fads, None)):
            if emb != "clap" and metric == "fad":
                continue
            p = {c: np.mean(v) for c, v in values.items()}
            row = {**base, "metric": metric}
            r = {c: np.mean(v, axis=0) for c, v in boot.items()} if boot else None
            for c in p:
                row[c] = p[c]
                row[c + "_lo"], row[c + "_hi"] = ci(r[c]) if r else ("", "")
            row["delta"] = p["lora"] - p["base"]
            row["delta_lo"], row["delta_hi"] = ci(r["lora"] - r["base"]) if r else ("", "")
            row["gap_closed"] = (p["base"] - p["lora"]) / (p["base"] - p["real"])
            row["gap_lo"], row["gap_hi"] = (ci((r["base"] - r["lora"]) / (r["base"] - r["real"]))
                                            if r else ("", ""))
            rows.append(row)
    return rows


def ltas_curve(db):
    return 10 * np.log10(np.mean(10 ** (db.astype(np.float64) / 10), axis=0))


def write_csv(name, rows):
    if not rows:
        return
    path = os.path.join(args.out, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: f"{v:.6g}" if isinstance(v, (float, np.floating)) else v for k, v in r.items()}
                    for r in rows)
    print(f"wrote {path} ({len(rows)} rows)")


zs = np.load(os.path.join(args.feats, "clap_text_genres.npz"))
if list(zs["genres"]) != GENRES:
    sys.exit(f"error: unexpected genre order in clap_text_genres.npz: {list(zs['genres'])}")
zs_emb, zs_scale = zs["emb"].astype(np.float64), float(zs["logit_scale_a"])

refs = {}
for g in args.genres:
    R = load(args.reference, g, "real")
    if R is None:
        print(f"warning: no {args.reference} features for {g}; skipping the genre")
        continue
    refs[g] = (R, *assign_folds(R["ids"], artist_map(g)))
if not refs:
    sys.exit(f"error: no {args.reference} features in {args.feats}")
label_index = {l: i for i, l in enumerate(next(iter(refs.values()))[0]["ast_labels"])}
if any(n not in label_index for n in AST_NAMES.values()):
    sys.exit("error: AST labels lack one of " + ", ".join(AST_NAMES.values()))

per_clip, paired, dist, conf, ltas_sum, folds_out, evaluated, curves = [], [], [], [], [], [], [], {}
freqs = next(iter(refs.values()))[0]["ltas_freqs"].astype(np.float64)
band, high = (freqs >= 50) & (freqs <= 16000), freqs > 16000


def add_ltas(set_name, genre, cond, curve, real_curve):
    curves[f"{set_name}__{genre}__{cond}"] = curve
    power = 10 ** (curve / 10)
    ltas_sum.append({"set": set_name, "genre": genre, "condition": cond,
                     "gap_db": np.abs(curve - real_curve)[band].mean(),
                     "above_16k_db": 10 * np.log10(power[high].sum() / power.sum())})


for g, (R, tracks, artists, folds) in refs.items():
    for a in sorted(set(artists.tolist())):
        folds_out.append({"genre": g, "artist": a, "fold": int(folds[artists == a][0]),
                          "n_clips": int((artists == a).sum())})
    m, p4, zs_p = clip_metrics(R, g)
    per_clip += clip_rows(args.reference, "real", g, R, m, p4, zs_p, tracks, artists, folds)
    conf += confusion_rows(args.reference, "real", g, p4, zs_p)
    real_curve = ltas_curve(R["ltas_db"])
    add_ltas(args.reference, g, "real", real_curve, real_curve)
    evaluated.append((args.reference, "real", g, R, folds, tracks))
    for s in args.sets:
        B, L = load(s, g, "base"), load(s, g, "lora")
        if B is None or L is None:
            print(f"warning: no {s} features for {g}; skipping")
            continue
        common = set(B["ids"].tolist()) & set(L["ids"].tolist())
        k = np.array([i for i, c in enumerate(R["ids"]) if c in common], dtype=int)
        if len(k) < len(R["ids"]):
            print(f"warning: {s}/{g}: only {len(k)} of {len(R['ids'])} held-out clips have base and lora features")
        keep = R["ids"][k]
        ev = {"real": subset(R, keep), "base": subset(B, keep), "lora": subset(L, keep)}
        tk, ak, fk = tracks[k], artists[k], folds[k]
        res = {"real": clip_metrics(ev["real"], g)[0]}
        for cond in ("base", "lora"):
            res[cond], p4, zs_p = clip_metrics(ev[cond], g, real=ev["real"])
            per_clip += clip_rows(s, cond, g, ev[cond], res[cond], p4, zs_p, tk, ak, fk)
            conf += confusion_rows(s, cond, g, p4, zs_p)
            add_ltas(s, g, cond, ltas_curve(ev[cond]["ltas_db"]), real_curve)
            evaluated.append((s, cond, g, ev[cond], fk, tk))
        for metric in HIGHER:
            paired.append(paired_row(s, g, metric, res["real"].get(metric), res["base"][metric],
                                     res["lora"][metric], tk))
        dist += distribution_rows(s, g, R, folds, tracks, ev, fk, tk)

cross = []
for emb, key in EMBEDDINGS.items():
    for ref_g, (R, tracks, _, folds) in refs.items():
        kads = [KAD(R[key][folds == f], tracks[folds == f]) for f in (0, 1)]
        for s, cond, g, d, fo, tr in evaluated:
            cross.append({"set": s, "condition": cond, "eval_genre": g, "ref_genre": ref_g, "embedding": emb,
                          "kad": np.mean([kads[f](d[key][fo == 1 - f], tr[fo == 1 - f])[0] for f in (0, 1)])})

os.makedirs(args.out, exist_ok=True)
write_csv("per_clip.csv", per_clip)
write_csv("paired.csv", paired)
write_csv("distribution.csv", dist)
write_csv("cross_genre.csv", cross)
write_csv("confusion.csv", conf)
write_csv("ltas_summary.csv", ltas_sum)
write_csv("folds.csv", folds_out)
np.savez(os.path.join(args.out, "ltas.npz"), freqs=freqs, **curves)

print("\nKAD vs held-out (CLAP embedding; real = floor), and LoRA - base on per-clip metrics:")
for r in dist:
    if r["metric"] == "kad" and r["embedding"] == "clap":
        print(f"  {r['set']:16} {r['genre']:9} real {r['real']:.3f}  base {r['base']:.3f}  lora {r['lora']:.3f}"
              f"  delta {r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]  gap closed {r['gap_closed']:.0%}")
for r in paired:
    if r["metric"] in ("ast_p_target", "clap_score"):
        print(f"  {r['set']:16} {r['genre']:9} {r['metric']:12} real {r['real']:.3f}  base {r['base']:.3f}"
              f"  lora {r['lora']:.3f}  delta {r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]"
              f"  p={r['p_perm']:.4f}")
