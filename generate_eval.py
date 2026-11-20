"""Generate the main evaluation set for one genre.

For every held-out clip in <heldout>/<genre>/ (<track>_<chunk>.wav + same-named .txt,
written by make_latents.slurm), generate one clip with the LoRA off and one with it
on, prompted by the clip's own caption with the same seed, into
<out>/<genre>/{base,lora}/ plus manifest.csv. Resumable: existing WAVs are skipped
and a manifest row is appended after each file lands, so rerunning the same command
finishes the set.

On Hawk, from the repo root (one genre per job):
  sbatch generate_eval.slurm rock
  ./hawk_container.sh python generate_eval.py --genre rock --lora runs/rock/<checkpoint>.ckpt
"""
import argparse
import csv
import glob
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel

FIELDS = ["genre", "clip_id", "track", "chunk", "condition", "strength", "seed",
          "caption", "model", "lora", "steps", "cfg_scale", "duration", "file",
          "gen_seconds"]

ap = argparse.ArgumentParser()
ap.add_argument("--genre", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--heldout", default="heldout_clips")
ap.add_argument("--out", default="eval_gen")
ap.add_argument("--model", default="small-music")
ap.add_argument("--duration", type=int, default=30)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--cfg-scale", type=float, default=1.0)
ap.add_argument("--rounds", type=int, default=None,
                help="only clips with chunk index < N (default: all)")
ap.add_argument("--limit", type=int, default=None,
                help="only the first N clips in processing order (smoke tests)")
args = ap.parse_args()

heldout_dir = os.path.join(args.heldout, args.genre)
out_dir = os.path.join(args.out, args.genre)
manifest_path = os.path.join(out_dir, "manifest.csv")

wav_paths = sorted(glob.glob(os.path.join(heldout_dir, "*.wav")))
if not wav_paths:
    sys.exit(f"error: no WAVs in {heldout_dir}")

clips = []
missing_txt = []
for path in wav_paths:
    clip_id = os.path.splitext(os.path.basename(path))[0]
    track, chunk = clip_id.rsplit("_", 1)
    chunk = int(chunk)
    txt_path = os.path.join(heldout_dir, clip_id + ".txt")
    if not os.path.exists(txt_path):
        missing_txt.append(clip_id)
        continue
    with open(txt_path) as f:
        caption = f.read().strip()
    clips.append({
        "clip_id": clip_id,
        "track": track,
        "chunk": chunk,
        "caption": caption,
        "seed": zlib.crc32(clip_id.encode()) & 0x7FFFFFFF,
    })
if missing_txt:
    sys.exit(f"error: {len(missing_txt)} held-out WAVs have no .txt caption "
             f"(e.g. {', '.join(missing_txt[:5])})")

# processing order: every track's chunk 0 first, then chunk 1, ... so stopping at
# any point still leaves base and LoRA covering the same clips
clips.sort(key=lambda c: (c["chunk"], c["track"]))
all_clips = clips
if args.rounds is not None:
    clips = [c for c in clips if c["chunk"] < args.rounds]
if args.limit is not None:
    clips = clips[: args.limit]


def out_path(clip, condition):
    return os.path.join(out_dir, condition, clip["clip_id"] + ".wav")


def clip_done(clip):
    return (os.path.exists(out_path(clip, "base"))
            and os.path.exists(out_path(clip, "lora")))


def make_row(clip, condition, strength, gen_seconds=""):
    return {
        "genre": args.genre,
        "clip_id": clip["clip_id"],
        "track": clip["track"],
        "chunk": str(clip["chunk"]),
        "condition": condition,
        "strength": str(strength),
        "seed": str(clip["seed"]),
        "caption": clip["caption"],
        "model": str(args.model),
        "lora": str(args.lora),
        "steps": str(args.steps),
        "cfg_scale": str(args.cfg_scale),
        "duration": str(args.duration),
        "file": f"{condition}/{clip['clip_id']}.wav",
        "gen_seconds": gen_seconds,
    }


# manifest repair: the manifest always reflects the WAVs on disk for every held-out
# clip of this genre, with rows written under the current settings only
settings = {k: str(getattr(args, k))
            for k in ("model", "lora", "steps", "cfg_scale", "duration")}
rows = []
if os.path.exists(manifest_path):
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
for row in rows:
    bad = [k for k in settings if row.get(k) != settings[k]]
    if bad:
        sys.exit(f"error: {manifest_path} was made with different settings "
                 f"({', '.join(bad)} differ); use another --out")

kept = {}
for row in rows:
    if row.get("file") and os.path.exists(os.path.join(out_dir, row["file"])):
        kept[(row["clip_id"], row["condition"])] = row

have = {(c["clip_id"], cond) for c in all_clips for cond in ("base", "lora")
        if os.path.exists(out_path(c, cond))}
reconstructed = [make_row(c, cond, 0.0 if cond == "base" else 1.0)
                 for c in all_clips for cond in ("base", "lora")
                 if (c["clip_id"], cond) in have
                 and (c["clip_id"], cond) not in kept]

os.makedirs(out_dir, exist_ok=True)
for condition in ("base", "lora"):
    os.makedirs(os.path.join(out_dir, condition), exist_ok=True)
tmp_manifest = manifest_path + ".tmp"
with open(tmp_manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for row in list(kept.values()) + reconstructed:
        w.writerow(row)
os.replace(tmp_manifest, manifest_path)

todo = [c for c in clips if not clip_done(c)]
if not todo:
    on_disk = {cond: sum(1 for c in all_clips
                         if os.path.exists(out_path(c, cond)))
               for cond in ("base", "lora")}
    remaining = sum(1 for c in all_clips if not clip_done(c))
    status = "complete" if remaining == 0 else f"{remaining} clips remain"
    print(f"nothing to generate: all {len(clips)} selected clips already have "
          f"both files")
    print(f"on disk vs {len(all_clips)} held-out clips: base {on_disk['base']}, "
          f"lora {on_disk['lora']} ({status})")
    print(f"manifest: {manifest_path}")
    sys.exit(0)

model = StableAudioModel.from_pretrained(args.model)
model.load_lora([args.lora])
print(f"device={model.device} model={args.model} lora={args.lora} "
      f"steps={args.steps} cfg_scale={args.cfg_scale} duration={args.duration}s")
print(f"heldout={heldout_dir} ({len(all_clips)} clips) out={out_dir}")

per_chunk = {}
for c in clips:
    e = per_chunk.setdefault(c["chunk"], [0, 0, 0])
    e[0] += 1
    if clip_done(c):
        e[1] += 1
    else:
        e[2] += 1
for chunk in sorted(per_chunk):
    total, done, left = per_chunk[chunk]
    print(f"chunk {chunk}: {total} clips, {done} done, {left} to do")
print(f"{len(todo)} of {len(clips)} selected clips to do")

manifest_f = open(manifest_path, "a", newline="")
manifest_w = csv.DictWriter(manifest_f, fieldnames=FIELDS, extrasaction="ignore")


def record(row):
    manifest_w.writerow(row)
    manifest_f.flush()


made = {"base": 0, "lora": 0}
gen_times = {"base": [], "lora": []}
first_call = True
n = len(clips)
for i, clip in enumerate(clips, 1):
    parts = []
    wavs = {}
    for condition, strength in (("base", 0.0), ("lora", 1.0)):
        path = out_path(clip, condition)
        if os.path.exists(path):
            parts.append(f"{condition} exists")
            continue
        model.set_lora_strength(strength)
        t0 = time.perf_counter()
        audio = model.generate(prompt=clip["caption"], duration=args.duration,
                               steps=args.steps, cfg_scale=args.cfg_scale,
                               seed=clip["seed"])
        wav = audio.detach().cpu().squeeze(0).T.numpy()
        elapsed = time.perf_counter() - t0
        tmp = path + ".part"
        sf.write(tmp, wav, model.model.sample_rate, format="WAV",
                 subtype="PCM_16")
        os.replace(tmp, path)
        record(make_row(clip, condition, strength, f"{elapsed:.2f}"))
        made[condition] += 1
        if first_call:
            first_call = False
        else:
            gen_times[condition].append(elapsed)
        wavs[condition] = wav
        parts.append(f"{condition} {elapsed:.1f} s")
    if ("base" in wavs and "lora" in wavs
            and np.array_equal(wavs["base"], wavs["lora"])):
        sys.exit("error: LoRA output identical to base: check --lora")
    print(f"[{i}/{n}] chunk {clip['chunk']} {clip['clip_id']}: "
          f"{', '.join(parts)}")
manifest_f.close()

print("summary:")
for condition in ("base", "lora"):
    times = gen_times[condition]
    mean = f"{sum(times) / len(times):.2f} s" if times else "n/a"
    on_disk = sum(1 for c in all_clips
                  if os.path.exists(out_path(c, condition)))
    print(f"  {condition}: {made[condition]} generated this run, "
          f"mean {mean} per clip (first generate() of the run excluded); "
          f"{on_disk}/{len(all_clips)} held-out clips on disk")
remaining = sum(1 for c in all_clips if not clip_done(c))
if remaining:
    print(f"  {remaining} held-out clips still missing base and/or lora files")
else:
    print("  complete: every held-out clip has both files")
if model.device == "cuda":
    print(f"  peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
