"""Extract per-clip features for the base-vs-LoRA evaluation (GPU node; eval_metrics.py uses them).

For one genre and each set in --sets:
  real       <heldout>/<genre>/*.wav, captions from the matching .txt files
  base/lora  <set>/<genre>/{base,lora}/*.wav from generate_eval.py, captions from its manifest.csv
Per clip: LAION CLAP (larger_clap_music_and_speech) audio embedding (mean of 10 s windows, L2-normalised) and its cosine
with the CLAP embedding of the clip's caption (CLAP score); AST AudioSet logits, tag probabilities
(mean sigmoid) and pooled embedding (means over 10 s windows); long-term average spectrum (dB);
RMS level, peak, clipped-sample fraction and side/mid energy ratio.
Writes <out>/<set name>/<genre>/<condition>.npz (an existing file is kept unless --force) and,
once, <out>/clap_text_genres.npz (CLAP text embeddings of the four genre names, for zero-shot
tagging). A generated folder that doesn't hold one WAV per held-out clip (still generating) is
skipped with a warning unless --allow-partial; rerun once it's done.

On Hawk: sbatch eval_features.slurm rock     (all sets of one genre)
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from transformers import (ASTFeatureExtractor, ASTForAudioClassification, AutoTokenizer,
                          ClapFeatureExtractor, ClapModel)

# not laion/larger_clap_music: it maps every caption to almost the same text embedding (cosine 0.999)
CLAP_ID = "laion/larger_clap_music_and_speech"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 44100
N_FFT = 2048
GENRE_PROMPTS = {"rock": "rock music", "hiphop": "hip hop music",
                 "classical": "classical music", "house": "house music"}
SETS = ["heldout_clips", "eval_gen", "eval_gen_steps4", "eval_gen_steps16",
        "eval_gen_steps32", "eval_gen_steps50", "eval_gen_base50"]

ap = argparse.ArgumentParser()
ap.add_argument("--genre", required=True)
ap.add_argument("--sets", nargs="+", default=SETS, help="folders; the one equal to --heldout is 'real'")
ap.add_argument("--heldout", default="heldout_clips")
ap.add_argument("--out", default="eval_feats")
ap.add_argument("--batch", type=int, default=16, help="clips per model batch")
ap.add_argument("--workers", type=int, default=4 if sys.platform.startswith("linux") else 0,
                help="audio loading processes (0 off Linux: spawned workers would re-run this script)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--limit", type=int, default=None, help="first N clips only (tests; output is partial)")
ap.add_argument("--force", action="store_true")
ap.add_argument("--allow-partial", action="store_true")
args = ap.parse_args()


def clip_id(path):
    return os.path.splitext(os.path.basename(path))[0]


heldout_dir = os.path.join(args.heldout, args.genre)
heldout_ids = sorted(clip_id(p) for p in glob.glob(os.path.join(heldout_dir, "*.wav")))
if not heldout_ids:
    sys.exit(f"error: no WAVs in {heldout_dir}")

jobs = []  # (set name, condition, [(clip id, path, caption)], output path)
for folder in args.sets:
    name = os.path.basename(os.path.normpath(folder))
    if os.path.normpath(folder) == os.path.normpath(args.heldout):
        clips = []
        for c in heldout_ids:
            with open(os.path.join(heldout_dir, c + ".txt")) as f:
                clips.append((c, os.path.join(heldout_dir, c + ".wav"), f.read().strip()))
        conditions = [("real", clips)]
    else:
        manifest = os.path.join(folder, args.genre, "manifest.csv")
        if not os.path.exists(manifest):
            print(f"warning: no {manifest}; skipping {folder}")
            continue
        with open(manifest, newline="") as f:
            caption = {r["clip_id"]: r["caption"] for r in csv.DictReader(f)}
        conditions = []
        for cond in ("base", "lora"):
            paths = sorted(glob.glob(os.path.join(folder, args.genre, cond, "*.wav")))
            ids = [clip_id(p) for p in paths]
            missing = [c for c in ids if c not in caption]
            if missing:
                sys.exit(f"error: {len(missing)} clips in {folder}/{args.genre}/{cond} have no "
                         f"manifest row (e.g. {missing[0]})")
            conditions.append((cond, [(c, p, caption[c]) for c, p in zip(ids, paths)]))
        incomplete = [f"{c} {len(clips)}" for c, clips in conditions if [x[0] for x in clips] != heldout_ids]
        if incomplete and not (args.allow_partial or args.limit):
            print(f"warning: skipping {folder}/{args.genre}: {', '.join(incomplete)} of "
                  f"{len(heldout_ids)} held-out clips (still generating?)")
            continue
    for cond, clips in conditions:
        out = os.path.join(args.out, name, args.genre, cond + ".npz")
        if os.path.exists(out) and not args.force:
            print(f"exists, skipping: {out}")
            continue
        jobs.append((name, cond, clips[:args.limit] if args.limit else clips, out))

genre_text = os.path.join(args.out, "clap_text_genres.npz")
if not jobs and os.path.exists(genre_text):
    print("nothing to do")
    sys.exit(0)


def windows(x, n):
    # consecutive n-sample windows; a 30 s clip gives exactly 3 (10 s each), so CLAP never crops at random
    k = max(1, round(len(x) / n))
    return [x[j * n:(j + 1) * n].numpy() for j in range(k)]


class Clips(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.clap_fe = ClapFeatureExtractor.from_pretrained(CLAP_ID)
        self.ast_fe = ASTFeatureExtractor.from_pretrained(AST_ID)
        self.window = torch.hann_window(N_FFT)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x, sr = sf.read(self.paths[i], dtype="float32", always_2d=True)
        if sr != SR:
            raise ValueError(f"{self.paths[i]}: {sr} Hz, expected {SR}")
        x = torch.from_numpy(x.T.copy())
        x = x.repeat(2, 1) if x.shape[0] == 1 else x[:2]
        mono = x.mean(0)
        clap_in = self.clap_fe(windows(AF.resample(mono, SR, 48000), 480000),
                               sampling_rate=48000, return_tensors="pt")
        ast_in = self.ast_fe(windows(AF.resample(mono, SR, 16000), 160000),
                             sampling_rate=16000, return_tensors="pt")
        spec = torch.stft(mono, N_FFT, N_FFT // 2, window=self.window, return_complex=True)
        ltas = 10 * torch.log10(spec.abs().pow(2).mean(1) / self.window.pow(2).sum() + 1e-12)
        mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
        stats = torch.stack([10 * torch.log10(x.pow(2).mean() + 1e-12), x.abs().max(),
                             (x.abs() >= 0.999).float().mean(),
                             10 * torch.log10((side.pow(2).mean() + 1e-12) / (mid.pow(2).mean() + 1e-12))])
        # ltas/stats are kept for the whole folder: numpy, so they don't each hold a shared-memory file descriptor
        return (clap_in["input_features"].float(), clap_in["is_longer"], ast_in["input_values"],
                ltas.numpy(), stats.numpy())


def collate(batch):
    return batch


def worker_init(_):
    torch.set_num_threads(1)


dev = torch.device(args.device)
clap = ClapModel.from_pretrained(CLAP_ID).to(dev).eval()
ast = ASTForAudioClassification.from_pretrained(AST_ID).to(dev).eval()
tok = AutoTokenizer.from_pretrained(CLAP_ID)
ast_labels = np.array([ast.config.id2label[i] for i in range(ast.config.num_labels)])
freqs = np.fft.rfftfreq(N_FFT, 1 / SR).astype(np.float32)
print(f"device={dev} clap={CLAP_ID} ast={AST_ID} genre={args.genre}")


@torch.inference_mode()
def text_embed(texts):
    out = []
    for i in range(0, len(texts), 64):
        t = tok(texts[i:i + 64], padding=True, truncation=True, return_tensors="pt").to(dev)
        h = clap.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output
        out.append(F.normalize(clap.text_projection(h), dim=-1).cpu())
    return torch.cat(out)


@torch.inference_mode()
def extract(paths, label):
    loader = torch.utils.data.DataLoader(Clips(paths), batch_size=args.batch, num_workers=args.workers,
                                         collate_fn=collate, worker_init_fn=worker_init)
    res = {k: [] for k in ("clap", "ast_emb", "ast_logits", "ast_prob", "ltas_db", "stats")}
    for batch in loader:
        a = clap.audio_model(input_features=torch.cat([b[0] for b in batch]).to(dev),
                             is_longer=torch.cat([b[1] for b in batch]).to(dev)).pooler_output
        a = F.normalize(clap.audio_projection(a), dim=-1).cpu()
        pooled = ast.audio_spectrogram_transformer(
            input_values=torch.cat([b[2] for b in batch]).to(dev)).pooler_output
        logits = ast.classifier(pooled).cpu()
        pooled = pooled.cpu()
        i = j = 0
        for b in batch:
            k, m = b[0].shape[0], b[2].shape[0]
            res["clap"].append(F.normalize(a[i:i + k].mean(0), dim=-1).numpy())
            res["ast_emb"].append(pooled[j:j + m].mean(0).numpy())
            res["ast_logits"].append(logits[j:j + m].mean(0).numpy())
            res["ast_prob"].append(torch.sigmoid(logits[j:j + m]).mean(0).numpy())
            res["ltas_db"].append(b[3])
            res["stats"].append(b[4])
            i, j = i + k, j + m
        n = len(res["clap"])
        if n % 160 < len(batch) or n == len(paths):
            print(f"  {label}: {n}/{len(paths)}")
    return {k: np.stack(v).astype(np.float32) for k, v in res.items()}


def save(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-len(".npz")] + f".{os.getpid()}.tmp.npz"  # per process: genre jobs run side by side
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


genres = list(GENRE_PROMPTS)
genre_emb = text_embed([GENRE_PROMPTS[g] for g in genres])
if (genre_emb @ genre_emb.T)[~torch.eye(len(genres), dtype=torch.bool)].mean() > 0.95:
    sys.exit(f"error: {CLAP_ID} gives near-identical text embeddings for different genre names")
if not os.path.exists(genre_text) or args.force:
    save(genre_text, genres=np.array(genres), prompts=np.array([GENRE_PROMPTS[g] for g in genres]),
         emb=genre_emb.numpy(), logit_scale_a=np.float32(clap.logit_scale_a.exp().item()))

for name, cond, clips, out in jobs:
    ids, paths, captions = map(list, zip(*clips))
    feats = extract(paths, f"{name}/{args.genre}/{cond}")
    unique = sorted(set(captions))
    pos = {c: i for i, c in enumerate(unique)}
    text = text_embed(unique)[[pos[c] for c in captions]].numpy()
    stats = feats.pop("stats")
    save(out, ids=np.array(ids), captions=np.array(captions), **feats,
         clap_score=(feats["clap"] * text).sum(1),
         rms_db=stats[:, 0], peak=stats[:, 1], clip_frac=stats[:, 2], side_db=stats[:, 3],
         ast_labels=ast_labels, ltas_freqs=freqs, clap_model=np.array(CLAP_ID), ast_model=np.array(AST_ID))
    print(f"saved {out} ({len(ids)} clips)")
