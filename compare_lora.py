"""Generate the same prompts with the LoRA off (strength 0) and on (strength 1) for an A/B listen.

Run from inside the stable-audio-3 folder:
  uv run python ../compare_lora.py --lora lora_rock_test/<checkpoint>.ckpt
"""
import argparse
import os

import soundfile as sf
from stable_audio_3 import StableAudioModel

PROMPTS = [
    "Instrumental rock, featuring electric guitar, drums, bass, energetic mood, fast tempo",
    "Instrumental rock with indie influences, featuring electric guitar, melancholic mood, medium tempo",
    "Instrumental rock with pop influences, featuring acoustic guitar, drums, happy mood, medium tempo",
]

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--model", default="small-music")
ap.add_argument("--duration", type=int, default=30)
ap.add_argument("--out", default="lora_comparison")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
model = StableAudioModel.from_pretrained(args.model)
model.load_lora([args.lora])

for i, prompt in enumerate(PROMPTS):
    for strength in (0.0, 1.0):
        model.set_lora_strength(strength)
        # same seed for both, so the LoRA is the main difference
        audio = model.generate(prompt=prompt, duration=args.duration, seed=i)
        path = os.path.join(args.out, f"prompt{i}_{'lora' if strength else 'base'}.wav")
        sf.write(path, audio.detach().cpu().squeeze(0).T.numpy(), 44100)
        print("saved", path)
