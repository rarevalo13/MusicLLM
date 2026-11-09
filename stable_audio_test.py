import soundfile as sf
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("small-music")
audio = model.generate(
    prompt=(
        "Neo soul lofi music, with a 120 BPM"
  
    ),
    duration=30,
    steps=50,
    cfg_scale=5.0,
)
sf.write("neo_soul.wav", audio.detach().cpu().squeeze(0).T.numpy(), 44100)
