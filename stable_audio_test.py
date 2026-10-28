import soundfile as sf
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("small-music")
audio = model.generate(
    prompt=(
        "Rap music that encapsulates the feeling of driving slowly after a long day "
        "in the sunny weather with all your friends 90 BPM"
    ),
    duration=30,
    steps=50,
    cfg_scale=7.0,
)
sf.write("festival_house.wav", audio.detach().cpu().squeeze(0).T.numpy(), 44100)
