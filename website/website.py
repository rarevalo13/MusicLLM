"""Local web UI for generating music with Stable Audio 3."""

from __future__ import annotations

import io
import threading
from typing import Any, Callable

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory

MODEL_NAME = "small-music"
SAMPLE_RATE = 44100
MAX_PROMPT_CHARS = 1000

# (min, max) for each setting. The page reads these from /api/config,
# so this is the only place to change them.
LIMITS: dict[str, tuple[float, float]] = {
    "duration": (1, 180),
    "steps": (1, 150),
    "cfg_scale": (1.0, 15.0),
}
DEFAULTS: dict[str, float] = {"duration": 30, "steps": 50, "cfg_scale": 7.0}
LABELS = {"duration": "Duration", "steps": "Steps", "cfg_scale": "CFG scale"}

GenerateFn = Callable[..., np.ndarray]


def parse_params(data: Any) -> dict[str, Any]:
    """Validate a request body and return keyword arguments for the model."""
    if not isinstance(data, dict):
        raise ValueError("Send the settings as a JSON object.")

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Write a prompt first.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Keep the prompt under {MAX_PROMPT_CHARS} characters.")

    try:
        values = {
            "duration": float(data.get("duration", DEFAULTS["duration"])),
            "steps": int(float(data.get("steps", DEFAULTS["steps"]))),
            "cfg_scale": float(data.get("cfg_scale", DEFAULTS["cfg_scale"])),
        }
    except (TypeError, ValueError):
        raise ValueError("Duration, steps and CFG scale must be numbers.") from None

    for key, value in values.items():
        low, high = LIMITS[key]
        if not low <= value <= high:
            raise ValueError(f"{LABELS[key]} must be between {low:g} and {high:g}.")

    if values["duration"].is_integer():
        values["duration"] = int(values["duration"])
    return {"prompt": prompt, **values}


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode an array shaped (frames, channels) as WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


class Generator:
    """Loads the model once and runs one generation at a time."""

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            from stable_audio_3 import StableAudioModel

            self._model = StableAudioModel.from_pretrained(MODEL_NAME)
        return self._model

    def preload(self) -> None:
        with self._lock:
            self._load()

    def generate(
        self, *, prompt: str, duration: float, steps: int, cfg_scale: float
    ) -> np.ndarray:
        with self._lock:
            audio = self._load().generate(
                prompt=prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
            )
        return audio.detach().cpu().squeeze(0).T.numpy()


def create_app(generate_fn: GenerateFn | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    generate = generate_fn or Generator().generate

    @app.get("/")
    def index() -> Response:
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/config")
    def config() -> Response:
        return jsonify(
            {
                "limits": {key: list(bounds) for key, bounds in LIMITS.items()},
                "defaults": DEFAULTS,
                "maxPromptChars": MAX_PROMPT_CHARS,
            }
        )

    @app.post("/api/generate")
    def generate_route() -> Response | tuple[Response, int]:
        try:
            params = parse_params(request.get_json(silent=True))
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

        try:
            samples = generate(**params)
            wav = to_wav_bytes(samples)
        except Exception:
            app.logger.exception("Generation failed")
            return (
                jsonify(
                    {
                        "error": "The model failed while generating. "
                        "Check the terminal running app.py for details."
                    }
                ),
                500,
            )

        return Response(wav, mimetype="audio/wav", headers={"Cache-Control": "no-store"})

    return app


def main() -> None:
    generator = Generator()
    print(f"Loading {MODEL_NAME}...")
    generator.preload()
    app = create_app(generator.generate)
    print("Ready at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)


if __name__ == "__main__":
    main()
