"""Local web UI for generating music with Stable Audio 3."""

from __future__ import annotations

import argparse
import io
import random
import threading
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory

MODEL_NAME = "small-music"
SAMPLE_RATE = 44100
MAX_PROMPT_CHARS = 1000
MAX_SEED = 4294967295

# (min, max) for each setting. The page reads these from /api/config,
# so this is the only place to change them.
LIMITS: dict[str, tuple[float, float]] = {
    "duration": (1, 120),
    "steps": (1, 150),
    "cfg_scale": (1.0, 15.0),
}
DEFAULTS: dict[str, float] = {"duration": 30, "steps": 8, "cfg_scale": 1.0}
LABELS = {"duration": "Duration", "steps": "Steps", "cfg_scale": "CFG scale"}
SEED_ERROR = "Seed must be a whole number between 0 and 4294967295."

GenerateFn = Callable[..., np.ndarray]


def parse_params(data: Any, lora_names: Sequence[str] = ()) -> dict[str, Any]:
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

    seed_raw = data.get("seed")
    if seed_raw is None or seed_raw == "":
        seed = random.randrange(100_000)
    else:
        try:
            seed_value = float(seed_raw)
        except (TypeError, ValueError):
            raise ValueError(SEED_ERROR) from None
        if not seed_value.is_integer() or not 0 <= seed_value <= MAX_SEED:
            raise ValueError(SEED_ERROR)
        seed = int(seed_value)

    lora = data.get("lora")
    if lora is None or lora == "":
        lora = None
    elif lora not in lora_names:
        if lora_names:
            raise ValueError(f"Unknown LoRA {lora!r}. Loaded: {', '.join(lora_names)}.")
        raise ValueError(f"Unknown LoRA {lora!r}. No LoRAs are loaded.")

    if values["duration"].is_integer():
        values["duration"] = int(values["duration"])
    return {"prompt": prompt, "seed": seed, "lora": lora, **values}


def to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode an array shaped (frames, channels) as WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


def parse_lora_args(values: Sequence[str]) -> dict[str, str]:
    """Turn NAME=PATH (or a bare PATH) arguments into a name → path dict."""
    loras: dict[str, str] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep:
            path = name
            name = Path(path).stem
        name = name.strip()
        if not name:
            raise ValueError(f"LoRA needs a name: {value!r}")
        if name in loras:
            raise ValueError(f"Duplicate LoRA name: {name}")
        if not Path(path).is_file():
            raise ValueError(f"LoRA file not found: {path}")
        loras[name] = path
    return loras


class Generator:
    """Loads the model once and runs one generation at a time."""

    def __init__(self, loras: dict[str, str] | None = None) -> None:
        self.loras = loras or {}
        self.lora_names = list(self.loras)
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            from stable_audio_3 import StableAudioModel

            model = StableAudioModel.from_pretrained(MODEL_NAME)
            if self.loras:
                model.load_lora(list(self.loras.values()))
            self._model = model
        return self._model

    def preload(self) -> None:
        with self._lock:
            self._load()

    def generate(
        self,
        *,
        prompt: str,
        duration: float,
        steps: int,
        cfg_scale: float,
        seed: int,
        lora: str | None = None,
    ) -> np.ndarray:
        with self._lock:
            model = self._load()
            if self.loras:
                model.set_lora_strength(0.0)
                if lora is not None:
                    model.set_lora_strength(
                        1.0, lora_index=self.lora_names.index(lora)
                    )
            audio = model.generate(
                prompt=prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
            )
        return audio.detach().cpu().squeeze(0).T.numpy()


def create_app(
    generate_fn: GenerateFn | None = None, lora_names: Sequence[str] = ()
) -> Flask:
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
                "loras": list(lora_names),
                "model": MODEL_NAME,
            }
        )

    @app.post("/api/generate")
    def generate_route() -> Response | tuple[Response, int]:
        try:
            params = parse_params(request.get_json(silent=True), lora_names)
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
                        "Check the terminal running website.py for details."
                    }
                ),
                500,
            )

        headers = {"Cache-Control": "no-store", "X-Seed": str(params["seed"])}
        return Response(wav, mimetype="audio/wav", headers=headers)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="LoRA checkpoint (.ckpt or .safetensors); repeat for several. "
        "NAME defaults to the file name.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default 8000; macOS AirPlay Receiver uses 5000).",
    )
    args = parser.parse_args()

    if args.lora:
        try:
            loras = parse_lora_args(args.lora)
        except ValueError as error:
            parser.error(str(error))
    else:
        loras = {}
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            for filename in models_dir.iterdir():
                if filename.suffix == ".ckpt":
                    loras[filename.stem] = str(filename)

    if loras:
        print(f"Loading {MODEL_NAME} with LoRAs: {', '.join(loras)}...")
    else:
        print(f"Loading {MODEL_NAME}...")
    generator = Generator(loras)
    generator.preload()
    app = create_app(generator.generate, generator.lora_names)
    print(f"Ready at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
