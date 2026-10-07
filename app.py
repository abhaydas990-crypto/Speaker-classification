"""
Speaker + Animal sound classifier - web app backend.

Pipeline (must match the training notebook exactly):
    audio -> 3 s windows (1.5 s hop) -> YAMNet embeddings + MFCC stats -> StandardScaler -> Keras classifier

Run:  python app.py      then open http://localhost:5000
"""
from __future__ import annotations

import os
import json
import shutil
import logging
import tempfile
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TFHUB_CACHE_DIR", str(BASE_DIR / "tfhub_cache"))

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_from_directory, Response

# --------------------------------------------------------------------------- settings
SR = 16000
CHUNK_SEC = 3.0
HOP_SEC = 1.5
WINDOW_SAMPLES = int(CHUNK_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
SILENCE_RMS = 1e-3

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.50"))
MAX_ANALYZE_SEC = 60
MAX_WINDOWS = int(MAX_ANALYZE_SEC / HOP_SEC)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(BASE_DIR / "model")))
YAMNET_SOURCE = os.environ.get("YAMNET_PATH", "https://tfhub.dev/google/yamnet/1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audio-app")


class AudioError(ValueError):
    """Raised for any problem with the audio itself (silent, too short, undecodable)."""


# --------------------------------------------------------------------------- model container
@dataclass
class Preproc:
    classes: list[str]
    mean: np.ndarray
    scale: np.ndarray
    kind_of: dict[str, str]


class Engine:
    """Holds the loaded model/YAMNet/preprocessing so no module-level variable is ever `None`
    after startup - this keeps static type checkers (Pylance) happy and avoids
    'Object of type None cannot be called' warnings."""

    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.yamnet = None
        self.pre: Preproc | None = None
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.yamnet is not None and self.pre is not None

    def load(self) -> None:
        self.pre = self._load_preproc()
        log.info("Loading classifier from %s ...", MODEL_DIR / "model.keras")
        self.model = tf.keras.models.load_model(str(MODEL_DIR / "model.keras"), compile=False)
        log.info("Loading YAMNet (first run downloads it) ...")
        self.yamnet = hub.load(YAMNET_SOURCE)
        n_human = sum(1 for v in self.pre.kind_of.values() if v == "human")
        n_animal = sum(1 for v in self.pre.kind_of.values() if v == "animal")
        log.info("Ready: %d classes (%d people, %d animals)", len(self.pre.classes), n_human, n_animal)

    @staticmethod
    def _load_preproc() -> Preproc:
        preproc_json = MODEL_DIR / "preproc.json"
        if preproc_json.exists():
            data = json.loads(preproc_json.read_text())
            return Preproc(
                classes=[str(c) for c in data["classes"]],
                mean=np.array(data["mean"], dtype=np.float32),
                scale=np.array(data["scale"], dtype=np.float32),
                kind_of={str(k): v for k, v in data["kind_of"].items()},
            )
        import joblib  # fallback only; needs scikit-learn installed

        d = joblib.load(MODEL_DIR / "preproc.joblib")
        return Preproc(
            classes=[str(c) for c in d["le"].classes_],
            mean=d["scaler"].mean_.astype(np.float32),
            scale=d["scaler"].scale_.astype(np.float32),
            kind_of={str(k): v for k, v in d["kind_of"].items()},
        )

    # ---- feature extraction (identical to the training notebook) ----
    def embed(self, window: np.ndarray) -> np.ndarray:
        assert self.yamnet is not None, "YAMNet is not loaded yet"
        w = window.astype(np.float32)
        peak = float(np.max(np.abs(w)))
        if peak <= 0.0:
            raise AudioError("The uploaded audio is silent or contains no usable waveform data.")
        w = w / (peak + 1e-9) * 0.9
        _, emb, _ = self.yamnet(w)
        emb_np = emb.numpy()
        mfcc = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=40)
        return np.concatenate([emb_np.mean(0), emb_np.std(0), mfcc.mean(1), mfcc.std(1)])

    def predict_windows(self, windows: list[np.ndarray]) -> np.ndarray:
        assert self.model is not None and self.pre is not None, "Model is not loaded yet"
        feats = np.stack([self.embed(w) for w in windows])
        x = ((feats - self.pre.mean) / self.pre.scale).astype(np.float32)
        with self.lock:
            probs = self.model(x, training=False)
        return np.asarray(probs)


engine = Engine()


# --------------------------------------------------------------------------- audio decoding
def decode_audio(data: bytes, filename: str | None) -> np.ndarray:
    """Bytes (any common container) -> mono float32 @16 kHz numpy array."""
    if not data:
        raise AudioError("No audio data was received.")
    suffix = Path(filename).suffix if filename else ""
    suffix = suffix if suffix else ".bin"

    tmp_in = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_in_path = Path(tmp_in.name)
    tmp_in.write(data)
    tmp_in.close()
    tmp_out_path: Path | None = None

    try:
        try:
            y, _ = librosa.load(str(tmp_in_path), sr=SR, mono=True, duration=MAX_ANALYZE_SEC)
            if y.size > 0:
                return y
        except Exception as exc:  # noqa: BLE001
            log.info("librosa could not decode %s (%s); trying ffmpeg", filename, type(exc).__name__)

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            tmp_out_path = tmp_in_path.with_suffix(".decoded.wav")
            cmd = [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(tmp_in_path),
                   "-ac", "1", "-ar", str(SR), "-t", str(MAX_ANALYZE_SEC), str(tmp_out_path)]
            try:
                subprocess.run(cmd, check=True, timeout=90)
                y, _ = librosa.load(str(tmp_out_path), sr=SR, mono=True)
                if y.size > 0:
                    return y
            except Exception as exc:  # noqa: BLE001
                log.warning("ffmpeg fallback failed: %s", exc)

        raise AudioError("Could not decode this audio file. Try WAV, MP3, OGG or M4A, "
                          "or install ffmpeg for wider format support.")
    finally:
        for p in (tmp_in_path, tmp_out_path):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass


def make_windows(y: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """-> list of (start_time_sec, window) for every non-silent 3 s slice."""
    if len(y) < WINDOW_SAMPLES:
        starts = [0]
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    else:
        starts = list(range(0, len(y) - WINDOW_SAMPLES + 1, HOP_SAMPLES))
        if starts[-1] + WINDOW_SAMPLES < len(y) - SR // 2:
            starts.append(len(y) - WINDOW_SAMPLES)
        starts = starts[:MAX_WINDOWS]

    out: list[tuple[float, np.ndarray]] = []
    for s in starts:
        w = y[s:s + WINDOW_SAMPLES]
        if np.sqrt(np.mean(w.astype(np.float64) ** 2)) >= SILENCE_RMS:
            out.append((s / SR, w))
    return out


def analyze(y: np.ndarray) -> dict:
    if not engine.ready or engine.pre is None:
        raise RuntimeError("Model is still loading - try again in a few seconds.")
    if len(y) < int(0.3 * SR):
        raise AudioError("Audio is too short - please provide at least half a second of sound.")

    windows = make_windows(y)
    if not windows:
        raise AudioError("No sound detected - the audio is silent or too quiet.")

    times = [t for t, _ in windows]
    probs_w = engine.predict_windows([w for _, w in windows])
    pre = engine.pre
    classes, kind_of = pre.classes, pre.kind_of

    probs = probs_w.mean(axis=0)
    order = list(np.argsort(probs)[::-1])
    best = classes[order[0]]
    confidence = float(probs[order[0]])
    p_human = float(sum(p for p, c in zip(probs, classes) if kind_of[c] == "human"))

    windows_out = []
    for t, pw in zip(times, probs_w):
        i = int(np.argmax(pw))
        windows_out.append({
            "start": round(t, 2), "end": round(t + CHUNK_SEC, 2),
            "label": classes[i], "kind": kind_of[classes[i]], "confidence": float(pw[i]),
        })

    return {
        "label": best,
        "kind": kind_of[best],
        "confidence": confidence,
        "uncertain": confidence < CONF_THRESHOLD,
        "p_human": p_human,
        "p_animal": 1.0 - p_human,
        "top": [{"label": classes[i], "kind": kind_of[classes[i]], "prob": float(probs[i])} for i in order[:5]],
        "windows": windows_out,
        "duration": round(len(y) / SR, 2),
    }


# --------------------------------------------------------------------------- Flask app
app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.get("/")
def index() -> Response:
    return send_from_directory(app.static_folder or "static", "index.html")


@app.get("/api/info")
def info() -> Response:
    if not engine.ready or engine.pre is None:
        return jsonify(ready=False, people=[], animals=[], threshold=CONF_THRESHOLD, max_seconds=MAX_ANALYZE_SEC)
    pre = engine.pre
    return jsonify(
        ready=True,
        people=sorted(c for c in pre.classes if pre.kind_of[c] == "human"),
        animals=sorted(c for c in pre.classes if pre.kind_of[c] == "animal"),
        threshold=CONF_THRESHOLD,
        max_seconds=MAX_ANALYZE_SEC,
    )


@app.post("/api/predict")
def predict() -> tuple[Response, int] | Response:
    uploaded = request.files.get("audio")
    if uploaded is None:
        return jsonify(error="No audio received."), 400
    if not engine.ready:
        return jsonify(error="Model is still loading - try again in a few seconds."), 503
    try:
        y = decode_audio(uploaded.read(), uploaded.filename)
        return jsonify(analyze(y))
    except AudioError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # noqa: BLE001
        log.exception("prediction failed")
        return jsonify(error=f"Internal error: {exc}"), 500


def main() -> None:
    engine.load()
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n  Open  http://localhost:{port}  in your browser\n")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
