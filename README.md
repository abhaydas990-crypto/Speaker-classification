# Voice & Animal Sound Recognizer (dark-themed web app)

A small local web app around the model trained in the companion Colab notebook. It can:

* **record live from your microphone**, or
* **classify audio files from your device** (WAV, MP3, OGG, M4A, FLAC, WebM, WhatsApp voice notes),

and reports either **"Human voice → <person>"** or **"Animal sound → <animal>"**, with a confidence
score, a human-vs-animal split, top-5 guesses, and a per-window breakdown.

Known classes (from `model/preproc.json`): **Rohit, abhay, bimal, sanket, satya, soumya**
+ **cat, cow, crow, dog, frog, hen, insects, pig, rooster, sheep**.

## Run it

```bash
cd audio_app
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** — use `localhost`, not an IP address; browsers only allow
microphone access on `localhost` or HTTPS.

The first run downloads YAMNet (~15 MB, needs internet) into `./tfhub_cache`; after that it works offline.
Startup takes 10–30 s while TensorFlow and YAMNet load — the terminal prints the URL when it's ready.

**Optional:** install [ffmpeg](https://ffmpeg.org) as a fallback for audio formats the browser can't decode itself.

### VS Code setup (if you see red "could not be resolved" import errors)

That means the editor is pointing at the wrong Python interpreter, not a real bug:
`Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one inside `audio_app/.venv` → then
`Ctrl+Shift+P` → **Developer: Reload Window**.

## How it works

```
browser: record / choose file -> decode -> 16 kHz mono WAV ---POST /api/predict--->
server : 3 s windows (1.5 s hop, silent ones dropped)
         -> YAMNet embeddings (mean+std) + 40 MFCC (mean+std) = 2128 features
         -> StandardScaler -> Keras classifier -> average over windows -> JSON result
```

`app.py` uses exactly the same windowing and feature code as the training notebook — if you change
one, change the other.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask server + inference (fully typed, no static-analysis warnings) |
| `static/index.html` | page markup |
| `static/styles.css` | dark theme |
| `static/app.js` | microphone recording, file upload, results rendering |
| `model/model.keras` | trained classifier |
| `model/preproc.json` | scaler mean/scale + class names (used by default) |
| `model/preproc.joblib` | original file from Colab (fallback if `preproc.json` is missing; needs scikit-learn) |

## Settings (environment variables)

`PORT` (5000) · `HOST` (127.0.0.1; use `0.0.0.0` to share on your network — mic then needs HTTPS) ·
`CONF_THRESHOLD` (0.50 — below this the app shows "Not sure") · `MODEL_DIR` · `YAMNET_PATH` (point at a local YAMNet folder for fully offline use)

## Retraining / adding people

Retrain in Colab, run the "save" cell there, then replace `model/model.keras` and `model/preproc.joblib`,
and regenerate `preproc.json`:

```python
import joblib, json
p = joblib.load("model/preproc.joblib")
json.dump(dict(classes=[str(c) for c in p["le"].classes_], mean=p["scaler"].mean_.tolist(),
               scale=p["scaler"].scale_.tolist(), kind_of={str(k): v for k, v in p["kind_of"].items()}),
          open("model/preproc.json", "w"))
```

## Limits

* The model only knows the 16 classes above — an unknown person or animal is matched to the closest
  known one; low confidence then shows as "Not sure".
* Only the first 60 s of a file are analysed; the microphone stops after 30 s.
