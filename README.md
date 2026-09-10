# SignLink — Real-Time Sign Language Detector

A real-time sign language recognition system that reads your webcam, detects hand signs live in the browser, and turns the detected signs into readable English. It combines **two machine-learning models** — one for static hand shapes (letters & digits) and one for dynamic gesture words — with an AI layer that stitches detected words into natural sentences.

Built with **FastAPI + WebSockets** on the backend, **MediaPipe** for hand tracking, **scikit-learn** and **TensorFlow/Keras** for the models, and a browser frontend that streams webcam frames to the server.

---

## Features

- **Two detection modes**, switchable live:
  - **Static mode** — recognizes fingerspelling: letters **A–Z** and digits **0–9** (36 classes) using a Random Forest classifier.
  - **Dynamic mode** — recognizes full-motion **word gestures** (16 words) using an LSTM sequence model.
- **Live webcam streaming** over WebSockets — frames are processed and results returned in real time.
- **Hand landmark tracking** via MediaPipe (up to 2 hands, 21 landmarks each).
- **Accuracy-first smoothing** — majority-voting buffers and confirmation thresholds reduce jitter and false detections.
- **Quality gating** — poorly visible or too-small hands are rejected before prediction.
- **AI transcription** — detected words are converted into grammatically natural sentences using a hosted LLM (Mistral-7B via Hugging Face), with a rule-based fallback when the API is unavailable.
- **Health endpoint** reporting which models are loaded.

### Recognized dynamic words

`bye`, `doctor`, `dumb & deaf`, `food`, `hospital`, `language`, `man`, `medicine`, `namaste`, `pain`, `sign`, `sorry`, `thankyou`, `water`, `welcome`, `woman`

---

## How it works

```
Webcam frame (browser)
        │  base64 over WebSocket
        ▼
   MediaPipe Hands ──► 21 landmarks × up to 2 hands (84 features)
        │
        ├── STATIC mode ──► Random Forest ──► letter / digit
        │
        └── DYNAMIC mode ─► buffer 30 frames ─► LSTM ──► word
        │
        ▼
  Smoothing + confirmation buffers (majority vote)
        │
        ▼
  Confirmed word/letter ──► AI Transcriber ──► natural sentence
        │
        ▼
   Result sent back to browser (bbox, landmarks, text, AI sentence)
```

**Feature extraction (shared by both models):** for each hand, the 21 landmark `(x, y)` coordinates are normalized relative to the hand's top-left corner. Two hands = 84 features. Missing hands are zero-padded so the feature vector is always length 84.

- **Static:** a single 84-feature vector → Random Forest → one of 36 classes.
- **Dynamic:** a rolling window of **30 frames** × 84 features → LSTM → one of the word classes.

---

## Project structure

```
signlanguagedetector/
├── main.py                     # FastAPI server + WebSocket detection pipeline
├── ai_transcription.py         # Signs → natural sentences (LLM + rule-based fallback)
├── static/
│   └── index.html              # Browser frontend (webcam UI)
│
├── collect_imgs.py             # Capture images for STATIC classes (letters/digits)
├── collect_imgs_dynamic.py     # Record video sequences for DYNAMIC word classes
├── create_dataset_static.py    # Extract landmarks from images  → data.pickle
├── create_dataset_dynamic.py   # Extract landmarks from sequences → sequences.pickle
├── train_classifier.py         # Train Random Forest (static)   → model_enhanced.p
├── train_lstm.py               # Train LSTM (dynamic) → lstm_model.h5 + label_encoder.pickle
│
├── inference_static.py         # Standalone webcam test for static model
├── inference_dynamic.py        # Standalone webcam test for dynamic model
├── tf.py                       # TensorFlow helper/experiment script
│
├── data/                       # Raw static training images (per-class folders)
├── dynamic_data/               # Raw dynamic training sequences (per-class folders)
│
├── data.pickle                 # Processed static features
├── sequences.pickle            # Processed dynamic sequences
├── model_enhanced.p            # Trained Random Forest model
├── lstm_model.h5               # Trained LSTM model
├── label_encoder.pickle        # LSTM label encoder
└── requirements.txt
```

---

## Getting started

### Prerequisites

- **Python 3.10 or 3.11** (required by TensorFlow 2.15 / MediaPipe)
- A working **webcam**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your Hugging Face API key

The AI transcription step calls a hosted model. Provide a key via environment variable — if none is set (or the call fails), the app automatically falls back to rule-based sentence formation.

```bash
# Windows PowerShell
$env:HF_API_KEY = "your_huggingface_token"

# macOS / Linux
export HF_API_KEY="your_huggingface_token"
```

You can also put it in a `.env` file (`python-dotenv` is installed).

### 3. Run the server

```bash
python main.py
```

The server starts on **http://127.0.0.1:8000**. Open that URL in your browser to launch the webcam interface.

### 4. Use it

- Allow camera access when prompted.
- Switch between **static** and **dynamic** modes in the UI.
- Sign in front of the camera — detected letters/words and the AI-generated sentence appear live.

---

## API endpoints

| Method | Path      | Description |
|--------|-----------|-------------|
| `GET`  | `/`       | Serves the web frontend (`static/index.html`) |
| `GET`  | `/health` | Reports server status and which models are loaded |
| `WS`   | `/ws`     | WebSocket for live frame streaming and detection |

**WebSocket message types (client → server):**

- `{"type": "frame", "data": "<base64 image>"}` — send a webcam frame for detection
- `{"type": "switch_mode", "mode": "static" | "dynamic"}` — change detection mode
- `{"type": "reset"}` — clear all detected text/words

---

## Training your own models

The repo ships with pre-trained models, but you can retrain or add new signs.

### Static model (letters / digits)

```bash
python collect_imgs.py            # 1. capture images per class from webcam
python create_dataset_static.py   # 2. extract hand landmarks  → data.pickle
python train_classifier.py        # 3. train Random Forest      → model_enhanced.p
```

### Dynamic model (word gestures)

```bash
python collect_imgs_dynamic.py    # 1. record 30-frame sequences per word
python create_dataset_dynamic.py  # 2. extract landmarks         → sequences.pickle
python train_lstm.py              # 3. train LSTM → lstm_model.h5 + label_encoder.pickle
```

**Notes**
- Data collection scripts resume from the highest existing class folder, so you can add classes incrementally.
- Both models are trained on **CPU** (`CUDA_VISIBLE_DEVICES=-1`); no GPU required.
- The LSTM uses a 3-layer stacked LSTM with dropout, early stopping, and up to 100 epochs.

---

## Tuning

Key parameters live at the top of `main.py`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SEQUENCE_LENGTH` | `30` | Frames per dynamic gesture |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum model confidence to accept a prediction |
| `PREDICTION_BUFFER_SIZE` | `15` | Buffer size for majority-vote smoothing |

MediaPipe detection/tracking confidence and hand-visibility gating can also be adjusted in `main.py`.

---

## Tech stack

- **FastAPI** + **Uvicorn** — async web server & WebSockets
- **MediaPipe** — hand landmark detection
- **OpenCV** — image decoding & frame handling
- **scikit-learn** — Random Forest (static model)
- **TensorFlow / Keras** — LSTM (dynamic model)
- **Hugging Face Inference API (Mistral-7B)** — natural-language transcription

---

## ⚠️ Security note

`ai_transcription.py` currently contains a **hard-coded Hugging Face API key as a fallback default**. Before sharing this repo publicly or committing it anywhere, you should:

1. Remove the hard-coded key and rely only on the `HF_API_KEY` environment variable.
2. **Rotate/revoke** that token in your Hugging Face account, since it has been exposed in the source.

---

## License

No license specified yet. Add one (e.g. MIT) if you plan to share this publicly.
