# Running DrainWatch / Catchment Operations locally

This project has two parts that both need to be running at the same time:
a Python (FastAPI) backend that does the actual image classification, and a
React frontend ("Flood Watcher") that shows the dashboard.

## 1. Prerequisites

- Python 3.10+
- Node.js 18+
- `git`

## 2. Clone the repo

```
git clone https://github.com/gseira/drainage-blockage-detection.git
cd drainage-blockage-detection
```

## 3. Get the model checkpoint files

These are NOT in the git repo (they're large binary files, hundreds of MB
each, and git isn't built for that). You'll need them sent separately
(Google Drive / WeTransfer / USB, etc.) and placed here:

```
results/checkpoints/best_model.pt
```

That's the only checkpoint the app uses — a per-camera specialist fine-tuning
approach was tried and removed (see the history comment above
`_WEBCAM_LIST`/model loading in `app/main.py` if you're curious why); the
single general model now serves every camera.

## 4. Set up your own API key

The LLM explanation feature uses Groq. Copy the template and fill in your
OWN key (don't reuse someone else's — it's tied to their account/billing):

```
cp .env.example .env
```

Then edit `.env` and paste in a real Groq API key (free tier available at
https://console.groq.com). The app still works without one — it just shows
a placeholder explanation instead of a full write-up for BLOCKED results.

## 5. Install Python dependencies

```
pip install -r requirements.txt
```

(Consider doing this inside a virtual environment: `python3 -m venv venv && source venv/bin/activate` first.)

## 6. Install frontend dependencies

```
cd "Flood Watcher"
npm install
cd ..
```

## 7. Run both parts (two separate terminal windows)

**Terminal 1 — backend:**
```
uvicorn app.main:app --reload --port 8000
```
Wait for `Application startup complete.` Check it's working:
```
curl http://127.0.0.1:8000/health
```

**Terminal 2 — frontend:**
```
cd "Flood Watcher"
npm run dev
```
This will print a local URL (usually `http://localhost:3000` or similar) —
open that in a browser.

## Notes

- The database (`monitoring.db`) starts empty on first run — you won't see
  any historical scans until the backend has had a chance to run its own
  sweeps (every 15 minutes) or you trigger a manual scan from the UI.
- The backend needs real internet access to reach the Environment Agency's
  webcam sites (`eacornwallwebcams.org` / `eadevonwebcams.org`) — it won't
  work somewhere that blocks outbound requests to those domains.
- No GPU is required to just run and use the app — CPU inference works
  fine for periodic scans, just a bit slower per image than on a GPU.
