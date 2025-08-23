# ZoneSight — Real-time People & Occupancy Analytics

FastAPI + YOLOv8 backend for its [React dashboard](https://github.com/Leo10250/ZoneSight-UI). Streams annotated MJPEG video and live metrics over WebSockets. Allows polygon zones, human occupancy, dwell time, and recent events.

## Stack
- Backend: FastAPI, Ultralytics YOLOv8, WebSockets, PyTorch (CUDA), OpenCV
- Dev: Conda env, Windows/RTX GPU

## Setup (Git Bash / Linux / macOS)
```bash
conda env create -f environment.yml
conda activate yolo

# copy config and edit if needed (e.g., SOURCE, ALLOW_ORIGINS)
cp .env.example .env

# run API (LAN-accessible)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration (in `.env`)

- `SOURCE` — camera index like `0` (DSHOW) or RTSP/HTTP URL  
- `IMG_SIZE`, `CONF`, `STREAM_FPS`, `JPEG_QUALITY`, `HALF`  
- `ALLOW_ORIGINS` — UI origin(s), e.g. `http://localhost:5173`  
- `TRACK_MISS_TTL`, `EVENTS_MAX`  
- `OCCUPANCY_CLASSES` — CSV of classes to count (default `person`)  
- `YOLO_WEIGHTS` — e.g. `yolov8n.pt`

Zones persist to `zones.json`.

---

## Endpoints

- `GET /video` — MJPEG stream of annotated frames  
- `WS /ws` — live metrics (~5 Hz): counts, detections (with `zone`, `dwell_s`), `occupancy`, `recent_events`  
- `GET /zones` / `PUT /zones` — read/write polygon zones  
- `GET /events?limit=50` — recent entry/exit/transfer events (in-memory)  
- `GET /health` — basic runtime info  

---

## Notes

- For phone/another device: set  
  `ALLOW_ORIGINS=http://<your-ip>:5173` and point your UI at `http://<your-ip>:8000`.  
- Human-only occupancy is on by default via `OCCUPANCY_CLASSES=person`.