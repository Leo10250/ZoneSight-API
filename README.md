# ZoneSight — Real-time People & Occupancy Analytics

FastAPI + YOLOv8 backend for its React dashboard. Streams annotated MJPEG video and live metrics over WebSockets; foundation for zone-based occupancy, dwell time, and event history.

## Stack
- Backend: FastAPI, Ultralytics YOLOv8, WebSockets, PyTorch (CUDA), OpenCV
- Dev: Conda env, Windows/RTX GPU

## Setup (PowerShell / Windows)
```powershell
conda env create -f environment.yml
conda activate yolo
# run dev server on LAN so phone can view:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```