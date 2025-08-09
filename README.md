# Environment Setup

## Create & activate the Conda env (CUDA runtime included)
`conda create -n yolo -y python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
`conda activate yolo`

## recommended to keep packages conda-managed
`conda install -y -c conda-forge opencv`

## Verify GPU PyTorch
`python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

## Download pip
`python -m pip install -U pip`

## Download ultralytics
`python -m pip install ultralytics --no-deps`

## Verify Ultralytics + download yolov8n once
`python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('ultralytics OK')"`

## If python -c "import cv2; print('cv2', cv2.__version__, '->', cv2.__file__)" fails, try:
### remove conda's opencv so we have only one source
`conda remove -y opencv`

### nuke any stray cv2 files (ignore errors if paths not found)
`Remove-Item -Recurse -Force "$env:CONDA_PREFIX\Lib\site-packages\cv2" -ErrorAction SilentlyContinue`
`Remove-Item -Recurse -Force "$env:CONDA_PREFIX\Lib\site-packages\opencv_python-*.dist-info" -ErrorAction SilentlyContinue`

### install the pip wheel (bundled DLLs)
`python -m pip install --no-cache-dir opencv-python==4.10.0.84`
#### (or the latest: opencv-python==4.12.0.88)

`conda install -y -c conda-forge pandas scipy pyyaml psutil matplotlib tqdm pillow`
`python -m pip install ultralytics-thop --no-deps`

#### Sanity Checks
`python -c "import torch,cv2,pandas,scipy,psutil,yaml,matplotlib,tqdm;print('OK')"`
`python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('ultralytics OK')"`

# Running YOLO

## Run the YOLO with one of these source to find the right camera source:
`yolo predict model=yolov8n.pt source=0 show=True conf=0.25`
`yolo predict model=yolov8n.pt source=1 show=True conf=0.25`
`yolo predict model=yolov8n.pt source=2 show=True conf=0.25`

# Project Setup

## install FastAPI
python -m pip install fastapi uvicorn[standard]

## run the py file
uvicorn main:app --host 0.0.0.0 --port 8000 --reload