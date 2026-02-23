# CipherNet: ML-Powered Cryptography and Steganography Suite

CipherNet encrypts a secret payload with AES-GCM and hides it in a cover image using a U-Net steganography model. A CNN detector is trained and then used adversarially to improve undetectability.

## Current Project Status

- Core modules implemented: encryption, data pipeline, U-Net, detector, training loops, quantization helper, FastAPI API.
- Test status: `14 passed` on current local run.
- Known technical gap: API `/embed -> /extract` decryption currently fails in realistic runs because secret recovery quality is not yet sufficient for cryptographic exactness.
- Known performance gap: current INT8 path is functional but slower than FP32 on this CPU setup.

## Repository Layout

```text
.
|-- api/
|-- core/
|   |-- detector/
|   `-- steganography/
|-- data/
|-- models/
|-- frontend/
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Data Used and Where It Loads From

### Cover Images

`data/prepare_data.py` supports two modes:

1. Local source folder:

```bash
python data/prepare_data.py --cover-source /path/to/images --cover-output data/cover --max-images 10000
```

2. Built-in downloadable fallback (currently used in local runs):

```bash
python data/prepare_data.py --download-default --cover-output data/cover --max-images 10000
```

This fallback uses **CIFAR-10** and upscales to `256x256`.

### Dataset Links

- CIFAR-10 overview: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10 Python archive: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

### Storage Paths (Local)

- Processed cover images: `data/cover/*.png`
- Download cache (when using fallback downloader): `data/cover/_cifar_cache/`
- Optional secret image pool: `data/secret/`
- Model outputs/checkpoints: `models/`

## Large File Policy

### Files over 100MB (current workspace check)

- Current count: **0 files over 100MB**.

### Where to Store Large Artifacts

Do **not** commit large datasets/checkpoints directly to normal Git history.

Recommended locations:

1. GitHub Releases (attach model artifacts per version)
2. External object storage (S3 / GCS / Azure Blob)
3. Git LFS if you need large files versioned in Git

Suggested naming for published artifacts:

- `unet_final_<date>.pth`
- `detector_final_<date>.pth`
- `detector_int8_<date>.onnx`

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional ONNX extras:

```bash
python -m pip install onnx onnxruntime onnxscript
```

## Training Workflow

1. Reconstruction pretraining:

```bash
python -m core.steganography.trainer --mode reconstruction --cover-dir data/cover --epochs 10 --batch-size 8 --model-output models/unet_reconstruction.pth
```

2. Detector training:

```bash
python -m core.detector.cnn_detector --cover-dir data/cover --unet-checkpoint models/unet_reconstruction.pth --epochs 10 --batch-size 8 --detector-output models/detector_final.pth
```

3. Adversarial training:

```bash
python -m core.steganography.trainer --mode adversarial --cover-dir data/cover --unet-init models/unet_reconstruction.pth --detector-init models/detector_final.pth --unet-output models/unet_final.pth --detector-output models/detector_final.pth --adv-weight 0.001 --epochs 20 --batch-size 8
```

## Quantization Workflow

```bash
python -m core.detector.quantize --detector-checkpoint models/detector_final.pth --onnx-output models/detector.onnx --int8-output models/detector_int8.onnx --runs 200 --warmup 20 --batch-size 1 --opset 17
```

## API

Start:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /embed`
- `POST /extract`
- `POST /decode`
- `POST /detect`

## TypeScript API (Proxy Layer)

A TypeScript/Express API is provided in `api_ts/` with matching endpoint surface:

- `GET /health`
- `POST /embed`
- `POST /extract`
- `POST /decode`
- `POST /detect`

This service forwards requests to the Python model API (`PYTHON_API_BASE_URL`, default `http://127.0.0.1:8000`).

Run:

```bash
cd api_ts
npm install
npm run dev
```

## Frontend Preview

A full React + TypeScript UI is included in `frontend/` with pages for:

- Overview
- Embed
- Decode
- Detect

Run locally:

```bash
cd frontend
npm install
npm run dev
```

Then visit: `http://localhost:5173`

Optional:

- set `VITE_API_BASE_URL` to use the TypeScript API (`http://localhost:9000`) or Python API (`http://localhost:8000`).

## Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest -q
```

## Known Issues To Fix Next

1. Secret recovery is not yet decryption-stable for realistic payloads.
2. INT8 benchmark is slower than FP32 on current CPU path and needs quantization strategy refinement.
3. End-to-end target metrics are not all met yet; SSIM is strong, but bit recovery remains far below target.
