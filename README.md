# CipherNet: ML-Powered Cryptography and Steganography Suite

CipherNet encrypts secret payloads with AES-GCM and embeds them into cover images using a U-Net-based steganography model. A CNN detector is trained for steganalysis, then used in adversarial training to improve undetectability.

## Implemented Components

- `core/encryption.py`
  - AES-256-GCM `encrypt()` and `decrypt()`
- `data/prepare_data.py`
  - cover preprocessing pipeline
  - `CoverSecretDataset` returning `(cover, secret)` tensors
- `core/steganography/model.py`
  - dual-decoder U-Net (`stego`, `recovered_secret`, `residual`)
- `core/steganography/trainer.py`
  - reconstruction-only training
  - adversarial training with detector
- `core/detector/cnn_detector.py`
  - CNN steganalysis detector + training loop
- `core/steganography/utils.py`
  - SSIM and bit-recovery metrics
- `core/detector/quantize.py`
  - ONNX export, INT8 quantization, and FP32 vs INT8 benchmark helpers
- `api/main.py`
  - `POST /embed`, `POST /extract`, `GET /health`
- `tests/`
  - encryption, metrics, and API integration tests
- `Dockerfile` and `docker-compose.yml`

## Project Structure

```text
.
|-- api/
|-- core/
|   |-- detector/
|   `-- steganography/
|-- data/
|-- models/
|-- tests/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Setup

1. Create environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. (Optional) For ONNX export/quantization with newer PyTorch exporters:

```bash
python -m pip install onnx onnxruntime onnxscript
```

## Data Preparation

Use local image folder:

```bash
python data/prepare_data.py --cover-source /path/to/images --cover-output data/cover --max-images 10000
```

Or download fallback dataset:

```bash
python data/prepare_data.py --download-default --cover-output data/cover --max-images 10000
```

## Training Workflow

1. Reconstruction pretraining (U-Net):

```bash
python core/steganography/trainer.py \
  --mode reconstruction \
  --cover-dir data/cover \
  --epochs 10 \
  --batch-size 8 \
  --model-output models/unet_reconstruction.pth
```

2. Detector training:

```bash
python core/detector/cnn_detector.py \
  --cover-dir data/cover \
  --unet-checkpoint models/unet_reconstruction.pth \
  --epochs 10 \
  --batch-size 8 \
  --detector-output models/detector_final.pth
```

3. Adversarial training (U-Net + detector):

```bash
python core/steganography/trainer.py \
  --mode adversarial \
  --cover-dir data/cover \
  --unet-init models/unet_reconstruction.pth \
  --detector-init models/detector_final.pth \
  --unet-output models/unet_final.pth \
  --detector-output models/detector_final.pth \
  --adv-weight 0.001 \
  --epochs 20 \
  --batch-size 8
```

## Quantization Workflow

Export + INT8 quantize + benchmark:

```bash
python core/detector/quantize.py \
  --detector-checkpoint models/detector_final.pth \
  --onnx-output models/detector.onnx \
  --int8-output models/detector_int8.onnx \
  --runs 200 \
  --warmup 20
```

The script reports latency and speedup percentage.

## API

Start server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### `POST /embed`

Form fields:
- `cover_image` (file)
- `secret_file` (file)
- `key_hex` (optional, 32-byte hex)

Returns:
- base64 stego PNG
- AES metadata (`nonce_hex`, `tag_hex`, `key_hex`)
- `ciphertext_length`

### `POST /extract`

Form fields:
- `stego_image` (file)
- `key_hex`, `nonce_hex`, `tag_hex`
- `ciphertext_length`

Returns decrypted secret as binary stream.

## Tests

Run all tests:

```bash
pytest -q
```

Current local status from development run:
- `14 passed`

## Docker

Build and run with compose:

```bash
docker compose up --build
```

API endpoint: `http://localhost:8000`

## Notes on Targets

- The codebase includes metric and training hooks needed to measure:
  - SSIM
  - bit recovery accuracy
  - detector accuracy before/after adversarial training
  - FP32 vs INT8 inference speedup
- Final target achievement (e.g., SSIM >= 0.96, bit recovery >= 99.5%, ~38% INT8 speedup, detector near 50%) depends on full training runs with suitable data and compute.
