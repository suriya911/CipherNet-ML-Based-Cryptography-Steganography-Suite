import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from cryptography.exceptions import InvalidTag
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

try:
    from api.schemas import DecodeResponse, DetectResponse, EmbedResponse
    from core.encryption import decrypt, encrypt
    from core.detector.cnn_detector import CNNDetector
    from core.steganography.model import StegoUNet
except ModuleNotFoundError:
    from ciphernet.api.schemas import DecodeResponse, DetectResponse, EmbedResponse
    from ciphernet.core.encryption import decrypt, encrypt
    from ciphernet.core.detector.cnn_detector import CNNDetector
    from ciphernet.core.steganography.model import StegoUNet


SECRET_SHAPE = (3, 64, 64)
SECRET_CAPACITY_BYTES = SECRET_SHAPE[0] * SECRET_SHAPE[1] * SECRET_SHAPE[2]

app = FastAPI(title="CipherNet API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_model_path() -> Path:
    candidate_paths = [
        Path("ciphernet/models/unet_final.pth"),
        Path("models/unet_final.pth"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return candidate_paths[0]


def _load_unet() -> StegoUNet:
    model = StegoUNet()
    model_path = _resolve_model_path()
    if model_path.exists():
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        print(f"Warning: U-Net weights not found at {model_path}; using random-initialized model.")
    model.eval()
    return model


MODEL = _load_unet()


def _resolve_detector_path() -> Path:
    candidate_paths = [
        Path("ciphernet/models/detector_final.pth"),
        Path("models/detector_final.pth"),
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return candidate_paths[0]


def _load_detector() -> CNNDetector:
    detector = CNNDetector()
    detector_path = _resolve_detector_path()
    if detector_path.exists():
        state = torch.load(detector_path, map_location="cpu")
        detector.load_state_dict(state)
    else:
        print(f"Warning: detector weights not found at {detector_path}; using random-initialized detector.")
    detector.eval()
    return detector


DETECTOR_MODEL = _load_detector()


def _parse_hex_bytes(value: str, expected_len: Optional[int] = None, field: str = "value") -> bytes:
    try:
        parsed = bytes.fromhex(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field} must be valid hex.") from exc
    if expected_len is not None and len(parsed) != expected_len:
        raise HTTPException(status_code=400, detail=f"{field} must be {expected_len} bytes.")
    return parsed


def _image_bytes_to_tensor(image_bytes: bytes, size: int = 256) -> torch.Tensor:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc
    return torch.from_numpy(arr).permute(2, 0, 1)


def _tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    arr = (
        image_tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
    )
    arr_u8 = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_u8)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _ciphertext_to_secret_tensor(ciphertext: bytes) -> torch.Tensor:
    if len(ciphertext) > SECRET_CAPACITY_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Encrypted secret too large: {len(ciphertext)} bytes exceeds "
                f"capacity {SECRET_CAPACITY_BYTES} bytes."
            ),
        )
    buf = np.zeros((SECRET_CAPACITY_BYTES,), dtype=np.uint8)
    if ciphertext:
        buf[: len(ciphertext)] = np.frombuffer(ciphertext, dtype=np.uint8)
    tensor = torch.from_numpy(buf.reshape(SECRET_SHAPE).astype(np.float32) / 255.0)
    return tensor


def _secret_tensor_to_ciphertext(secret_tensor: torch.Tensor, length: int) -> bytes:
    flat = (
        secret_tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .reshape(-1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    if length > flat.shape[0]:
        raise HTTPException(status_code=400, detail="Requested ciphertext length exceeds secret capacity.")
    return flat[:length].tobytes()


def _decode_secret_bytes(
    stego_bytes: bytes,
    key_hex: str,
    nonce_hex: str,
    tag_hex: str,
    ciphertext_length: int,
) -> bytes:
    if ciphertext_length < 0:
        raise HTTPException(status_code=400, detail="ciphertext_length must be non-negative.")

    key = _parse_hex_bytes(key_hex, expected_len=32, field="key_hex")
    nonce = _parse_hex_bytes(nonce_hex, expected_len=12, field="nonce_hex")
    tag = _parse_hex_bytes(tag_hex, expected_len=16, field="tag_hex")

    stego_tensor = _image_bytes_to_tensor(stego_bytes).unsqueeze(0)
    placeholder_secret = torch.zeros((1, *SECRET_SHAPE), dtype=torch.float32)

    with torch.no_grad():
        _, recovered_secret, _ = MODEL(stego_tensor, placeholder_secret)

    ciphertext = _secret_tensor_to_ciphertext(recovered_secret[0], ciphertext_length)
    try:
        return decrypt(ciphertext=ciphertext, nonce=nonce, key=key, tag=tag)
    except InvalidTag as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Decryption failed (InvalidTag). The stego image may be corrupted, "
                "metadata may be wrong, or the model is not sufficiently trained."
            ),
        ) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
async def embed(
    cover_image: UploadFile = File(...),
    secret_file: UploadFile = File(...),
    key_hex: Optional[str] = Form(default=None),
) -> EmbedResponse:
    cover_bytes = await cover_image.read()
    secret_bytes = await secret_file.read()
    if not secret_bytes:
        raise HTTPException(status_code=400, detail="secret_file is empty.")

    key = _parse_hex_bytes(key_hex, expected_len=32, field="key_hex") if key_hex else None
    encrypted = encrypt(secret_bytes, key=key)

    cover_tensor = _image_bytes_to_tensor(cover_bytes).unsqueeze(0)
    secret_tensor = _ciphertext_to_secret_tensor(encrypted["ciphertext"]).unsqueeze(0)

    with torch.no_grad():
        stego_tensor, _, _ = MODEL(cover_tensor, secret_tensor)

    stego_png = _tensor_to_png_bytes(stego_tensor[0])
    return EmbedResponse(
        stego_image_base64=base64.b64encode(stego_png).decode("utf-8"),
        nonce_hex=encrypted["nonce"].hex(),
        tag_hex=encrypted["tag"].hex(),
        key_hex=encrypted["key"].hex(),
        ciphertext_length=len(encrypted["ciphertext"]),
        secret_capacity_bytes=SECRET_CAPACITY_BYTES,
        original_secret_length=len(secret_bytes),
    )


@app.post("/extract")
async def extract(
    stego_image: UploadFile = File(...),
    key_hex: str = Form(...),
    nonce_hex: str = Form(...),
    tag_hex: str = Form(...),
    ciphertext_length: int = Form(...),
):
    stego_bytes = await stego_image.read()
    plain_secret = _decode_secret_bytes(
        stego_bytes=stego_bytes,
        key_hex=key_hex,
        nonce_hex=nonce_hex,
        tag_hex=tag_hex,
        ciphertext_length=ciphertext_length,
    )

    headers = {"Content-Disposition": f'attachment; filename="secret.bin"'}
    return StreamingResponse(io.BytesIO(plain_secret), media_type="application/octet-stream", headers=headers)


@app.post("/decode", response_model=DecodeResponse)
async def decode(
    stego_image: UploadFile = File(...),
    key_hex: str = Form(...),
    nonce_hex: str = Form(...),
    tag_hex: str = Form(...),
    ciphertext_length: int = Form(...),
) -> DecodeResponse:
    stego_bytes = await stego_image.read()
    plain_secret = _decode_secret_bytes(
        stego_bytes=stego_bytes,
        key_hex=key_hex,
        nonce_hex=nonce_hex,
        tag_hex=tag_hex,
        ciphertext_length=ciphertext_length,
    )
    return DecodeResponse(
        secret_base64=base64.b64encode(plain_secret).decode("utf-8"),
        secret_length=len(plain_secret),
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...)) -> DetectResponse:
    image_bytes = await image.read()
    image_tensor = _image_bytes_to_tensor(image_bytes).unsqueeze(0)
    with torch.no_grad():
        cover_probability = float(DETECTOR_MODEL(image_tensor).squeeze().item())
    cover_probability = float(np.clip(cover_probability, 0.0, 1.0))
    stego_probability = 1.0 - cover_probability
    predicted_label = "cover" if cover_probability >= 0.5 else "stego"
    return DetectResponse(
        cover_probability=cover_probability,
        stego_probability=stego_probability,
        predicted_label=predicted_label,
    )
