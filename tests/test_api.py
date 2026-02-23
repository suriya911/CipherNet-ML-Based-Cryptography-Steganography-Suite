from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

from api.main import MODEL, app
from core.encryption import encrypt


def _png_bytes(size: int = 256) -> bytes:
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


class DummyIdentityModel(torch.nn.Module):
    def forward(self, cover: torch.Tensor, secret: torch.Tensor):
        residual = torch.zeros_like(cover)
        return cover, secret, residual


class DummyRecoverModel(torch.nn.Module):
    def __init__(self, recovered_secret: torch.Tensor):
        super().__init__()
        self.recovered_secret = recovered_secret

    def forward(self, cover: torch.Tensor, secret: torch.Tensor):
        residual = torch.zeros_like(cover)
        stego = cover
        batch = cover.shape[0]
        recovered = self.recovered_secret.repeat(batch, 1, 1, 1)
        return stego, recovered, residual


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_embed_returns_expected_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.main.MODEL", DummyIdentityModel())

    client = TestClient(app)
    resp = client.post(
        "/embed",
        files={
            "cover_image": ("cover.png", _png_bytes(), "image/png"),
            "secret_file": ("secret.bin", b"ciphernet-api-secret", "application/octet-stream"),
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["nonce_hex"]) == 24
    assert len(payload["tag_hex"]) == 32
    assert len(payload["key_hex"]) == 64
    assert payload["ciphertext_length"] > 0
    assert payload["secret_capacity_bytes"] == 3 * 64 * 64
    assert payload["original_secret_length"] == len(b"ciphernet-api-secret")
    assert isinstance(payload["stego_image_base64"], str) and len(payload["stego_image_base64"]) > 10


def test_extract_roundtrip_with_mocked_recovered_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    plain = b"recover-me"
    enc = encrypt(plain)

    from api.main import _ciphertext_to_secret_tensor

    recovered_secret = _ciphertext_to_secret_tensor(enc["ciphertext"]).unsqueeze(0)
    monkeypatch.setattr("api.main.MODEL", DummyRecoverModel(recovered_secret))

    client = TestClient(app)
    resp = client.post(
        "/extract",
        files={"stego_image": ("stego.png", _png_bytes(), "image/png")},
        data={
            "key_hex": enc["key"].hex(),
            "nonce_hex": enc["nonce"].hex(),
            "tag_hex": enc["tag"].hex(),
            "ciphertext_length": str(len(enc["ciphertext"])),
        },
    )
    assert resp.status_code == 200
    assert resp.content == plain


def test_extract_invalid_key_rejected() -> None:
    client = TestClient(app)
    resp = client.post(
        "/extract",
        files={"stego_image": ("stego.png", _png_bytes(), "image/png")},
        data={
            "key_hex": "abcd",
            "nonce_hex": "00" * 12,
            "tag_hex": "00" * 16,
            "ciphertext_length": "1",
        },
    )
    assert resp.status_code == 400
    assert "key_hex" in resp.text
