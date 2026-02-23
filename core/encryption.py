import os
from typing import Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


KEY_SIZE_BYTES = 32
NONCE_SIZE_BYTES = 12
TAG_SIZE_BYTES = 16


def encrypt(secret_bytes: bytes, key: Optional[bytes] = None) -> Dict[str, bytes]:
    """Encrypt bytes using AES-GCM.

    Returns ciphertext and tag separately for API convenience.
    """
    if key is None:
        key = os.urandom(KEY_SIZE_BYTES)
    if len(key) != KEY_SIZE_BYTES:
        raise ValueError("AES-256 key must be 32 bytes long.")

    nonce = os.urandom(NONCE_SIZE_BYTES)
    aesgcm = AESGCM(key)
    encrypted = aesgcm.encrypt(nonce, secret_bytes, associated_data=None)
    ciphertext = encrypted[:-TAG_SIZE_BYTES]
    tag = encrypted[-TAG_SIZE_BYTES:]

    return {
        "ciphertext": ciphertext,
        "nonce": nonce,
        "tag": tag,
        "key": key,
    }


def decrypt(ciphertext: bytes, nonce: bytes, key: bytes, tag: Optional[bytes] = None) -> bytes:
    """Decrypt AES-GCM payload.

    If tag is provided, it will be appended to ciphertext before decryption.
    """
    if len(key) != KEY_SIZE_BYTES:
        raise ValueError("AES-256 key must be 32 bytes long.")
    if len(nonce) != NONCE_SIZE_BYTES:
        raise ValueError("AES-GCM nonce must be 12 bytes long.")

    payload = ciphertext + tag if tag is not None else ciphertext
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, payload, associated_data=None)
