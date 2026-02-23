import pytest
from cryptography.exceptions import InvalidTag

from core.encryption import NONCE_SIZE_BYTES, TAG_SIZE_BYTES, decrypt, encrypt


def test_encrypt_decrypt_roundtrip_with_generated_key() -> None:
    secret = b"ciphernet-secret-payload"
    result = encrypt(secret)

    assert set(result.keys()) == {"ciphertext", "nonce", "tag", "key"}
    assert len(result["nonce"]) == NONCE_SIZE_BYTES
    assert len(result["tag"]) == TAG_SIZE_BYTES
    assert len(result["key"]) == 32

    recovered = decrypt(
        ciphertext=result["ciphertext"],
        nonce=result["nonce"],
        key=result["key"],
        tag=result["tag"],
    )
    assert recovered == secret


def test_encrypt_decrypt_roundtrip_with_provided_key() -> None:
    secret = b"hello-world"
    key = bytes(range(32))
    result = encrypt(secret, key=key)

    assert result["key"] == key
    recovered = decrypt(result["ciphertext"], result["nonce"], result["key"], result["tag"])
    assert recovered == secret


def test_decrypt_fails_with_tampered_ciphertext() -> None:
    secret = b"top-secret"
    result = encrypt(secret)

    tampered = bytearray(result["ciphertext"])
    tampered[0] ^= 0xFF

    with pytest.raises(InvalidTag):
        decrypt(bytes(tampered), result["nonce"], result["key"], result["tag"])


def test_encrypt_rejects_invalid_key_length() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        encrypt(b"x", key=b"short-key")


def test_decrypt_rejects_invalid_nonce_length() -> None:
    result = encrypt(b"abc")
    with pytest.raises(ValueError, match="12 bytes"):
        decrypt(result["ciphertext"], b"short", result["key"], result["tag"])
