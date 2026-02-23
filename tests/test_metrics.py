import torch

from core.steganography.utils import (
    bit_recovery_accuracy,
    compute_batch_ssim,
    compute_secret_bit_recovery,
)


def test_ssim_identical_images_is_one() -> None:
    x = torch.rand(2, 3, 64, 64)
    value = compute_batch_ssim(x, x.clone())
    assert value == 1.0


def test_ssim_perturbed_images_is_lower() -> None:
    cover = torch.ones(1, 3, 64, 64) * 0.5
    stego = (cover + 0.2).clamp(0.0, 1.0)
    value = compute_batch_ssim(cover, stego)
    assert 0.0 < value < 1.0


def test_bit_recovery_accuracy_exact_match() -> None:
    data = b"\x00\x0f\xaa\xff"
    assert bit_recovery_accuracy(data, data) == 1.0


def test_bit_recovery_accuracy_known_flip() -> None:
    a = b"\x00"
    b = b"\x01"  # one bit differs
    assert bit_recovery_accuracy(a, b) == 0.875


def test_secret_bit_recovery_from_tensor() -> None:
    secret = torch.zeros(1, 3, 4, 4)
    recovered = secret.clone()
    recovered[0, 0, 0, 0] = 1.0
    # one full byte differs after quantization: 8 bits over 3*4*4 bytes
    expected = 1.0 - (8 / (3 * 4 * 4 * 8))
    assert compute_secret_bit_recovery(secret, recovered) == expected
