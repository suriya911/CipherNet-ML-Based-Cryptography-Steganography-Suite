from typing import Iterable, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_metric


def _to_numpy_image_batch(tensor: torch.Tensor) -> np.ndarray:
    """Convert BCHW float tensor in [0,1] to BHWC float numpy."""
    if tensor.ndim != 4:
        raise ValueError("Expected BCHW tensor.")
    clamped = tensor.detach().cpu().float().clamp(0.0, 1.0)
    return clamped.permute(0, 2, 3, 1).numpy()


def compute_batch_ssim(cover: torch.Tensor, stego: torch.Tensor) -> float:
    """Mean SSIM over batch for RGB images in [0,1]."""
    cover_np = _to_numpy_image_batch(cover)
    stego_np = _to_numpy_image_batch(stego)
    if cover_np.shape != stego_np.shape:
        raise ValueError("Cover and stego shapes must match.")

    values = []
    for c, s in zip(cover_np, stego_np):
        values.append(
            ssim_metric(
                c,
                s,
                data_range=1.0,
                channel_axis=2,
            )
        )
    return float(np.mean(values))


def tensor_to_uint8_bytes(tensor: torch.Tensor) -> bytes:
    """Encode image tensor CHW in [0,1] as raw uint8 bytes."""
    if tensor.ndim != 3:
        raise ValueError("Expected CHW tensor.")
    arr = (tensor.detach().cpu().float().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    return arr.tobytes()


def bit_recovery_accuracy(expected_bytes: bytes, recovered_bytes: bytes) -> float:
    """Per-bit equality rate between two byte strings."""
    n = min(len(expected_bytes), len(recovered_bytes))
    if n == 0:
        raise ValueError("Inputs must not be empty.")

    a = np.frombuffer(expected_bytes[:n], dtype=np.uint8)
    b = np.frombuffer(recovered_bytes[:n], dtype=np.uint8)
    xor = np.bitwise_xor(a, b)
    bit_errors = int(np.unpackbits(xor).sum())
    total_bits = n * 8
    return float((total_bits - bit_errors) / total_bits)


def compute_secret_bit_recovery(secret: torch.Tensor, recovered_secret: torch.Tensor) -> float:
    """Bit recovery for batched secret tensors in BCHW [0,1]."""
    if secret.ndim != 4 or recovered_secret.ndim != 4:
        raise ValueError("Expected BCHW tensors.")
    if secret.shape != recovered_secret.shape:
        raise ValueError("Secret and recovered_secret shapes must match.")

    accuracies = []
    for s, r in zip(secret, recovered_secret):
        s_bytes = tensor_to_uint8_bytes(s)
        r_bytes = tensor_to_uint8_bytes(r)
        accuracies.append(bit_recovery_accuracy(s_bytes, r_bytes))
    return float(np.mean(accuracies))


def detector_accuracy(
    detector: torch.nn.Module,
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float = 0.5,
) -> float:
    """Compute binary accuracy from iterable of (inputs, labels)."""
    detector.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in batches:
            probs = detector(x)
            preds = (probs >= threshold).float()
            total += y.numel()
            correct += int((preds == y).sum().item())
    if total == 0:
        raise ValueError("No samples provided.")
    return correct / total
