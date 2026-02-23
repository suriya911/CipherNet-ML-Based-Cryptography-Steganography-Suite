import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    from data.prepare_data import CoverSecretDataset
    from core.detector.cnn_detector import CNNDetector
    from core.steganography.model import StegoUNet
except ModuleNotFoundError:
    from ciphernet.data.prepare_data import CoverSecretDataset
    from ciphernet.core.detector.cnn_detector import CNNDetector
    from ciphernet.core.steganography.model import StegoUNet


def _run_epoch(
    model: StegoUNet,
    loader: DataLoader,
    mse_loss: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lambda_secret: float = 1.0,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_stego = 0.0
    total_secret = 0.0
    steps = 0

    for covers, secrets in loader:
        covers = covers.to(device)
        secrets = secrets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        stego, recovered_secret, _ = model(covers, secrets)
        mse_stego = mse_loss(stego, covers)
        mse_secret = mse_loss(recovered_secret, secrets)
        loss = mse_stego + (lambda_secret * mse_secret)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_stego += mse_stego.item()
        total_secret += mse_secret.item()
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    if steps == 0:
        raise ValueError("No steps were run. Ensure dataset and loader are non-empty.")

    return {
        "loss": total_loss / steps,
        "mse_stego": total_stego / steps,
        "mse_secret": total_secret / steps,
    }


def train_reconstruction(
    cover_dir: str,
    secret_dir: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    lambda_secret: float = 1.0,
    val_split: float = 0.1,
    model_output: str = "ciphernet/models/unet_reconstruction.pth",
    device: Optional[str] = None,
    max_steps_per_epoch: Optional[int] = None,
) -> Dict[str, float]:
    dataset = CoverSecretDataset(
        cover_dir=cover_dir,
        secret_dir=secret_dir,
        secret_mode="image" if secret_dir else "random",
    )

    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        train_len = len(dataset) - 1
        val_len = 1
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = StegoUNet().to(resolved_device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    output_path = Path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            mse_loss=mse_loss,
            device=resolved_device,
            optimizer=optimizer,
            lambda_secret=lambda_secret,
            max_steps=max_steps_per_epoch,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                mse_loss=mse_loss,
                device=resolved_device,
                optimizer=None,
                lambda_secret=lambda_secret,
                max_steps=max_steps_per_epoch,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_mse_stego={train_metrics['mse_stego']:.6f} "
            f"train_mse_secret={train_metrics['mse_secret']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = {
                "best_val_loss": best_val_loss,
                "best_val_mse_stego": val_metrics["mse_stego"],
                "best_val_mse_secret": val_metrics["mse_secret"],
            }
            torch.save(model.state_dict(), output_path)

    return best_metrics


def _binary_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred >= 0.5).float() == target).float().mean().item()


def _run_adversarial_epoch(
    unet: StegoUNet,
    detector: CNNDetector,
    loader: DataLoader,
    mse_loss: nn.Module,
    bce_loss: nn.Module,
    device: torch.device,
    unet_optimizer: Optional[torch.optim.Optimizer] = None,
    detector_optimizer: Optional[torch.optim.Optimizer] = None,
    lambda_secret: float = 1.0,
    adv_weight: float = 0.001,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    is_train = unet_optimizer is not None and detector_optimizer is not None
    unet.train(is_train)
    detector.train(is_train)

    total_unet = 0.0
    total_detector = 0.0
    total_mse_stego = 0.0
    total_mse_secret = 0.0
    total_adv = 0.0
    total_detector_acc = 0.0
    steps = 0

    for covers, secrets in loader:
        covers = covers.to(device)
        secrets = secrets.to(device)
        ones = torch.ones((covers.size(0), 1), device=device)
        zeros = torch.zeros((covers.size(0), 1), device=device)

        # 1) Detector step: classify cover=1 and stego=0.
        with torch.no_grad():
            stego_detached, _, _ = unet(covers, secrets)
        detector_inputs = torch.cat([covers, stego_detached], dim=0)
        detector_targets = torch.cat([ones, zeros], dim=0)

        detector_preds = detector(detector_inputs)
        detector_loss = bce_loss(detector_preds, detector_targets)
        detector_acc = _binary_accuracy(detector_preds.detach(), detector_targets)

        if is_train:
            detector_optimizer.zero_grad(set_to_none=True)
            detector_loss.backward()
            detector_optimizer.step()

        # 2) U-Net step: preserve quality + fool detector.
        for p in detector.parameters():
            p.requires_grad_(False)

        stego, recovered_secret, _ = unet(covers, secrets)
        stego_pred = detector(stego)

        mse_stego = mse_loss(stego, covers)
        mse_secret = mse_loss(recovered_secret, secrets)
        adv_loss = bce_loss(stego_pred, ones)
        unet_loss = mse_stego + (lambda_secret * mse_secret) + (adv_weight * adv_loss)

        if is_train:
            unet_optimizer.zero_grad(set_to_none=True)
            unet_loss.backward()
            unet_optimizer.step()

        for p in detector.parameters():
            p.requires_grad_(True)

        total_unet += unet_loss.item()
        total_detector += detector_loss.item()
        total_mse_stego += mse_stego.item()
        total_mse_secret += mse_secret.item()
        total_adv += adv_loss.item()
        total_detector_acc += detector_acc
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    if steps == 0:
        raise ValueError("No adversarial steps were run. Ensure dataset and loader are non-empty.")

    return {
        "unet_loss": total_unet / steps,
        "detector_loss": total_detector / steps,
        "mse_stego": total_mse_stego / steps,
        "mse_secret": total_mse_secret / steps,
        "adv_loss": total_adv / steps,
        "detector_acc": total_detector_acc / steps,
    }


def train_adversarial(
    cover_dir: str,
    secret_dir: Optional[str] = None,
    epochs: int = 20,
    batch_size: int = 8,
    unet_lr: float = 1e-4,
    detector_lr: float = 1e-4,
    lambda_secret: float = 1.0,
    adv_weight: float = 0.001,
    val_split: float = 0.1,
    unet_init: str = "ciphernet/models/unet_reconstruction.pth",
    detector_init: str = "ciphernet/models/detector_final.pth",
    unet_output: str = "ciphernet/models/unet_final.pth",
    detector_output: str = "ciphernet/models/detector_final.pth",
    device: Optional[str] = None,
    max_steps_per_epoch: Optional[int] = None,
) -> Dict[str, float]:
    dataset = CoverSecretDataset(
        cover_dir=cover_dir,
        secret_dir=secret_dir,
        secret_mode="image" if secret_dir else "random",
    )

    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        train_len = len(dataset) - 1
        val_len = 1
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    unet = StegoUNet().to(resolved_device)
    detector = CNNDetector().to(resolved_device)

    unet_init_path = Path(unet_init)
    detector_init_path = Path(detector_init)
    if unet_init_path.exists():
        unet.load_state_dict(torch.load(unet_init_path, map_location=resolved_device))
    if detector_init_path.exists():
        detector.load_state_dict(torch.load(detector_init_path, map_location=resolved_device))

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=unet_lr)
    detector_optimizer = torch.optim.Adam(detector.parameters(), lr=detector_lr)

    best_score = float("-inf")
    best_metrics: Dict[str, float] = {}
    unet_output_path = Path(unet_output)
    detector_output_path = Path(detector_output)
    unet_output_path.parent.mkdir(parents=True, exist_ok=True)
    detector_output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_adversarial_epoch(
            unet=unet,
            detector=detector,
            loader=train_loader,
            mse_loss=mse_loss,
            bce_loss=bce_loss,
            device=resolved_device,
            unet_optimizer=unet_optimizer,
            detector_optimizer=detector_optimizer,
            lambda_secret=lambda_secret,
            adv_weight=adv_weight,
            max_steps=max_steps_per_epoch,
        )
        with torch.no_grad():
            val_metrics = _run_adversarial_epoch(
                unet=unet,
                detector=detector,
                loader=val_loader,
                mse_loss=mse_loss,
                bce_loss=bce_loss,
                device=resolved_device,
                unet_optimizer=None,
                detector_optimizer=None,
                lambda_secret=lambda_secret,
                adv_weight=adv_weight,
                max_steps=max_steps_per_epoch,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_unet={train_metrics['unet_loss']:.6f} "
            f"train_detector={train_metrics['detector_loss']:.6f} "
            f"train_det_acc={train_metrics['detector_acc']:.4f} | "
            f"val_unet={val_metrics['unet_loss']:.6f} "
            f"val_detector={val_metrics['detector_loss']:.6f} "
            f"val_det_acc={val_metrics['detector_acc']:.4f}"
        )

        # Prefer low detector accuracy near 0.5 while maintaining low reconstruction loss.
        score = -abs(val_metrics["detector_acc"] - 0.5) - val_metrics["mse_stego"] - val_metrics["mse_secret"]
        if score > best_score:
            best_score = score
            best_metrics = {
                "best_score": best_score,
                "best_val_detector_acc": val_metrics["detector_acc"],
                "best_val_mse_stego": val_metrics["mse_stego"],
                "best_val_mse_secret": val_metrics["mse_secret"],
            }
            torch.save(unet.state_dict(), unet_output_path)
            torch.save(detector.state_dict(), detector_output_path)

    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stego U-Net (reconstruction or adversarial mode).")
    parser.add_argument("--mode", type=str, choices=["reconstruction", "adversarial"], default="reconstruction")
    parser.add_argument("--cover-dir", type=str, default="ciphernet/data/cover")
    parser.add_argument("--secret-dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unet-lr", type=float, default=1e-4)
    parser.add_argument("--detector-lr", type=float, default=1e-4)
    parser.add_argument("--lambda-secret", type=float, default=1.0)
    parser.add_argument("--adv-weight", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--model-output", type=str, default="ciphernet/models/unet_reconstruction.pth")
    parser.add_argument("--unet-init", type=str, default="ciphernet/models/unet_reconstruction.pth")
    parser.add_argument("--detector-init", type=str, default="ciphernet/models/detector_final.pth")
    parser.add_argument("--unet-output", type=str, default="ciphernet/models/unet_final.pth")
    parser.add_argument("--detector-output", type=str, default="ciphernet/models/detector_final.pth")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "reconstruction":
        metrics = train_reconstruction(
            cover_dir=args.cover_dir,
            secret_dir=args.secret_dir or None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_secret=args.lambda_secret,
            val_split=args.val_split,
            model_output=args.model_output,
            device=args.device or None,
            max_steps_per_epoch=args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else None,
        )
        print(f"Reconstruction training complete. Best metrics: {metrics}")
        return

    metrics = train_adversarial(
        cover_dir=args.cover_dir,
        secret_dir=args.secret_dir or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        unet_lr=args.unet_lr,
        detector_lr=args.detector_lr,
        lambda_secret=args.lambda_secret,
        adv_weight=args.adv_weight,
        val_split=args.val_split,
        unet_init=args.unet_init,
        detector_init=args.detector_init,
        unet_output=args.unet_output,
        detector_output=args.detector_output,
        device=args.device or None,
        max_steps_per_epoch=args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else None,
    )
    print(f"Adversarial training complete. Best metrics: {metrics}")


if __name__ == "__main__":
    main()
