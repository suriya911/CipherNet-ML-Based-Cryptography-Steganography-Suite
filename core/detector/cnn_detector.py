import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    from data.prepare_data import CoverSecretDataset
    from core.steganography.model import StegoUNet
except ModuleNotFoundError:
    from ciphernet.data.prepare_data import CoverSecretDataset
    from ciphernet.core.steganography.model import StegoUNet


class CNNDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _build_balanced_batch(
    cover_batch: torch.Tensor,
    secret_batch: torch.Tensor,
    stego_model: StegoUNet,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        stego_batch, _, _ = stego_model(cover_batch, secret_batch)

    x = torch.cat([cover_batch, stego_batch], dim=0)
    y_cover = torch.ones((cover_batch.shape[0], 1), device=cover_batch.device)
    y_stego = torch.zeros((stego_batch.shape[0], 1), device=stego_batch.device)
    y = torch.cat([y_cover, y_stego], dim=0)
    return x, y


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_label = (pred >= 0.5).float()
    return (pred_label == target).float().mean().item()


def _run_epoch(
    detector: CNNDetector,
    stego_model: StegoUNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    detector.train(is_train)
    stego_model.eval()

    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for covers, secrets in loader:
        covers = covers.to(device)
        secrets = secrets.to(device)
        inputs, targets = _build_balanced_batch(covers, secrets, stego_model)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        preds = detector(inputs)
        loss = criterion(preds, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += _accuracy(preds.detach(), targets)
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    if steps == 0:
        raise ValueError("No detector steps were run. Check dataset and dataloader.")

    return {"loss": total_loss / steps, "accuracy": total_acc / steps}


def train_detector(
    cover_dir: str,
    secret_dir: Optional[str] = None,
    unet_checkpoint: str = "ciphernet/models/unet_reconstruction.pth",
    detector_output: str = "ciphernet/models/detector_final.pth",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    val_split: float = 0.1,
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

    stego_model = StegoUNet().to(resolved_device)
    checkpoint_path = Path(unet_checkpoint)
    if checkpoint_path.exists():
        stego_model.load_state_dict(torch.load(checkpoint_path, map_location=resolved_device))
    else:
        print(f"Warning: U-Net checkpoint not found at {unet_checkpoint}; using untrained U-Net for stego generation.")
    stego_model.eval()
    for param in stego_model.parameters():
        param.requires_grad = False

    detector = CNNDetector().to(resolved_device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=lr)

    best_val_acc = 0.0
    best_metrics: Dict[str, float] = {}
    output_path = Path(detector_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            detector=detector,
            stego_model=stego_model,
            loader=train_loader,
            criterion=criterion,
            device=resolved_device,
            optimizer=optimizer,
            max_steps=max_steps_per_epoch,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                detector=detector,
                stego_model=stego_model,
                loader=val_loader,
                criterion=criterion,
                device=resolved_device,
                optimizer=None,
                max_steps=max_steps_per_epoch,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = {
                "best_val_accuracy": best_val_acc,
                "best_val_loss": val_metrics["loss"],
            }
            torch.save(detector.state_dict(), output_path)

    return best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN detector to classify cover vs stego images.")
    parser.add_argument("--cover-dir", type=str, default="ciphernet/data/cover")
    parser.add_argument("--secret-dir", type=str, default="")
    parser.add_argument("--unet-checkpoint", type=str, default="ciphernet/models/unet_reconstruction.pth")
    parser.add_argument("--detector-output", type=str, default="ciphernet/models/detector_final.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_detector(
        cover_dir=args.cover_dir,
        secret_dir=args.secret_dir or None,
        unet_checkpoint=args.unet_checkpoint,
        detector_output=args.detector_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device or None,
        max_steps_per_epoch=args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else None,
    )
    print(f"Detector training complete. Best metrics: {metrics}")


if __name__ == "__main__":
    main()
