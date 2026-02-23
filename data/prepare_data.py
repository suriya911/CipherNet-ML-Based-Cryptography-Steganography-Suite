import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


def preprocess_and_save_images(
    source_dir: Path,
    output_dir: Path,
    image_size: int = 256,
    max_images: Optional[int] = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list_images(source_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    resize = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
        ]
    )

    saved = 0
    for idx, image_path in enumerate(image_paths):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = resize(img)
                img.save(output_dir / f"cover_{idx:06d}.png")
                saved += 1
        except Exception:
            continue

    return saved


def download_cover_dataset(output_dir: Path, max_images: int = 10000) -> int:
    """Download a small open dataset as a practical default for cover images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(root=str(output_dir / "_cifar_cache"), train=True, download=True)
    count = min(max_images, len(dataset))

    resize = transforms.Resize((256, 256))
    for idx in range(count):
        img, _ = dataset[idx]
        img = resize(img.convert("RGB"))
        img.save(output_dir / f"cover_{idx:06d}.png")

    return count


class CoverSecretDataset(Dataset):
    """Returns (cover, secret) tensors normalized to [0, 1]."""

    def __init__(
        self,
        cover_dir: str,
        secret_dir: Optional[str] = None,
        cover_size: int = 256,
        secret_size: int = 64,
        secret_mode: str = "random",
    ) -> None:
        self.cover_paths = list_images(Path(cover_dir))
        if not self.cover_paths:
            raise ValueError(f"No cover images found in {cover_dir}")

        self.secret_paths = list_images(Path(secret_dir)) if secret_dir else []
        self.secret_mode = secret_mode if self.secret_paths else "random"

        self.cover_transform = transforms.Compose(
            [
                transforms.Resize((cover_size, cover_size)),
                transforms.ToTensor(),
            ]
        )
        self.secret_transform = transforms.Compose(
            [
                transforms.Resize((secret_size, secret_size)),
                transforms.ToTensor(),
            ]
        )
        self.secret_size = secret_size

    def __len__(self) -> int:
        return len(self.cover_paths)

    def _load_cover(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            return self.cover_transform(img.convert("RGB"))

    def _load_secret(self) -> torch.Tensor:
        if self.secret_mode == "image" and self.secret_paths:
            secret_path = random.choice(self.secret_paths)
            with Image.open(secret_path) as img:
                return self.secret_transform(img.convert("RGB"))

        return torch.rand(3, self.secret_size, self.secret_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cover = self._load_cover(self.cover_paths[idx])
        secret = self._load_secret()
        return cover, secret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cover images and dataset inputs.")
    parser.add_argument("--cover-source", type=str, default="", help="Path to local cover image folder.")
    parser.add_argument("--cover-output", type=str, default="ciphernet/data/cover")
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument(
        "--download-default",
        action="store_true",
        help="Download CIFAR10 and upscale to 256x256 as fallback covers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cover_output = Path(args.cover_output)

    if args.cover_source:
        saved = preprocess_and_save_images(
            source_dir=Path(args.cover_source),
            output_dir=cover_output,
            image_size=256,
            max_images=args.max_images,
        )
        print(f"Prepared {saved} cover images into {cover_output}.")
        return

    if args.download_default:
        saved = download_cover_dataset(cover_output, max_images=args.max_images)
        print(f"Downloaded and prepared {saved} cover images into {cover_output}.")
        return

    raise ValueError("Provide --cover-source or enable --download-default.")


if __name__ == "__main__":
    main()
