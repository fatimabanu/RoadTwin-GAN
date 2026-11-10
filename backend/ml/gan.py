"""Conditional DCGAN (PyTorch) for synthetic road damage image generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CLASS_NAMES, CLASS_TO_IDX, DEFAULT_SEED, IMAGE_SIZE, LATENT_DIM, PREVIEW_SIZE
from ml.preprocessing import encode_image_base64, tensor_to_bgr


def _init_weights(module: nn.Module) -> None:
    """Apply DCGAN-standard Gaussian weight initialization."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class RoadConditionalGenerator(nn.Module):
    """Conditional DCGAN Generator producing 64x64 RGB road images from (z, class_id)."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = len(CLASS_NAMES),
        embed_dim: int = 16,
        base_channels: int = 48,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        in_channels = latent_dim + embed_dim
        self.net = nn.Sequential(
            # (in_channels, 1, 1) -> (base*4, 4, 4)
            nn.ConvTranspose2d(in_channels, base_channels * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # -> (base*2, 8, 8)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # -> (base, 16, 16)
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # -> (base//2, 32, 32)
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(True),
            # -> (3, 64, 64)
            nn.ConvTranspose2d(base_channels // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embed = self.label_embedding(labels)
        x = torch.cat([z, embed], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class RoadConditionalDiscriminator(nn.Module):
    """Conditional DCGAN Discriminator scoring real vs synthetic 64x64 road patches."""

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        image_size: int = IMAGE_SIZE,
        base_channels: int = 48,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)

        self.net = nn.Sequential(
            # (4, 64, 64) -> (base//2, 32, 32)
            nn.Conv2d(4, base_channels // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (base, 16, 16)
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (base*2, 8, 8)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (base*4, 4, 4)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (1, 1, 1)
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )
        self.apply(_init_weights)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_map = self.label_embedding(labels).view(-1, 1, self.image_size, self.image_size)
        x = torch.cat([images, label_map], dim=1)
        return self.net(x).view(-1)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_conditional_gan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 55,
    batch_size: int = 24,
    lr: float = 0.00025,
    latent_dim: int = LATENT_DIM,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> tuple[RoadConditionalGenerator, list[dict[str, float]]]:
    """Train the Conditional DCGAN on real road images and return (generator, loss_history)."""
    set_global_seed(seed)
    torch_device = torch.device(device)

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)

    generator = RoadConditionalGenerator(latent_dim=latent_dim).to(torch_device)
    discriminator = RoadConditionalDiscriminator().to(torch_device)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    history: list[dict[str, float]] = []
    generator.train()
    discriminator.train()

    for epoch in range(1, epochs + 1):
        d_losses: list[float] = []
        g_losses: list[float] = []

        for real_imgs, labels in loader:
            real_imgs = real_imgs.to(torch_device)
            labels = labels.to(torch_device)
            bsz = real_imgs.size(0)

            real_targets = torch.full((bsz,), 0.9, device=torch_device)
            fake_targets = torch.zeros(bsz, device=torch_device)

            # Update Discriminator
            opt_d.zero_grad()
            real_logits = discriminator(real_imgs, labels)
            loss_real = criterion(real_logits, real_targets)

            z = torch.randn(bsz, latent_dim, device=torch_device)
            fake_imgs = generator(z, labels)
            fake_logits = discriminator(fake_imgs.detach(), labels)
            loss_fake = criterion(fake_logits, fake_targets)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # Update Generator
            opt_g.zero_grad()
            gen_logits = discriminator(fake_imgs, labels)
            loss_g = criterion(gen_logits, torch.ones(bsz, device=torch_device))
            loss_g.backward()
            opt_g.step()

            d_losses.append(float(loss_d.item()))
            g_losses.append(float(loss_g.item()))

        history.append(
            {
                "epoch": epoch,
                "d_loss": round(float(np.mean(d_losses)), 4),
                "g_loss": round(float(np.mean(g_losses)), 4),
            }
        )

    generator.eval()
    return generator, history


def save_gan_artifacts(
    generator: RoadConditionalGenerator,
    history: list[dict[str, float]],
    artifacts_dir: Path,
) -> Path:
    """Persist trained GAN generator weights and training loss history."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifacts_dir / "gan_generator.pt"
    torch.save(
        {
            "state_dict": generator.state_dict(),
            "latent_dim": generator.latent_dim,
            "num_classes": generator.num_classes,
            "base_channels": generator.base_channels,
        },
        weights_path,
    )
    history_path = artifacts_dir / "gan_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return weights_path


def load_gan_generator(weights_path: Path, device: str = "cpu") -> RoadConditionalGenerator:
    """Load a trained RoadConditionalGenerator checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = RoadConditionalGenerator(
        latent_dim=int(checkpoint.get("latent_dim", LATENT_DIM)),
        num_classes=int(checkpoint.get("num_classes", len(CLASS_NAMES))),
        base_channels=int(checkpoint.get("base_channels", 48)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def generate_synthetic_tensors(
    generator: RoadConditionalGenerator,
    damage_class: str,
    count: int = 4,
    seed: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate normalized [-1, 1] synthetic tensors conditioned on damage_class."""
    if damage_class not in CLASS_TO_IDX:
        raise ValueError(f"Unsupported damage_class '{damage_class}'. Expected one of {CLASS_NAMES}.")
    if count < 1 or count > 32:
        raise ValueError("count must be between 1 and 32.")

    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(int(seed))

    z = torch.randn((count, generator.latent_dim), generator=rng, device=device)
    labels = torch.full((count,), CLASS_TO_IDX[damage_class], dtype=torch.long, device=device)
    return generator(z, labels).cpu()


def generate_synthetic_images(
    generator: RoadConditionalGenerator,
    damage_class: str,
    count: int = 4,
    seed: int | None = None,
    preview_size: int = PREVIEW_SIZE,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """Generate synthetic images formatted as base64 previews with pixel summary stats."""
    tensors = generate_synthetic_tensors(
        generator=generator,
        damage_class=damage_class,
        count=count,
        seed=seed,
        device=device,
    )
    results: list[dict[str, object]] = []
    for idx in range(tensors.size(0)):
        bgr = tensor_to_bgr(tensors[idx], target_size=preview_size)
        results.append(
            {
                "index": idx + 1,
                "damage_type": damage_class,
                "width": preview_size,
                "height": preview_size,
                "mean_brightness": round(float(np.mean(bgr)), 2),
                "image_base64": encode_image_base64(bgr),
            }
        )
    return results
