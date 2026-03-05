from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0
    device: str = "cuda"


@dataclass
class EvalSummary:
    train_loss: float
    val_loss: float
    val_mae: float
    latent_dim: int
    compression_ratio: float


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    for batch in dataloader:
        x = batch.to(device, non_blocking=True)
        recon, _ = model(x)
        loss = criterion(recon, x)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader[torch.Tensor],
    val_loader: DataLoader[torch.Tensor],
    config: TrainConfig,
) -> list[dict[str, float]]:
    """Train neural autoencoder and return epoch history."""
    device = _resolve_device(config.device)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    train_start = time.perf_counter()
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        train_loss = _run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        with torch.no_grad():
            val_loss = _run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=None,
                criterion=criterion,
                device=device,
            )

        row = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": float(time.perf_counter() - epoch_start),
        }
        history.append(row)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | epoch_time=%.2fs",
            epoch + 1,
            config.epochs,
            train_loss,
            val_loss,
            row["epoch_time_sec"],
        )

    LOGGER.info("Training completed | total_time=%.2fs", time.perf_counter() - train_start)
    return history


def evaluate_autoencoder(
    model: nn.Module,
    dataloader: DataLoader[torch.Tensor],
    device: str = "cuda",
) -> dict[str, float]:
    """Compute reconstruction metrics on a dataloader."""
    dev = _resolve_device(device)
    model.eval().to(dev)

    mse_values: list[float] = []
    mae_values: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch.to(dev, non_blocking=True)
            recon, _ = model(x)
            mse_values.append(float(torch.mean((recon - x) ** 2).cpu()))
            mae_values.append(float(torch.mean(torch.abs(recon - x)).cpu()))

    mse = float(np.mean(mse_values)) if mse_values else float("nan")
    mae = float(np.mean(mae_values)) if mae_values else float("nan")

    return {
        "mse": mse,
        "mae": mae,
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_reconstruction_artifacts(
    model: nn.Module,
    sample_batch: torch.Tensor,
    output_dir: Path,
    device: str = "cuda",
    prefix: str = "sample",
) -> None:
    """Save input traces, latent vectors, and reconstructed traces for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_device(device)
    model.eval().to(dev)

    with torch.no_grad():
        x = sample_batch.to(dev)
        recon, latents = model(x)

    torch.save(x.cpu(), output_dir / f"{prefix}.input.pt")
    torch.save(latents.cpu(), output_dir / f"{prefix}.latents.pt")
    torch.save(recon.cpu(), output_dir / f"{prefix}.reconstruction.pt")


def save_reconstruction_plots(
    model: nn.Module,
    sample_batch: torch.Tensor,
    output_dir: Path,
    device: str = "cuda",
    prefix: str = "sample",
    num_neurons: int = 3,
) -> None:
    """Save simple before/after plots for neuron traces and population heatmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = _resolve_device(device)
    model.eval().to(dev)

    with torch.no_grad():
        x = sample_batch.to(dev)
        recon, _ = model(x)

    if x.ndim != 3 or x.shape[0] == 0:
        raise ValueError(f"Expected sample_batch shape [B, T, C], got {tuple(x.shape)}")

    original = x[0].detach().cpu().numpy()  # [T, C]
    reconstructed = recon[0].detach().cpu().numpy()  # [T, C]
    time_axis = np.arange(original.shape[0], dtype=np.float32)

    n_channels = int(original.shape[1])
    k = max(1, min(int(num_neurons), n_channels))
    neuron_indices = np.linspace(0, n_channels - 1, num=k, dtype=int).tolist()

    for neuron_idx in neuron_indices:
        plt.figure(figsize=(8, 3))
        plt.plot(time_axis, original[:, neuron_idx], label="Before")
        plt.plot(time_axis, reconstructed[:, neuron_idx], label="After")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.title(f"Neuron {neuron_idx}: before vs after reconstruction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}.neuron_{neuron_idx}.png")
        plt.close()

    stacked = np.concatenate([original, reconstructed], axis=0)
    vmin = 0
    vmax = float(original[original != 0].mean())

    plt.figure(figsize=(10, 5))
    plt.imshow(original.T, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Response Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Neuron")
    plt.title("Population activity before reconstruction")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}.population_before.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(reconstructed.T, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Response Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Neuron")
    plt.title("Population activity after reconstruction")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}.population_after.png")
    plt.close()
