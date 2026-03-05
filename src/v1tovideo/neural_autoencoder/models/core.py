from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelSpec:
    """Model specification used by the architecture factory."""

    architecture: str
    sequence_length: int
    num_channels: int
    latent_dim: int
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


class BaseNeuralAutoencoder(nn.Module):
    """Base interface for neural autoencoders."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class MLPNeuralAutoencoder(BaseNeuralAutoencoder):
    """Simple baseline autoencoder that flattens the trace sequence."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        input_dim = spec.sequence_length * spec.num_channels

        self._sequence_length = spec.sequence_length
        self._num_channels = spec.num_channels

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(spec.latent_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.reshape(z.shape[0], self._sequence_length, self._num_channels)


class TransformerNeuralAutoencoder(BaseNeuralAutoencoder):
    """Transformer skeleton with a learned latent bottleneck token set."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.sequence_length = spec.sequence_length
        self.num_channels = spec.num_channels

        self.input_proj = nn.Linear(spec.num_channels, spec.hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, spec.sequence_length, spec.hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden_dim,
            nhead=spec.num_heads,
            dim_feedforward=spec.hidden_dim * 4,
            dropout=spec.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=spec.num_layers)

        self.to_latent = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, spec.latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(spec.latent_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden_dim,
            nhead=spec.num_heads,
            dim_feedforward=spec.hidden_dim * 4,
            dropout=spec.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=spec.num_layers)
        self.output_proj = nn.Linear(spec.hidden_dim, spec.num_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.pos_embed[:, : x.shape[1], :]
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.to_latent(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        seed = self.from_latent(z).unsqueeze(1).repeat(1, self.sequence_length, 1)
        seed = seed + self.pos_embed[:, : self.sequence_length, :]
        decoded = self.decoder(seed)
        return self.output_proj(decoded)


def build_model(spec: ModelSpec) -> BaseNeuralAutoencoder:
    """Construct a neural autoencoder from a config specification."""
    architecture = spec.architecture.lower().strip()
    if architecture == "mlp":
        return MLPNeuralAutoencoder(spec)
    if architecture == "transformer":
        return TransformerNeuralAutoencoder(spec)

    supported = ["mlp", "transformer"]
    raise ValueError(f"Unsupported architecture '{spec.architecture}'. Supported: {supported}")
