from __future__ import annotations

import torch
from torch import nn


class TemplateNeuralAutoencoder(nn.Module):
    """Starter template for custom neural autoencoder experiments.

    Expected input shape: [batch, sequence_length, num_channels]
    Forward return contract: (reconstruction, latents)
    """

    def __init__(
        self,
        sequence_length: int,
        num_channels: int,
        latent_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.latent_dim = int(latent_dim)

        input_dim = self.sequence_length * self.num_channels
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.reshape(z.shape[0], self.sequence_length, self.num_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents
