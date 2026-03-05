"""Neural activity autoencoder components."""

from .data import NeuralDataConfig, NeuralTraceDataset, build_dataloaders
from .models import BaseNeuralAutoencoder, ModelSpec, build_model, build_model_from_target
from .trainer import (
    EvalSummary,
    TrainConfig,
    evaluate_autoencoder,
    save_checkpoint,
    save_reconstruction_plots,
    save_reconstruction_artifacts,
    train_autoencoder,
)

__all__ = [
    "BaseNeuralAutoencoder",
    "EvalSummary",
    "ModelSpec",
    "NeuralDataConfig",
    "NeuralTraceDataset",
    "TrainConfig",
    "build_dataloaders",
    "build_model",
    "build_model_from_target",
    "evaluate_autoencoder",
    "save_checkpoint",
    "save_reconstruction_plots",
    "save_reconstruction_artifacts",
    "train_autoencoder",
]
