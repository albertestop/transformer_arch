from __future__ import annotations

import json
import logging
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.neural_autoencoder import (
    ModelSpec,
    NeuralDataConfig,
    TrainConfig,
    build_dataloaders,
    build_model,
    build_model_from_target,
    evaluate_autoencoder,
    save_checkpoint,
    save_reconstruction_plots,
    save_reconstruction_artifacts,
    train_autoencoder,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "neural_ae_experiment.toml"
LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    data: NeuralDataConfig
    model: ModelSpec | None
    model_target: str | None
    model_kwargs: dict[str, Any]
    expected_trace_shape: tuple[int, int] | None
    latent_dim: int | None
    train: TrainConfig
    output_dir: Path


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _resolve_maybe_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_toml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fp:
        data = tomllib.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid TOML structure in {config_path}")
    return data


def _parse_config(config_path: Path) -> ExperimentConfig:
    data = _load_toml(config_path)

    data_cfg = data.get("data")
    model_cfg = data.get("model")
    train_cfg = data.get("train", {})
    output_cfg = data.get("output", {})

    if not isinstance(data_cfg, dict):
        raise ValueError("Config must define [data]")
    if not isinstance(model_cfg, dict):
        raise ValueError("Config must define [model]")
    if not isinstance(train_cfg, dict):
        raise ValueError("Config [train] must be a table")
    if not isinstance(output_cfg, dict):
        raise ValueError("Config [output] must be a table")

    data_source = str(data_cfg.get("source", "array")).strip().lower()
    sessions_raw = data_cfg.get("sessions")
    sessions: list[str] | None
    if sessions_raw is None:
        sessions = None
    else:
        if not isinstance(sessions_raw, list):
            raise ValueError("data.sessions must be a list of session names")
        sessions = [str(session) for session in sessions_raw]

    data_config = NeuralDataConfig(
        source=data_source,  # validated in data module
        path=_resolve_repo_path(data_cfg["path"]) if "path" in data_cfg else None,
        npz_key=str(data_cfg["npz_key"]) if "npz_key" in data_cfg else None,
        proc_session_root=_resolve_maybe_repo_path(data_cfg["proc_session_root"])
        if "proc_session_root" in data_cfg
        else None,
        sessions=sessions,
        responses_subdir=str(data_cfg.get("responses_subdir", "data/responses")),
        file_pattern=str(data_cfg.get("file_pattern", "*.npy")),
        max_files_per_session=int(data_cfg["max_files_per_session"])
        if "max_files_per_session" in data_cfg
        else None,
        transpose_proc_session=bool(data_cfg.get("transpose_proc_session", True)),
        channel_mode=str(data_cfg.get("channel_mode", "error")),
        time_mode=str(data_cfg.get("time_mode", "error")),
        batch_size=int(data_cfg.get("batch_size", 32)),
        val_split=float(data_cfg.get("val_split", 0.1)),
        shuffle_train=bool(data_cfg.get("shuffle_train", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
    )

    architecture = str(model_cfg.get("architecture", "transformer")).strip().lower()
    supported_architectures = {"mlp", "transformer", "target"}
    if architecture not in supported_architectures:
        raise ValueError(
            f"Unsupported model.architecture '{architecture}'. Supported: {sorted(supported_architectures)}"
        )

    model_target_raw = model_cfg.get("target")
    model_target = str(model_target_raw).strip() if model_target_raw is not None else None

    model_kwargs_raw = model_cfg.get("kwargs", {})
    if not isinstance(model_kwargs_raw, dict):
        raise ValueError("Config [model.kwargs] must be a table")
    model_kwargs: dict[str, Any] = dict(model_kwargs_raw)

    model_config: ModelSpec | None = None
    expected_trace_shape: tuple[int, int] | None = None
    latent_dim: int | None = None

    if architecture == "target":
        if not model_target:
            raise ValueError("model.target is required when model.architecture = 'target'")
        if "sequence_length" in model_cfg and "num_channels" in model_cfg:
            expected_trace_shape = (int(model_cfg["sequence_length"]), int(model_cfg["num_channels"]))
        if "latent_dim" in model_cfg:
            latent_dim = int(model_cfg["latent_dim"])
    else:
        if model_target:
            raise ValueError(
                "model.target is only allowed when model.architecture = 'target'. "
                f"Current architecture is '{architecture}'."
            )
        if model_kwargs:
            raise ValueError(
                "model.kwargs is only allowed when model.architecture = 'target'. "
                "Use top-level model fields for built-in architectures."
            )
        model_config = ModelSpec(
            architecture=architecture,
            sequence_length=int(model_cfg["sequence_length"]),
            num_channels=int(model_cfg["num_channels"]),
            latent_dim=int(model_cfg.get("latent_dim", 128)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        expected_trace_shape = (model_config.sequence_length, model_config.num_channels)
        latent_dim = model_config.latent_dim

    train_config = TrainConfig(
        epochs=int(train_cfg.get("epochs", 25)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        device=str(train_cfg.get("device", "cuda")),
    )

    output_dir = _resolve_repo_path(output_cfg.get("dir", "outputs/neural_autoencoder"))

    return ExperimentConfig(
        data=data_config,
        model=model_config,
        model_target=model_target,
        model_kwargs=model_kwargs,
        expected_trace_shape=expected_trace_shape,
        latent_dim=latent_dim,
        train=train_config,
        output_dir=output_dir,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = ArgumentParser(description="Train/evaluate a configurable neural autoencoder skeleton.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config file (default: scripts/configs/neural_ae_experiment.toml).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    LOGGER.info("Using config: %s", config_path)

    config = _parse_config(config_path)
    LOGGER.info("Preparing dataset from source: %s", config.data.source)

    train_loader, val_loader, dataset = build_dataloaders(config.data)
    LOGGER.info("Dataset loaded | samples=%d | shape=%s", len(dataset), getattr(dataset, "shape", None))

    dataset_shape = dataset.shape[1:]
    if config.expected_trace_shape is not None and dataset_shape != config.expected_trace_shape:
        raise ValueError(
            f"Model expects traces [T, C]={config.expected_trace_shape}, but dataset has {dataset_shape}. "
            "Update [model] sequence_length/num_channels or your data preprocessing."
        )

    if config.model_target:
        model = build_model_from_target(config.model_target, kwargs=config.model_kwargs)
        model_name = config.model_target
    else:
        if config.model is None:
            raise RuntimeError("Internal error: no model spec available")
        model = build_model(config.model)
        model_name = config.model.architecture
    LOGGER.info("Model initialized: %s", model_name)

    LOGGER.info("Training started | epochs=%d | device=%s", config.train.epochs, config.train.device)
    history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
    )

    eval_metrics = evaluate_autoencoder(model=model, dataloader=val_loader, device=config.train.device)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "model.pt"
    save_checkpoint(model, checkpoint_path)

    first_batch = next(iter(val_loader))
    save_reconstruction_artifacts(
        model=model,
        sample_batch=first_batch,
        output_dir=output_dir,
        device=config.train.device,
        prefix="val_sample",
    )
    save_reconstruction_plots(
        model=model,
        sample_batch=first_batch,
        output_dir=output_dir,
        device=config.train.device,
        prefix="val_sample",
        num_neurons=3,
    )
    LOGGER.info("Saved reconstruction plots (3 neurons + population heatmaps)")

    summary: dict[str, Any] = {
        "dataset_shape": dataset.shape,
        "model_name": model_name,
        "model_target": config.model_target,
        "model_kwargs": config.model_kwargs,
        "latent_dim": config.latent_dim,
        "train_loss": history[-1]["train_loss"] if history else float("nan"),
        "val_loss": history[-1]["val_loss"] if history else float("nan"),
        "val_mse": eval_metrics["mse"],
        "val_mae": eval_metrics["mae"],
    }
    if config.expected_trace_shape is not None and config.latent_dim is not None:
        summary["compression_ratio"] = float(
            (config.expected_trace_shape[0] * config.expected_trace_shape[1]) / config.latent_dim
        )
    else:
        summary["compression_ratio"] = None

    with (output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    LOGGER.info("Run finished | output_dir=%s", output_dir)
    print(summary)


if __name__ == "__main__":
    main()
