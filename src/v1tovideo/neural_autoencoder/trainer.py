from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 25
    learning_rate: float = 1e-4
    lr_start: float | None = None
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0
    device: str = "cuda:1"
    loss_name: str = "masked_mse"
    poisson_log_input: bool = True
    poisson_full: bool = False
    poisson_eps: float = 1e-8
    loss_weight_id: float = 1.0
    loss_weight_time: float = 1.0
    loss_weight_rec: float = 1.0
    combined_loss_name_id: str = "cross_entropy"
    combined_loss_name_time: str = "masked_mse"
    combined_loss_name_rec: str = "masked_mse"


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _lightning_trainer_kwargs(device: str) -> dict[str, Any]:
    resolved = _resolve_device(device)
    if resolved.type == "cuda":
        # Respect an explicit CUDA index from config (e.g. "cuda:1").
        if resolved.index is not None:
            return {"accelerator": "gpu", "devices": [resolved.index]}
        return {"accelerator": "gpu", "devices": 1}
    if resolved.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    return {"accelerator": "cpu", "devices": 1}


class AutoencoderLightningModule(pl.LightningModule):
    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.config = config
        self._loss_name = str(config.loss_name).strip().lower()
        supported_losses = {"masked_mse", "masked_mae", "poisson_nll", "combined"}
        if self._loss_name not in supported_losses:
            raise ValueError(f"Unsupported loss_name '{config.loss_name}'. Supported: {sorted(supported_losses)}")
        self._combined_loss_names = {
            "id": str(config.combined_loss_name_id).strip().lower(),
            "time": str(config.combined_loss_name_time).strip().lower(),
            "rec": str(config.combined_loss_name_rec).strip().lower(),
        }
        supported_id_losses = {"cross_entropy"}
        supported_value_losses = {"masked_mse", "masked_mae", "poisson_nll"}
        if self._combined_loss_names["id"] not in supported_id_losses:
            raise ValueError(
                "Unsupported combined id loss "
                f"'{config.combined_loss_name_id}'. Supported: {sorted(supported_id_losses)}"
            )
        for key in ("time", "rec"):
            if self._combined_loss_names[key] not in supported_value_losses:
                raise ValueError(
                    f"Unsupported combined {key} loss '{self._combined_loss_names[key]}'. "
                    f"Supported: {sorted(supported_value_losses)}"
                )
        self._loss_weights = (
            float(config.loss_weight_id),
            float(config.loss_weight_time),
            float(config.loss_weight_rec),
        )
        self._poisson_nll = nn.PoissonNLLLoss(
            log_input=bool(config.poisson_log_input),
            full=bool(config.poisson_full),
            eps=float(config.poisson_eps),
            reduction="none",
        )

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(batch):
            x = batch
            padding_mask = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
            return x, x, padding_mask
        if isinstance(batch, (tuple, list)) and len(batch) >= 3 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[2]):
            x = batch[0]
            padding_mask = batch[2].bool()
            target = batch[3] if len(batch) >= 4 and torch.is_tensor(batch[3]) else x
            return x, target, padding_mask
        raise ValueError(f"Unsupported batch format: {type(batch)!r}")

    def _masked_mse(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (((recon - x) ** 2) * valid).sum() / denom

    def _masked_mae(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum().clamp_min(1.0)
        return (torch.abs(recon - x) * valid).sum() / denom

    def _masked_poisson_nll(self, recon: torch.Tensor, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        denom = valid.sum().clamp_min(1.0)
        per_element = self._poisson_nll(recon, x)
        return (per_element * valid).sum() / denom

    def _masked_combined_loss(
        self,
        id_logits: torch.Tensor,
        time_pred: torch.Tensor,
        rec_pred: torch.Tensor,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = ~padding_mask
        id_target = x[..., 0].long()
        time_target = x[..., 1]
        rec_target = x[..., 2]

        ignore_index = -100
        id_target_masked = id_target.masked_fill(~valid, ignore_index)
        w_id, w_time, w_rec = self._loss_weights
        if w_id == 0.0:
            loss_id = id_logits.new_zeros(())
        else:
            loss_id = F.cross_entropy(
                id_logits.reshape(-1, id_logits.shape[-1]),
                id_target_masked.reshape(-1),
                ignore_index=ignore_index,
            )

        loss_time = self._masked_value_loss(
            time_pred.squeeze(-1),
            time_target,
            padding_mask,
            self._combined_loss_names["time"],
        )
        loss_rec = self._masked_value_loss(
            rec_pred.squeeze(-1),
            rec_target,
            padding_mask,
            self._combined_loss_names["rec"],
        )

        total = (w_id * loss_id) + (w_time * loss_time) + (w_rec * loss_rec)
        return total, {"loss_id": loss_id, "loss_time": loss_time, "loss_rec": loss_rec}

    def _masked_value_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor,
        loss_name: str,
    ) -> torch.Tensor:
        pred_3d = pred.unsqueeze(-1)
        target_3d = target.unsqueeze(-1)
        if loss_name == "masked_mse":
            return self._masked_mse(pred_3d, target_3d, padding_mask)
        if loss_name == "masked_mae":
            return self._masked_mae(pred_3d, target_3d, padding_mask)
        if loss_name == "poisson_nll":
            return self._masked_poisson_nll(pred_3d, target_3d, padding_mask)
        raise ValueError(f"Unsupported value loss '{loss_name}'")

    def _forward_outputs(self, x: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x, padding_mask=padding_mask)
        if not isinstance(out, (tuple, list)):
            raise ValueError("Model forward must return a tuple/list")
        if len(out) == 2:
            recon, latents = out
            return {"recon": recon, "latents": latents}
        if len(out) == 4:
            id_logits, time_pred, rec_pred, latents = out
            recon = self.model.predict(x, padding_mask)
            return {
                "id_logits": id_logits,
                "time_pred": time_pred,
                "rec_pred": rec_pred,
                "recon": recon,
                "latents": latents,
            }
        raise ValueError(f"Unsupported model output tuple length: {len(out)}")

    def _compute_loss_and_terms(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._loss_name == "masked_mse":
            return self._masked_mse(outputs["recon"], target, padding_mask), {}
        if self._loss_name == "masked_mae":
            return self._masked_mae(outputs["recon"], target, padding_mask), {}
        if self._loss_name == "poisson_nll":
            return self._masked_poisson_nll(outputs["recon"], target, padding_mask), {}
        if self._loss_name == "combined":
            if not all(k in outputs for k in ("id_logits", "time_pred", "rec_pred")):
                raise ValueError("combined loss requires model outputs: id_logits, time_pred, rec_pred")
            return self._masked_combined_loss(outputs["id_logits"], outputs["time_pred"], outputs["rec_pred"], target, padding_mask)
        raise ValueError(f"Unsupported loss_name '{self._loss_name}'")

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss, _ = self._compute_loss_and_terms(outputs, target, padding_mask)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, target, padding_mask = self._unpack_batch(batch)
        outputs = self._forward_outputs(x, padding_mask)
        loss = self._compute_loss(outputs, target, padding_mask)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, target, padding_mask = self._unpack_batch(batch)
        outputs = self._forward_outputs(x, padding_mask)
        loss, terms = self._compute_loss_and_terms(outputs, target, padding_mask)
        mse = self._masked_mse(outputs["recon"], target, padding_mask)
        mae = self._masked_mae(outputs["recon"], target, padding_mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        for name, value in terms.items():
            self.log(f"val_{name}", value, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("val_mse", mse, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("val_mae", mae, on_step=False, on_epoch=True, batch_size=x.shape[0])
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        target_lr = float(self.config.learning_rate)
        initial_lr = float(self.config.lr_start) if self.config.lr_start is not None else target_lr
        optimizer = AdamW(
            self.model.parameters(),
            lr=target_lr,
            weight_decay=self.config.weight_decay,
        )
        if math.isclose(initial_lr, target_lr):
            raise ValueError("Initial and final LR are the same")
        start_factor = initial_lr / target_lr
        total_steps = max(int(self.config.epochs) - 1, 1)

        def lr_lambda(epoch: int) -> float:
            progress = min(max(float(epoch) / total_steps, 0.0), 1.0)
            return start_factor + ((1.0 - start_factor) * progress)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }



class TrainHistoryCallback(pl.Callback):  # type: ignore[misc]
    _TERM_NAMES = ("loss_id", "loss_time", "loss_rec")

    def __init__(self, train_loader: DataLoader[Any]) -> None:
        self.train_loader = train_loader
        self.history: list[dict[str, float]] = []
        self._epoch_start_time = 0.0
        self._latest_train_loss = float("nan")
        self._latest_train_terms = {name: float("nan") for name in self._TERM_NAMES}
        self._logged_epochs: set[float] = set()

    def _move_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(item, device) for item in batch)
        if isinstance(batch, list):
            return [self._move_batch_to_device(item, device) for item in batch]
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value, device) for key, value in batch.items()}
        return batch

    def _compute_train_eval_metrics(self, pl_module: AutoencoderLightningModule) -> dict[str, float]:
        was_training = pl_module.training
        pl_module.eval()
        total_loss = 0.0
        total_terms = {name: 0.0 for name in self._TERM_NAMES}
        seen_terms = set()
        total_samples = 0
        device = pl_module.device

        with torch.inference_mode():
            for batch in self.train_loader:
                batch = self._move_batch_to_device(batch, device)
                x, target, padding_mask = pl_module._unpack_batch(batch)
                outputs = pl_module._forward_outputs(x, padding_mask)
                loss, terms = pl_module._compute_loss_and_terms(outputs, target, padding_mask)
                batch_size = int(x.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                for name in self._TERM_NAMES:
                    if name in terms:
                        total_terms[name] += float(terms[name].detach().cpu()) * batch_size
                        seen_terms.add(name)
                total_samples += batch_size

        if was_training:
            pl_module.train()
        denom = max(total_samples, 1)
        metrics = {"train_loss": total_loss / denom}
        for name in self._TERM_NAMES:
            metrics[f"train_{name}"] = total_terms[name] / denom if name in seen_terms else float("nan")
        return metrics

    def _log_epoch_row(self, trainer: Any, row: dict[str, float]) -> None:
        epoch = row["epoch"]
        if epoch in self._logged_epochs:
            return
        if row["train_loss"] != row["train_loss"] or row["val_loss"] != row["val_loss"]:
            return
        self._logged_epochs.add(epoch)
        LOGGER.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | epoch_time=%.2fs",
            int(epoch),
            trainer.max_epochs,
            row["train_loss"],
            row["val_loss"],
            row["epoch_time_sec"],
        )

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        train_metrics = self._compute_train_eval_metrics(pl_module)
        self._latest_train_loss = train_metrics["train_loss"]
        self._latest_train_terms = {name: train_metrics[f"train_{name}"] for name in self._TERM_NAMES}
        pl_module.log("train_loss", self._latest_train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in self._latest_train_terms.items():
            pl_module.log(f"train_{name}", value, on_step=False, on_epoch=True)
        if self.history and self.history[-1]["epoch"] == float(trainer.current_epoch + 1):
            self.history[-1]["train_loss"] = self._latest_train_loss
            for name, value in self._latest_train_terms.items():
                self.history[-1][f"train_{name}"] = value
            self.history[-1]["epoch_time_sec"] = float(time.perf_counter() - self._epoch_start_time)
            self._log_epoch_row(trainer, self.history[-1])

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = trainer.callback_metrics
        row = {
            "epoch": float(trainer.current_epoch + 1),
            "train_loss": self._latest_train_loss,
            **{f"train_{name}": self._latest_train_terms[name] for name in self._TERM_NAMES},
            "val_loss": float(metrics["val_loss"].detach().cpu()) if "val_loss" in metrics else float("nan"),
            **{
                f"val_{name}": float(metrics[f"val_{name}"].detach().cpu())
                if f"val_{name}" in metrics
                else float("nan")
                for name in self._TERM_NAMES
            },
            "epoch_time_sec": float(time.perf_counter() - self._epoch_start_time),
        }
        self.history.append(row)
        self._log_epoch_row(trainer, row)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    config: TrainConfig,
    logger: Any | bool = False,
) -> list[dict[str, float]]:
    """Train neural autoencoder with PyTorch Lightning and return epoch history."""
    lightning_model = AutoencoderLightningModule(model=model, config=config)
    history_callback = TrainHistoryCallback(train_loader)
    train_start = time.perf_counter()
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        gradient_clip_val=float(config.grad_clip_norm) if config.grad_clip_norm is not None else 0.0,
        enable_progress_bar=False,
        callbacks=[history_callback],
        **_lightning_trainer_kwargs(config.device),
    )
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    LOGGER.info("Training completed | total_time=%.2fs", time.perf_counter() - train_start)
    return history_callback.history


def evaluate_autoencoder(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: str = "cuda",
) -> dict[str, float]:
    """Compute reconstruction metrics with PyTorch Lightning validation."""
    eval_config = TrainConfig(device=device)
    lightning_model = AutoencoderLightningModule(model=model, config=eval_config)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        **_lightning_trainer_kwargs(device),
    )
    metrics = trainer.validate(lightning_model, dataloaders=dataloader, verbose=False)
    if not metrics:
        return {"mse": float("nan"), "mae": float("nan")}
    return {
        "mse": float(metrics[0].get("val_mse", float("nan"))),
        "mae": float(metrics[0].get("val_mae", float("nan"))),
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
