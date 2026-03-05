from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class NeuralDataConfig:
    """Configuration for loading neural traces."""

    source: Literal["array", "proc_session_responses"] = "array"
    path: Path | None = None
    npz_key: str | None = None

    proc_session_root: Path | None = None
    sessions: list[str] | None = None
    responses_subdir: str = "data/responses"
    file_pattern: str = "*.npy"
    max_files_per_session: int | None = None
    transpose_proc_session: bool = True
    channel_mode: Literal["error", "truncate_min"] = "error"
    time_mode: Literal["error", "truncate_min"] = "error"

    batch_size: int = 32
    val_split: float = 0.1
    shuffle_train: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False


class NeuralTraceDataset(Dataset[torch.Tensor]):
    """Dataset wrapper for in-memory neural trace tensors with shape [N, T, C]."""

    def __init__(self, traces: torch.Tensor) -> None:
        if traces.ndim != 3:
            raise ValueError(f"Expected traces with shape [N, T, C], got {tuple(traces.shape)}")
        self._traces = traces.float().contiguous()

    @classmethod
    def from_file(cls, path: Path, npz_key: str | None = None) -> "NeuralTraceDataset":
        if not path.exists():
            raise FileNotFoundError(f"Neural trace file not found: {path}")

        if path.suffix == ".npy":
            data = np.load(path)
        elif path.suffix == ".npz":
            npz_data = np.load(path)
            if npz_key:
                if npz_key not in npz_data:
                    raise ValueError(f"npz_key '{npz_key}' not found in {path}")
                data = npz_data[npz_key]
            else:
                first_key = next(iter(npz_data.files), None)
                if first_key is None:
                    raise ValueError(f"No arrays found in {path}")
                data = npz_data[first_key]
        else:
            raise ValueError("Unsupported neural trace format. Use .npy or .npz")

        if data.ndim != 3:
            raise ValueError(f"Loaded traces must have shape [N, T, C], got {data.shape}")

        return cls(torch.from_numpy(data))

    @property
    def shape(self) -> tuple[int, int, int]:
        n, t, c = self._traces.shape
        return int(n), int(t), int(c)

    def __len__(self) -> int:
        return int(self._traces.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._traces[idx]


class ProcSessionResponsesDataset(Dataset[torch.Tensor]):
    """Dataset for proc_session response files where each file has shape [C, T] or [T, C]."""

    def __init__(
        self,
        file_paths: list[Path],
        transpose: bool,
        channel_mode: Literal["error", "truncate_min"],
        time_mode: Literal["error", "truncate_min"],
    ) -> None:
        if not file_paths:
            raise ValueError("No proc_session response files found")

        self._file_paths = file_paths
        self._transpose = transpose
        self._channel_mode = channel_mode
        self._time_mode = time_mode

        shapes_tc: list[tuple[int, int]] = []
        for path in self._file_paths:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D response arrays in {path}, got shape {arr.shape}")
            c, t = int(arr.shape[0]), int(arr.shape[1])
            shapes_tc.append((t, c) if transpose else (c, t))

        time_values = {t for t, _ in shapes_tc}
        channel_values = {c for _, c in shapes_tc}

        if time_mode == "error" and len(time_values) != 1:
            raise ValueError(
                f"Inconsistent sequence lengths across sessions/files: {sorted(time_values)}. "
                "Set data.time_mode = 'truncate_min' to align automatically."
            )
        if channel_mode == "error" and len(channel_values) != 1:
            raise ValueError(
                f"Inconsistent neuron counts across sessions/files: {sorted(channel_values)}. "
                "Set data.channel_mode = 'truncate_min' to align automatically."
            )

        self._time_dim = min(time_values) if time_mode == "truncate_min" else next(iter(time_values))
        self._channel_dim = (
            min(channel_values) if channel_mode == "truncate_min" else next(iter(channel_values))
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self._file_paths), self._time_dim, self._channel_dim

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self._file_paths[idx]).astype(np.float32, copy=False)
        if self._transpose:
            arr = arr.T

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array after transpose, got {arr.shape}")

        arr = arr[: self._time_dim, : self._channel_dim]
        return torch.from_numpy(arr).contiguous()


def _list_proc_session_response_files(
    proc_session_root: Path,
    sessions: list[str],
    responses_subdir: str,
    file_pattern: str,
    max_files_per_session: int | None,
) -> list[Path]:
    all_paths: list[Path] = []
    for session in sessions:
        responses_dir = proc_session_root / session / responses_subdir
        if not responses_dir.exists():
            raise FileNotFoundError(f"Session responses directory not found: {responses_dir}")

        session_paths = sorted([p for p in responses_dir.glob(file_pattern) if p.is_file()])
        if not session_paths:
            raise ValueError(f"No files matching '{file_pattern}' in {responses_dir}")

        if max_files_per_session is not None:
            if max_files_per_session <= 0:
                raise ValueError("data.max_files_per_session must be > 0")
            session_paths = session_paths[:max_files_per_session]

        all_paths.extend(session_paths)

    if not all_paths:
        raise ValueError("No proc_session response files selected")
    return all_paths


def build_dataset(config: NeuralDataConfig) -> Dataset[torch.Tensor]:
    """Build one dataset from config."""
    if config.channel_mode not in {"error", "truncate_min"}:
        raise ValueError("data.channel_mode must be 'error' or 'truncate_min'")
    if config.time_mode not in {"error", "truncate_min"}:
        raise ValueError("data.time_mode must be 'error' or 'truncate_min'")

    if config.source == "array":
        if config.path is None:
            raise ValueError("data.path is required when data.source = 'array'")
        return NeuralTraceDataset.from_file(config.path, npz_key=config.npz_key)

    if config.source == "proc_session_responses":
        if config.proc_session_root is None:
            raise ValueError("data.proc_session_root is required when data.source = 'proc_session_responses'")
        if not config.sessions:
            raise ValueError("data.sessions must contain at least one session for 'proc_session_responses'")
        if any(not session.strip() for session in config.sessions):
            raise ValueError("data.sessions contains an empty session name")

        file_paths = _list_proc_session_response_files(
            proc_session_root=config.proc_session_root,
            sessions=config.sessions,
            responses_subdir=config.responses_subdir,
            file_pattern=config.file_pattern,
            max_files_per_session=config.max_files_per_session,
        )
        return ProcSessionResponsesDataset(
            file_paths=file_paths,
            transpose=config.transpose_proc_session,
            channel_mode=config.channel_mode,
            time_mode=config.time_mode,
        )

    raise ValueError("Unsupported data.source. Use 'array' or 'proc_session_responses'")


def build_dataloaders(config: NeuralDataConfig) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor], Dataset[torch.Tensor]]:
    """Build train/validation dataloaders from configured neural data source."""
    dataset = build_dataset(config)

    val_split = float(config.val_split)
    if not 0.0 < val_split < 1.0:
        raise ValueError("data.val_split must be in (0, 1)")

    val_size = max(1, int(round(len(dataset) * val_split)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the dataset size")

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, dataset
