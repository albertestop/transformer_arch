from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from v1tovideo.image_autoencoder import encode_decode_image, load_sd3_vae

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "configs" / "image_vae_single.toml"
LOGGER = logging.getLogger(__name__)


@dataclass
class SingleRunConfig:
    image_path: Path
    output_dir: Path
    height: int
    width: int
    prefix: str


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_toml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fp:
        data = tomllib.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid TOML structure in {config_path}")
    return data


def _parse_config(config_path: Path) -> SingleRunConfig:
    data = _load_toml(config_path)
    run = data.get("run")
    if not isinstance(run, dict):
        raise ValueError(f"Config must define a [run] table: {config_path}")

    try:
        image_path = _resolve_repo_path(run["image_path"])
    except KeyError as exc:
        raise ValueError("Missing required config key: run.image_path") from exc

    output_dir = _resolve_repo_path(run.get("output_dir", "outputs/image_compression/single"))
    height = int(run.get("height", 144))
    width = int(run.get("width", 256))
    prefix = str(run.get("prefix", "sample"))

    if height <= 0 or width <= 0:
        raise ValueError("run.height and run.width must be positive integers")

    return SingleRunConfig(
        image_path=image_path,
        output_dir=output_dir,
        height=height,
        width=width,
        prefix=prefix,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = ArgumentParser(description="Run SD3 VAE compression on a single image.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to TOML config file (default: scripts/configs/image_vae_single.toml).",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    LOGGER.info("Using config: %s", config_path)

    config = _parse_config(config_path)
    LOGGER.info(
        "Running single-image VAE | image=%s | size=%dx%d",
        config.image_path,
        config.height,
        config.width,
    )

    LOGGER.info("Loading SD3 VAE model")
    vae = load_sd3_vae()
    result = encode_decode_image(
        image_path=config.image_path,
        output_dir=config.output_dir,
        vae=vae,
        target_height=config.height,
        target_width=config.width,
        save_prefix=config.prefix,
    )

    summary = {k: v for k, v in result.items() if isinstance(v, float)}
    LOGGER.info("Completed single-image VAE run")
    print(summary)


if __name__ == "__main__":
    main()
