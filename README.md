# V1-to-Video Reconstruction

This repository contains the initial implementation for a 3-stage pipeline to reconstruct viewed video from V1 neural recordings:

1. Neural autoencoder: compress/decompress neural activity signals.
2. Image autoencoder: compress/decompress video frames.
3. Latent mapper: map neural latents to image latents.

Stage 1 and stage 2 are now implemented as research skeletons. The image module uses the Stable Diffusion 3 VAE from `diffusers`; the neural module provides configurable baseline architectures intended for rapid experimentation.

## Current status

- Implemented:
  - SD3 VAE image compression/decompression module.
  - Reconstruction quality metrics (MSE, MAE, frequency-domain scores, SSIM, compression ratio).
  - Single-image and batch image evaluation scripts.
  - Config-driven neural trace autoencoder skeleton:
    - Data loading from `.npy` / `.npz` neural trace tensors.
    - Configurable architecture factory (`mlp`, `transformer`).
    - Train + validation loop and checkpoint/artifact saving.
- Planned:
  - Transformer-based neural-latent -> image-latent mapper.

## Project structure

- `src/v1tovideo/image_autoencoder/`: image compression module.
- `src/v1tovideo/neural_autoencoder/`: neural trace compression module.
- `src/v1tovideo/latent_mapper/`: placeholder package for latent mapping model.
- `scripts/`: runnable entrypoints.
- `docs/`: project documentation and roadmap.
- `legacy/`: original prototype files retained for reference.
- `assets/`: tracked sample images/results from the original prototype.
- `data/`: local datasets (ignored by git).
- `outputs/`: generated artifacts (ignored by git).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your Hugging Face access requires authentication, export:

```bash
export HF_TOKEN=<your_token>
```

## Run examples

Single image:

```bash
python scripts/run_image_vae_single.py
```

Batch frame evaluation:

```bash
python scripts/run_image_vae_batch.py
```

Neural autoencoder experiment:

```bash
python scripts/run_neural_ae_experiment.py
```

Custom config path:

```bash
python scripts/run_image_vae_single.py --config scripts/configs/image_vae_single.toml
python scripts/run_image_vae_batch.py --config scripts/configs/image_vae_batch.toml
python scripts/run_neural_ae_experiment.py --config scripts/configs/neural_ae_experiment.toml
```

## Neural trace data format

For `run_neural_ae_experiment.py`, the dataset file must contain a 3D tensor with shape `[N, T, C]`:

- `N`: number of examples (trials/windows).
- `T`: number of time bins per example.
- `C`: number of recorded neurons/channels.

Supported files:

- `.npy` containing one array of shape `[N, T, C]`.
- `.npz` containing one or more arrays; choose with `data.npz_key`.

proc_session session mode is also supported:

- Set `data.source = "proc_session_responses"`.
- Set `data.proc_session_root` and `data.sessions = ["session_a", "session_b"]`.
- The loader reads `<proc_session_root>/<session>/data/responses/*.npy` (configurable subdir/pattern).
- Each response file is expected as `[n_neurons, length]` and is transposed to `[length, n_neurons]` for model input.
- With multiple sessions, use `data.channel_mode = "truncate_min"` and/or `data.time_mode = "truncate_min"` if shapes differ.

## Notes

- Edit config files under `scripts/configs/` to set input/output paths and runtime parameters.
- For neural autoencoder architecture sweeps, change values in `[model]` (including `architecture`) without code edits.
- For custom neural models, set `model.architecture = "target"`, then set `model.target` to an import path (`module.submodule.ClassName` or `module.submodule:ClassName`) and pass constructor args in `[model.kwargs]`.
- A ready-to-edit scaffold is available at `src/v1tovideo/neural_autoencoder/models/template_autoencoder.py`.
- The current image module resizes frames to `144x256` before encoding. This can be changed in config.
- `legacy/` preserves the original scripts.
- `assets/` preserves representative sample inputs and legacy outputs.
