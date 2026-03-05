from __future__ import annotations

from importlib import import_module

from torch import nn


def build_model_from_target(target: str, kwargs: dict[str, object] | None = None) -> nn.Module:
    """Load and instantiate a model class from `module.submodule.ClassName` or `module:ClassName`."""
    spec = target.strip()
    if not spec:
        raise ValueError("model.target must be a non-empty import path")

    if ":" in spec:
        module_name, class_name = spec.split(":", maxsplit=1)
    elif "." in spec:
        module_name, class_name = spec.rsplit(".", maxsplit=1)
    else:
        raise ValueError(
            "Invalid model.target format. Use `module.submodule.ClassName` or `module.submodule:ClassName`."
        )

    module = import_module(module_name)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ValueError(f"Could not find class '{class_name}' in module '{module_name}'")
    if not isinstance(model_class, type):
        raise ValueError(f"Target '{target}' does not resolve to a class")

    model = model_class(**(kwargs or {}))
    if not isinstance(model, nn.Module):
        raise ValueError(f"Instantiated object from '{target}' is not a torch.nn.Module")
    return model
