"""Device selection and portable checkpoints for the basic TrajGPT workflow."""

from __future__ import annotations

import os
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 1
TIME_CONFIG = {
    'time_unit': 'months', 'history_time_encoding': 'temporal_rope_headwise_v1',
    'target_time_encoding': 'log1p_delta_add_v1', 'sos_time_policy': 'first_event',
    'pretraining_objective': 'time_conditioned_next_event_v1',
}


def _checkpoint_time_config(config, model=None, require_metadata=False):
    """Keep old event-order weights distinct from newly trained temporal models."""
    result = dict(config)
    mode = result.get('time_mode', 'event_order')
    if mode not in ('event_order', 'time_conditioned'):
        raise ValueError('Unknown checkpoint time_mode')
    if model is not None and mode != getattr(model, 'time_mode', 'event_order'):
        raise ValueError('Checkpoint time_mode must match the model being saved')
    result['time_mode'] = mode
    if mode == 'time_conditioned':
        for key, value in TIME_CONFIG.items():
            if require_metadata and key not in result:
                raise ValueError(f'Time-conditioned checkpoint is missing {key}')
            if result.get(key, value) != value:
                raise ValueError(f'Unsupported checkpoint {key}: {result[key]}')
            result[key] = value
        if require_metadata and 'time_scale' not in result:
            raise ValueError('Time-conditioned checkpoint is missing time_scale')
        scale = float(result.get('time_scale', 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('Checkpoint time_scale must be finite and positive')
        if model is not None and scale != getattr(model, 'time_scale', 1.0):
            raise ValueError('Checkpoint time_scale must match the model being saved')
        result['time_scale'] = scale
    return result


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def _check_device(device: torch.device) -> None:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no usable CUDA device is available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable.")
    elif device.type == "mps":
        if not _mps_available():
            raise RuntimeError("MPS was requested, but Apple MPS is unavailable.")
    elif device.type != "cpu":
        raise ValueError("Supported devices are auto, cpu, cuda[:index], and mps.")
    # Availability alone may not guarantee that the process can allocate memory.
    torch.empty(1, device=device)


def select_device(requested: str | torch.device = "auto") -> torch.device:
    """Choose CUDA, then MPS, then CPU; explicit requests never silently change."""
    if str(requested).lower() != "auto":
        device = torch.device(requested)
        _check_device(device)
        return device
    candidates = []
    if torch.cuda.is_available():
        candidates.append(torch.device("cuda"))
    if _mps_available():
        candidates.append(torch.device("mps"))
    candidates.append(torch.device("cpu"))
    for device in candidates:
        try:
            _check_device(device)
            return device
        except RuntimeError as exc:
            if device.type == "cpu":
                raise
            warnings.warn(f"Cannot initialize {device}; trying the next device: {exc}")
    raise RuntimeError("No supported execution device is available.")


def seed_everything(seed: int) -> None:
    """Seed the random generators used by training and patient sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _mps_available():
        torch.mps.manual_seed(seed)


def capture_rng_state() -> dict:
    """Represent NumPy state with basic types for torch's safe weights loader."""
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": keys.tolist(),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
    if _mps_available():
        state["mps"] = torch.mps.get_rng_state().cpu().clone()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore CPU generators and any accelerator generators present locally."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        for index, value in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(), device=index)
    if _mps_available() and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def save_checkpoint(
    path,
    model,
    optimizer,
    config: dict,
    split_ids: dict,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    extra: dict | None = None,
) -> None:
    """Atomically save a self-contained basic-workflow checkpoint on CPU."""
    if getattr(model, "head_type", None) != "pretrain":
        raise ValueError("Basic-workflow checkpoints require the trained pretrain head.")
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model_state_dict": _cpu_copy(model.state_dict()),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()) if optimizer is not None else None,
        "config": _checkpoint_time_config(config, model=model),
        "vocabulary": [int(code) for code in model.token_embedding.phecodes],
        "split_ids": _cpu_copy(split_ids),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "rng_state": capture_rng_state(),
        "extra": _cpu_copy(extra) if extra is not None else {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, device: str | torch.device = "cpu"):
    """Return (model, payload), without requiring the original embedding CSV."""
    # Do not fall back to arbitrary pickle loading for incompatible old files.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Unsupported checkpoint format. Use a checkpoint produced by the repaired "
            "basic training workflow; older checkpoints are not migrated automatically."
        )
    required = {
        "model_state_dict", "optimizer_state_dict", "config", "vocabulary", "split_ids",
        "epoch", "global_step", "best_val_loss", "rng_state",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    config = payload["config"]
    vocabulary = payload["vocabulary"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary.")
    config = _checkpoint_time_config(config, require_metadata=True)
    payload['config'] = config
    if (not isinstance(vocabulary, list) or not vocabulary
            or any(type(code) is not int or code <= 0 for code in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError("Checkpoint vocabulary must contain unique positive integer PheCodes.")
    weights = payload["model_state_dict"].get("token_embedding.emb.weight")
    if (not isinstance(weights, torch.Tensor) or weights.ndim != 2
            or weights.shape != (len(vocabulary) + 1, 200)
            or config.get("d_model") != 200):
        raise ValueError("Checkpoint vocabulary and fixed 200-dimensional embeddings do not agree.")
    if not torch.isfinite(weights).all():
        raise ValueError("Checkpoint embeddings contain non-finite values.")
    from models.TrajGPT import TrajGPT

    model = TrajGPT(
        SimpleNamespace(**config),
        head_type="pretrain",
        embedding_data={"phecodes": vocabulary, "embeddings": weights[:-1]},
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(select_device(device))
    return model, payload
