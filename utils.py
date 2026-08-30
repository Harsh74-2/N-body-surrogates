"""
utils.py
========
Shared utilities used across the 3D N-body surrogate pipeline.

These helpers remove the repetitive boilerplate that every top-level
script was carrying (UTF-8 stdout setup, importlib loading of sibling
modules, device selection, Colab Drive mounting, quick-mode dataloader
capping, and loss-curve plotting).
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import time
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import torch.nn as nn


def configure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows cp1252 consoles."""
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def pick_device() -> torch.device:
    """CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sibling_module(name: str, filename: str, anchor: str | None = None):
    """
    Load a sibling .py file by path and register it in sys.modules.

    Used because several module filenames start with digits, which makes
    normal imports a syntax error.
    """
    if anchor is None:
        anchor = __file__
    path = Path(anchor).with_name(filename)
    if not path.is_file():
        raise FileNotFoundError(f"missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mount_drive_if_possible() -> str:
    """
    Mount Google Drive when running inside the Colab IPython kernel;
    otherwise return the local repo root.
    """
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        local_root = Path(__file__).resolve().parent
        print(f"[drive] google.colab not available, using local repo: {local_root}")
        return str(local_root)

    try:
        from IPython import get_ipython
    except ImportError:
        get_ipython = None  # type: ignore

    if get_ipython is None or get_ipython() is None:
        sys.exit(
            "[drive] google.colab is installed but this script is not running\n"
            "[drive] inside the Colab IPython kernel, drive.mount() requires a\n"
            "[drive] live kernel. Use `%run script.py` instead of `!python script.py`."
        )

    drive.mount("/content/drive", force_remount=False)
    drive_root = "/content/drive/MyDrive"
    print(f"[drive] mounted at {drive_root}")
    return drive_root


def cap_dataloader(loader: DataLoader, n: int) -> DataLoader:
    """Return a DataLoader containing only the first `n` batches of `loader`.

    The capped loader is rebuilt from individual samples (not batches) so
    the output tensors keep the same shape as the original loader.
    """
    batches = list(islice(iter(loader), n))
    if not batches:
        return DataLoader([], batch_size=loader.batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

    samples: list[tuple] = []
    for batch in batches:
        if len(batch) == 3:
            x, y, m = batch
            for i in range(x.size(0)):
                samples.append((x[i], y[i], m[i]))
        else:
            x, y = batch
            for i in range(x.size(0)):
                samples.append((x[i], y[i]))

    return DataLoader(samples, batch_size=loader.batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


def timestamp() -> str:
    """Compact current timestamp for run directories."""
    return time.strftime("%Y%m%d-%H%M%S")


def save_loss_curve(history: list[dict],
                    out_path: str | Path,
                    title: str = "Training loss",
                    keys: tuple[str, str] = ("train_mse", "val_mse")) -> None:
    """Save a simple log-y MSE/loss curve from a trainer history list."""
    if not history:
        return
    matplotlib.use("Agg")
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in zip(keys, ["train", "val"]):
        if key in history[0]:
            ax.semilogy(epochs, [h[key] for h in history], label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


