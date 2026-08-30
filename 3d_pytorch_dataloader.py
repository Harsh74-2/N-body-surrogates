"""
3d_pytorch_dataloader.py
========================
PyTorch `Dataset` and `DataLoader` for the 3D N-body simulation produced by
`simulation_3d.py` -> `3d_export_pipeline.py`.

Consumes a single `.npz` archive with the layout written by the pipeline:
    X           : (n_windows, W, N, 6)        float32, (x, y, z, vx, vy, vz)
    y           : (n_windows, N, 6)           float32, next state
    train_idx   : (n_train,)                  int64  , persisted split
    val_idx     : (n_val,)                    int64
    test_idx    : (n_test,)                   int64
    mass        : (n_sims, N)                 float64, optional, per-sim masses
    meta        : (n_sims, 4)                 float64, optional, [dt, eps, G, seed]

All model types receive the same shaped tensors; the DataLoader does *not*
flatten or reshape samples. Each batch returned by `get_dataloaders` is:
    x : (B, W, N, F) : input window of per-body states
    y : (B, N, F)    : next per-body state
    mass : (B, N)    : per-window per-body mass
where F is the number of kept feature channels (6 by default). The three
surrogates all expect this layout and reshape internally if needed, so
the dataloader is model-agnostic.

The dataset also exposes:
    - the per-simulation mass array (concatenated across windows so each
      window knows its own mass vector) when `include_mass=True`.
    - the channel mask so training scripts can drop, e.g., velocities.
    - a `meta` dict with window/horizon/stride/normalize/seed.

Usage
-----
    from 3d_pytorch_dataloader import get_dataloaders
    train, val, test = get_dataloaders(
        npz_path="ml_ready_data/dataset_3d_w5h1s1r.npz",
        model_type="gnn",
        batch_size=32,
    )
    for x, y in train:
        ...
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from utils import configure_utf8_stdout

configure_utf8_stdout()

from torch.utils.data import DataLoader, Dataset, Subset

from pipeline_config import (
    DEFAULT_NPZ,
    FEATURE_DIM,
    HORIZON,
    ModelType,
    SPLIT_SEED,
    STRIDE,
    VAL_FRAC,
    WINDOW_SIZE,
)


# ── Defaults ─────────────────────────────────────────────────────────────────
VALID_MODELS = tuple(ModelType.values())


# ── Container for everything a training loop needs to log ────────────────────
@dataclass(frozen=True)
class DatasetMeta:
    """Sidecar metadata extracted from the .npz (and matching .json if any)."""
    n_simulations: int
    n_windows:     int
    n_train:       int
    n_val:         int
    n_test:        int
    window_size:   int
    horizon:       int
    stride:        int
    normalize:     bool
    split_seed:    int
    has_mass:      bool
    feature_dim:   int
    has_persisted_split: bool
    raw_files:     tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "n_windows":     self.n_windows,
            "n_train":       self.n_train,
            "n_val":         self.n_val,
            "n_test":        self.n_test,
            "window_size":   self.window_size,
            "horizon":       self.horizon,
            "stride":        self.stride,
            "normalize":     self.normalize,
            "split_seed":    self.split_seed,
            "has_mass":      self.has_mass,
            "feature_dim":   self.feature_dim,
            "has_persisted_split": self.has_persisted_split,
            "raw_files":     list(self.raw_files),
        }


# ── Dataset ──────────────────────────────────────────────────────────────────
class NBody3DDataset(Dataset):
    """
    Wraps the ML-ready `.npz` produced by `3d_export_pipeline.py`.

    The dataset itself does *not* perform train/val splitting, the split
    is applied externally via `Subset`. We do, however, pre-compute a
    length and a per-window mass tensor (so the GNN models can optionally
    condition on mass without re-indexing the .npz on every
    `__getitem__`).

    Memory model
    ------------
    The `.npz` is opened lazily with `mmap_mode="r"` and per-window
    samples are copied into contiguous `float32` tensors on demand. This
    keeps resident memory at O(1) in the number of windows, which matters
    when scaling `num_simulations` and `frames`.

    Parameters
    ----------
    npz_path       : path to the .npz archive
    model_type     : "mlp" | "lstm" | "gnn"
    include_mass   : if True, each `__getitem__` returns (mass_per_window,)
                     as a third element alongside (x, y).
    channel_mask   : optional boolean / 0-1 mask of shape (6,) selecting
                     which of (x, y, z, vx, vy, vz) to keep. Saves a
                     little RAM and a lot of compute when, e.g., you
                     only want positions.
    dtype          : storage dtype for the in-RAM copies (default float32)
    """

    def __init__(self,
                 npz_path: str,
                 model_type: str = "mlp",
                 include_mass: bool = False,
                 channel_mask: np.ndarray | None = None,
                 dtype: torch.dtype = torch.float32) -> None:

        if not Path(npz_path).exists():
            raise FileNotFoundError(
                f"Cannot find dataset at {npz_path!r}. "
                f"Run `python 3d_export_pipeline.py` first to produce it."
            )

        model_type = model_type.lower()
        if model_type not in VALID_MODELS:
            raise ValueError(
                f"model_type must be one of {VALID_MODELS}, got {model_type!r}."
            )

        # ── Load everything (mmap'd; the OS will page in as needed) ────────
        with np.load(npz_path, mmap_mode="r") as data:
            files = set(data.files)
            if "X" not in files or "y" not in files:
                raise KeyError(
                    f"{npz_path!r} does not contain the required 'X' / 'y' "
                    f"arrays. Found keys: {sorted(files)}."
                )

            # Load the full arrays into RAM once as float32. The .npz is
            # DEFLATE-compressed, so a per-__getitem__ re-open (np.load +
            # arch["X"][idx]) decompresses the whole X on *every* access;
            # those large transient buffers are not returned to the OS fast
            # enough and accumulate ~O(X_size) per call, on batch=512 that
            # reaches ~15 GB and gets OOM-killed. An in-RAM copy is small
            # (≤~1.5 GB even at N=100) and indexed in O(1) per item. The
            # single dataset instance is shared by the train/val/test
            # Subsets, so this is a one-time cost.
            self._X = np.array(data["X"], dtype=np.float32)
            self._y = np.array(data["y"], dtype=np.float32)
            X = self._X
            y = self._y
            self._X_shape = tuple(X.shape)
            self._y_shape = tuple(y.shape)

            # Persisted train/val/test split
            self._train_idx = (data["train_idx"].copy()
                               if "train_idx" in files else None)
            self._val_idx   = (data["val_idx"].copy()
                               if "val_idx"   in files else None)
            self._test_idx  = (data["test_idx"].copy()
                               if "test_idx"  in files else None)

            # Per-simulation mass (shape (n_sims, N)): expand to a per-window
            # mass vector so a window knows which simulation it came from.
            self._mass_per_sim = (data["mass"].astype(np.float32)
                                  if "mass" in files else None)

            # Per-simulation meta: [dt, epsilon, G, seed]
            self._meta_per_sim = (data["meta"].astype(np.float64)
                                  if "meta" in files else None)

        # ── Shape / sanity checks ───────────────────────────────────────────
        n_samples, window, N, features = self._X_shape
        if features != FEATURE_DIM:
            raise ValueError(
                f"Dataset features must be {FEATURE_DIM} for 3D data, "
                f"got {features}."
            )
        if self._y_shape != (n_samples, N, FEATURE_DIM):
            raise ValueError(
                f"'y' shape {self._y_shape} inconsistent with 'X' shape "
                f"{self._X_shape} (expected (n, N, 6))."
            )

        self.n_samples = int(n_samples)
        self.window    = int(window)
        self.n_bodies  = int(N)
        self.features  = int(features)
        self.model_type = model_type
        self.include_mass = bool(include_mass)
        self.dtype       = dtype

        # ── Channel mask (optional) ─────────────────────────────────────────
        if channel_mask is None:
            self.channel_mask = np.ones(features, dtype=bool)
        else:
            cm = np.asarray(channel_mask, dtype=bool)
            if cm.shape != (features,):
                raise ValueError(
                    f"channel_mask must have shape ({features},), got {cm.shape}."
                )
            if not cm.any():
                raise ValueError("channel_mask selects zero channels.")
            self.channel_mask = cm
        self.kept_features = int(self.channel_mask.sum())

        # ── Store the source path (used by _open_archive and __repr__) ─────
        self.npz_path = npz_path

        # ── Per-window mass vector ──────────────────────────────────────────
        # The pipeline stores mass per simulation; concatenate windows for
        # each simulation in order so we can index by absolute window idx.
        # If `mass` is not in the .npz we skip this whole step, there is
        # nothing to assign, and `_recover_sim_ids` would otherwise divide
        # by zero when called with n_sims=0.
        if self._mass_per_sim is not None:
            n_sims, _N = self._mass_per_sim.shape
            if _N != self.n_bodies:
                raise ValueError(
                    f"Mass array has N={_N} but dataset has N={self.n_bodies}."
                )
            # The pipeline appends `n_windows_per_sim` per simulation in
            # order. We don't have that list here, but we can recover
            # simulation ids if the sidecar .json is present, or fall back
            # to assuming all sims contributed equal windows (the default
            # in the pipeline is exactly that).
            sim_ids = self._recover_sim_ids(npz_path, n_sims, n_samples)
            self._mass_per_window = self._mass_per_sim[sim_ids]   # (n_samples, N)
        else:
            self._mass_per_window = None

        # ── Sidecar .json metadata (best-effort) ───────────────────────────
        self.meta = self._load_sidecar(npz_path, n_sims=(
            self._mass_per_sim.shape[0] if self._mass_per_sim is not None else 0
        ))

        # ── Reporting ───────────────────────────────────────────────────────
        print(f"[NBody3DDataset] {Path(npz_path).name}")
        print(f"  X shape       : {self._X_shape}  -> [Samples, Window, N, {FEATURE_DIM}]")
        print(f"  y shape       : {self._y_shape}  -> [Samples, N, {FEATURE_DIM}]")
        print(f"  model_type    : {self.model_type}")
        print(f"  include_mass  : {self.include_mass}  "
              f"(per-window mass available: {self._mass_per_window is not None})")
        print(f"  channel_mask  : {self.channel_mask.astype(int).tolist()}  "
              f"({self.kept_features} kept)")
        if self._train_idx is not None:
            print(f"  persisted split: train={self._train_idx.shape[0]}  "
                  f"val={self._val_idx.shape[0]}  "
                  f"test={self._test_idx.shape[0] if self._test_idx is not None else 0}")

    # ── Sidecar helpers ─────────────────────────────────────────────────────
    def _load_sidecar(self, npz_path: str, n_sims: int) -> DatasetMeta:
        """Best-effort load of the .json metadata written by the pipeline."""
        sidecar = str(Path(npz_path).with_suffix(".json"))
        if not Path(sidecar).exists():
            return DatasetMeta(
                n_simulations=n_sims,
                n_windows=self.n_samples,
                n_train=self._train_idx.shape[0] if self._train_idx is not None else 0,
                n_val=self._val_idx.shape[0]     if self._val_idx   is not None else 0,
                n_test=self._test_idx.shape[0]  if self._test_idx  is not None else 0,
                window_size=self.window,
                horizon=HORIZON, stride=STRIDE, normalize=False, split_seed=SPLIT_SEED,
                has_mass=self._mass_per_sim is not None,
                feature_dim=self.features,
                has_persisted_split=self._train_idx is not None,
                raw_files=(),
            )
        with open(sidecar, "r", encoding="utf-8") as f:
            j = json.load(f)
        return DatasetMeta(
            n_simulations=int(j.get("n_simulations", n_sims)),
            n_windows=int(j.get("n_windows", self.n_samples)),
            n_train=int(j.get("n_train",
                              self._train_idx.shape[0] if self._train_idx is not None else 0)),
            n_val=int(j.get("n_val",
                            self._val_idx.shape[0]   if self._val_idx   is not None else 0)),
            n_test=int(j.get("n_test",
                             self._test_idx.shape[0]  if self._test_idx  is not None else 0)),
            window_size=int(j.get("window_size", self.window)),
            horizon=int(j.get("horizon", HORIZON)),
            stride=int(j.get("stride", STRIDE)),
            normalize=bool(j.get("normalize", False)),
            split_seed=int(j.get("split_seed", SPLIT_SEED)),
            has_mass=bool(j.get("has_mass", self._mass_per_sim is not None)),
            feature_dim=self.features,
            has_persisted_split=self._train_idx is not None,
            raw_files=tuple(j.get("raw_files", ())),
        )

    def _recover_sim_ids(self, npz_path: str, n_sims: int, n_windows: int) -> np.ndarray:
        """
        Recover (n_windows,) array of per-window simulation ids.

        Order: read `n_windows_per_sim` from the sidecar .json if present;
        fall back to equal-sized sims (matches the pipeline default).

        Defensive guard: if `n_sims <= 0` there is nothing meaningful to
        recover (the dataset was loaded without a `mass` array), return
        an empty array rather than dividing by zero.
        """
        if n_sims <= 0:
            return np.empty(0, dtype=np.int64)
        sidecar = str(Path(npz_path).with_suffix(".json"))
        if Path(sidecar).exists():
            with contextlib.suppress(OSError, ValueError, KeyError):
                with open(sidecar, "r", encoding="utf-8") as f:
                    j = json.load(f)
                per_sim = j.get("n_windows_per_sim")
                if per_sim and len(per_sim) == n_sims and sum(per_sim) == n_windows:
                    return np.repeat(np.arange(n_sims), per_sim)
        # Fall back: equal-sized simulations with remainder distributed to the
        # first simulations so every window is assigned.
        base, rem = divmod(n_windows, n_sims)
        counts = np.array([base + (1 if i < rem else 0) for i in range(n_sims)])
        return np.repeat(np.arange(n_sims), counts)

    # ── Standard dunder methods ─────────────────────────────────────────────
    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"NBody3DDataset(path='{Path(getattr(self, 'npz_path', '?')).name}', "
            f"n={self.n_samples}, W={self.window}, N={self.n_bodies}, "
            f"F={self.kept_features}/{FEATURE_DIM}, model='{self.model_type}', "
            f"include_mass={self.include_mass})"
        )

    @property
    def train_indices(self) -> np.ndarray | None:
        return None if self._train_idx is None else self._train_idx.copy()

    @property
    def val_indices(self) -> np.ndarray | None:
        return None if self._val_idx is None else self._val_idx.copy()

    @property
    def test_indices(self) -> np.ndarray | None:
        return None if self._test_idx is None else self._test_idx.copy()

    def apply_channel_mask(self, arr: np.ndarray) -> np.ndarray:
        """Apply the configured channel mask to a (..., F) array."""
        return arr[..., self.channel_mask]

    # ── Item fetch ──────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        # Index the in-RAM arrays loaded once in __init__. This avoids
        # re-opening/re-decompressing the .npz on every access (which leaked
        # ~O(X_size) per call and OOM-killed the process on large batches).
        x_window = np.array(self._X[idx])   # (W, N, 6), small copy
        y_target = np.array(self._y[idx])   # (N, 6)

        if not self.channel_mask.all():
            x_window = self.apply_channel_mask(x_window)
            y_target = self.apply_channel_mask(y_target)

        # → torch tensors
        x_t = torch.as_tensor(x_window, dtype=self.dtype)
        y_t = torch.as_tensor(y_target, dtype=self.dtype)

        # Reshape per model_type
        if self.model_type == "mlp":
            x_out = x_t.reshape(-1)        # (W * N * F')
            y_out = y_t.reshape(-1)        # (N * F')
        elif self.model_type == "lstm":
            # (W, N*F') and (N*F')
            x_out = x_t.reshape(self.window, -1)
            y_out = y_t.reshape(-1)
        else:  # "gnn"
            x_out = x_t                    # (W, N, F')
            y_out = y_t                    # (N,  F')

        if self.include_mass and self._mass_per_window is not None:
            m = torch.as_tensor(self._mass_per_window[idx], dtype=self.dtype)
            return x_out, y_out, m

        return x_out, y_out

    # `npz_path` is set as a regular attribute in `__init__`; nothing
    # else in this file (or its callers) needs an explicit getter/setter.
    # Each `__getitem__` opens the .npz with `mmap_mode="r"` directly so
    # we never hold a persistent NpzFile handle.


# ── Loaders ──────────────────────────────────────────────────────────────────
def get_dataloaders(npz_path: str,
                    model_type: str = "mlp",
                    batch_size: int = 32,
                    val_frac: float = VAL_FRAC,
                    split_seed: int = SPLIT_SEED,
                    include_mass: bool = False,
                    channel_mask: np.ndarray | None = None,
                    num_workers: int = 0,
                    pin_memory: bool = False,
                    drop_last: bool = False,
                    persistent_workers: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    The split is taken from `train_idx` / `val_idx` / `test_idx` inside the
    .npz when present (the default output of the refined pipeline). When
    absent, a fresh deterministic random split is generated with
    `split_seed` (and *replaces* the in-memory split, but is not
    persisted back to disk: re-run `3d_export_pipeline.py` to save it).

    Parameters
    ----------
    npz_path        : ML-ready .npz archive
    model_type      : "mlp" | "lstm" | "gnn"
    batch_size      : batch size for all loaders
    val_frac        : validation fraction (only used when no persisted split)
    split_seed      : RNG seed for fallback split (must be fixed for project)
    include_mass    : if True, items are (x, y, mass_per_window)
    channel_mask    : optional (6,) boolean mask of features to keep
    num_workers     : DataLoader worker count
    pin_memory      : pin host memory for CUDA transfers
    drop_last       : drop the last partial batch (train loader)
    persistent_workers : keep workers alive across epochs (set True with num_workers>0)

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    full_dataset = NBody3DDataset(
        npz_path=npz_path,
        model_type=model_type,
        include_mass=include_mass,
        channel_mask=channel_mask,
    )

    # ── Use the persisted split if it exists ────────────────────────────────
    if (full_dataset.train_indices is not None and
        full_dataset.val_indices is not None and
        full_dataset.test_indices is not None):
        train_subset = Subset(full_dataset, full_dataset.train_indices.tolist())
        val_subset   = Subset(full_dataset, full_dataset.val_indices.tolist())
        test_subset  = Subset(full_dataset, full_dataset.test_indices.tolist())
        print(f"  using persisted split: "
              f"train={len(train_subset)}  val={len(val_subset)}  "
              f"test={len(test_subset)}  (seed={full_dataset.meta.split_seed})")
    else:
        # Fall back to a deterministic random split.
        n_val   = int(round(len(full_dataset) * val_frac))
        n_test  = int(round(len(full_dataset) * 0.1))
        n_train = len(full_dataset) - n_val - n_test
        g = torch.Generator().manual_seed(split_seed)
        train_subset, val_subset, test_subset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val, n_test], generator=g,
        )
        print(f"  no persisted split, generated one in-memory: "
              f"train={n_train}  val={n_val}  test={n_test}  (seed={split_seed})")

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    train_loader = DataLoader(train_subset, shuffle=True,  drop_last=drop_last, **common)
    val_loader   = DataLoader(val_subset,   shuffle=False, drop_last=False,    **common)
    test_loader  = DataLoader(test_subset,  shuffle=False, drop_last=False,    **common)

    print(f"[{model_type.upper()}] DataLoaders ready: "
          f"{len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches.")
    return train_loader, val_loader, test_loader


# ── CLI smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Smoke-test the 3D N-body DataLoaders.",
    )
    p.add_argument("--npz",  default=DEFAULT_NPZ)
    p.add_argument("--model", default="gnn",
                   choices=list(VALID_MODELS))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--include-mass", action="store_true")
    p.add_argument("--channel-mask", default=None,
                   help="Comma-separated 0/1 list of length 6, e.g. '1,1,1,0,0,0'")
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = p.parse_args()

    cm = None
    if args.channel_mask:
        cm = np.array([int(x) for x in args.channel_mask.split(",")], dtype=bool)
        if cm.shape != (FEATURE_DIM,):
            raise SystemExit(f"--channel-mask must have {FEATURE_DIM} entries")

    train, val, test = get_dataloaders(
        npz_path=args.npz,
        model_type=args.model,
        batch_size=args.batch_size,
        include_mass=args.include_mass,
        channel_mask=cm,
    )
    for name, loader in [("train", train), ("val", val), ("test", test)]:
        batch = next(iter(loader))
        if args.include_mass:
            x, y, m = batch
            print(f"  [{name}] X: {tuple(x.shape)}  y: {tuple(y.shape)}  mass: {tuple(m.shape)}")
        else:
            x, y = batch
            print(f"  [{name}] X: {tuple(x.shape)}  y: {tuple(y.shape)}")
