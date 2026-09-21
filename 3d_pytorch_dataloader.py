"""
3d_pytorch_dataloader.py
========================
PyTorch `Dataset` and `DataLoader` for the 3D N-body simulation produced by
`simulation_3d.py` -> `3d_export_pipeline.py`.

Consumes a single `.npz` archive with the layout written by the pipeline:
    X           : (n_windows, W, N, 6)        float32, (x, y, z, vx, vy, vz)
    y           : (n_windows, N, 6)           float32, next state
    train_idx   : (n_train,)                  int64  , persisted window split
                                                     (IGNORED by get_dataloaders;
                                                      see the simulation-level
                                                      split rationale there)
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
    TEST_FRAC,
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
                 dtype: torch.dtype = torch.float32,
                 rollout_K: int = 0) -> None:

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
            # Store X/y as torch tensors in shared memory (2026-09-20).
            # With num_workers > 0 each worker process otherwise gets its
            # own copy: fork (Linux) copies them lazily via COW, but any
            # refcount touch materialises the full copy per worker; spawn
            # (Windows/Colab) pickles the ENTIRE array to every worker.
            # share_memory_() puts the payload in shared memory once so all
            # workers attach to the same pages -- an 8-worker run of a
            # ~1.5 GB N=100 dataset drops from ~12 GB RSS to ~1.5 GB.
            self._X = torch.from_numpy(self._X).share_memory_()
            self._y = torch.from_numpy(self._y).share_memory_()
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

        # ── Sidecar .json metadata (best-effort) ───────────────────────────
        self.meta = self._load_sidecar(npz_path, n_sims=(
            self._mass_per_sim.shape[0] if self._mass_per_sim is not None else 0
        ))

        # ── Per-window simulation ids ──────────────────────────────────────
        # Recovered once here and reused for two things: the per-window
        # mass lookup below, and the simulation-level train/val/test split
        # in `get_dataloaders`. A window-level split leaks with STRIDE=1
        # (adjacent windows share W-1 of their W frames, so near-duplicate
        # samples land in different splits), which is why the split is
        # done per simulation, not per window.
        _n_sims = (self._mass_per_sim.shape[0]
                   if self._mass_per_sim is not None
                   else self.meta.n_simulations)
        self._sim_ids = self._recover_sim_ids(npz_path, _n_sims, n_samples)

        # ── Per-window mass vector ──────────────────────────────────────────
        # The pipeline stores mass per simulation; index the per-sim table
        # by each window's simulation id so every window knows its own
        # mass vector. If `mass` is not in the .npz there is nothing to
        # assign (and `_recover_sim_ids` returned an empty array).
        if self._mass_per_sim is not None:
            _n_mass, _N = self._mass_per_sim.shape
            if _N != self.n_bodies:
                raise ValueError(
                    f"Mass array has N={_N} but dataset has N={self.n_bodies}."
                )
            self._mass_per_window = self._mass_per_sim[self._sim_ids]  # (n_samples, N)
        else:
            self._mass_per_window = None

        # ── K-step rollout targets (opt-in, 2026-09-21) ─────────────────────
        # The stability-trained variants train with a K-step autoregressive
        # rollout MSE (losses.rollout_mse_loss), which needs the TRUE future
        # frames for every rollout step. The (X, y) pair alone only carries
        # one frame ahead, but the windows are contiguous with STRIDE=1, so
        # the true frame for rollout step k of window i is y[i + k − 1]:
        #   window i = frames [i .. i+W-1], y[i] = frame i+W, and the sim's
        #   windows tile it contiguously — so step k's ground truth is
        #   simply the y of window i+k−1. Precomputed here once (a gather
        #   of `y`), with a validity mask: windows within K−1 of a
        #   simulation boundary have no in-sim frame for the later steps.
        # Off by default (rollout_K=0) so evaluate_models /
        # stability_benchmark and every existing consumer keep the plain
        # (x, y[, mass]) item contract.
        self.rollout_K = int(rollout_K)
        if self.rollout_K > 0:
            if self._sim_ids.size == 0:
                raise RuntimeError(
                    f"rollout_K={self.rollout_K} requires per-window "
                    f"simulation ids (mass table / sidecar), which could "
                    f"not be recovered from {npz_path!r}."
                )
            _K  = self.rollout_K
            _yr = torch.zeros((self.n_samples, _K, self.n_bodies, self.features),
                              dtype=torch.float32)
            _ok = torch.zeros((self.n_samples, _K), dtype=torch.bool)
            _sim = torch.from_numpy(self._sim_ids.astype(np.int64))
            _idx = torch.arange(self.n_samples)
            _y_all = self._y                     # (n, N, F), shared
            for _j in range(_K):
                # target of step j of window i is y[i + j], valid iff
                # i + j stays inside the same simulation. Sim-id lookup is
                # restricted to in-bounds windows first, so the gather
                # never indexes past the array end.
                _same = _idx + _j < self.n_samples
                if _j > 0:
                    _in = _idx[_same]                 # in-bounds candidates
                    _same[_in] = _sim[_in + _j] == _sim[_in]
                _yr[_same, _j] = _y_all[_idx[_same] + _j]
                _ok[_same, _j] = True
            self._y_roll    = _yr.share_memory_()
            self._roll_valid = _ok.share_memory_()
        else:
            self._y_roll    = None
            self._roll_valid = None

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
    def sim_ids(self) -> np.ndarray:
        """
        (n_samples,) simulation id of each window, or an empty array when
        the ids could not be recovered (no mass table and no sidecar).
        Ids are contiguous 0..n_simulations-1 because both recovery paths
        (`n_windows_per_sim` sidecar and the equal-size fallback) repeat
        `np.arange(n_sims)`; a skipped simulation (too few frames) simply
        has no windows and no id.
        """
        return self._sim_ids.copy()

    @property
    def n_simulations(self) -> int:
        """Number of distinct simulations the windows came from (0 if unknown)."""
        return 0 if self._sim_ids.size == 0 else int(self._sim_ids.max()) + 1

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
            if self.rollout_K > 0:
                # (K, N, F') rollout targets + validity mask (see __init__).
                yr = torch.as_tensor(np.array(self._y_roll[idx]),
                                     dtype=self.dtype)
                if not self.channel_mask.all():
                    yr = self.apply_channel_mask(yr)
                if self.model_type in ("mlp", "lstm"):
                    yr_out = yr.reshape(self.rollout_K, -1)
                else:  # gnn: (K, N, F')
                    yr_out = yr
                return x_out, y_out, m, yr_out, self._roll_valid[idx]
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
                    test_frac: float = TEST_FRAC,
                    split_seed: int = SPLIT_SEED,
                    include_mass: bool = False,
                    channel_mask: np.ndarray | None = None,
                    num_workers: int = 0,
                    pin_memory: bool = False,
                    drop_last: bool = False,
                    persistent_workers: bool = False,
                    rollout_K: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders.

    The split is SIMULATION-LEVEL: whole simulations are assigned to
    train/val/test, then all of a simulation's windows go to its split.
    The window-level split persisted inside the .npz (`train_idx` /
    `val_idx` / `test_idx`) is deliberately IGNORED: with STRIDE=1,
    adjacent windows share W-1 of their W frames, so scattering windows
    across splits leaks near-duplicate samples between train, val and
    test and reports artificially low val/test errors. The per-window
    simulation ids are recovered from the sidecar
    (`n_windows_per_sim`, written by `3d_export_pipeline`), so existing
    .npz files keep working without re-exporting. Only when the ids
    cannot be recovered (no mass table and no sidecar) does this fall
    back to a window-level random split, with a warning.

    Parameters
    ----------
    npz_path        : ML-ready .npz archive
    model_type      : "mlp" | "lstm" | "gnn"
    batch_size      : batch size for all loaders
    val_frac        : fraction of SIMULATIONS for validation
    test_frac       : fraction of SIMULATIONS for test
    split_seed      : RNG seed for the simulation assignment (must be fixed
                      for project; also seeds the window-level fallback)
    include_mass    : if True, items are (x, y, mass_per_window)
    channel_mask    : optional (6,) boolean mask of features to keep
    num_workers     : DataLoader worker count
    pin_memory      : pin host memory for CUDA transfers
    drop_last       : drop the last partial batch (train loader)
    persistent_workers : keep workers alive across epochs (set True with num_workers>0)
    rollout_K       : if > 0, precompute K-step rollout targets and yield
                      (x, y, mass, y_roll, roll_valid) items (see
                      NBody3DDataset); 0 keeps the plain (x, y[, mass])
                      contract for every existing consumer

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    full_dataset = NBody3DDataset(
        npz_path=npz_path,
        model_type=model_type,
        include_mass=include_mass,
        channel_mask=channel_mask,
        rollout_K=rollout_K,
    )

    # ── Simulation-level split (see docstring for why the persisted
    #    window-level split is ignored) ────────────────────────────────────
    sim_ids = full_dataset.sim_ids
    n_sims  = full_dataset.n_simulations
    if sim_ids.size and n_sims >= 3:
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(n_sims)
        n_test_sims = max(1, int(round(n_sims * test_frac)))
        n_val_sims  = max(1, int(round(n_sims * val_frac)))
        if n_test_sims + n_val_sims >= n_sims:
            # Tiny n_sims: never let val+test consume every simulation.
            n_test_sims = 1
            n_val_sims  = 1
        test_sims  = perm[:n_test_sims]
        val_sims   = perm[n_test_sims:n_test_sims + n_val_sims]
        train_sims = perm[n_test_sims + n_val_sims:]

        train_idx = np.flatnonzero(np.isin(sim_ids, train_sims))
        val_idx   = np.flatnonzero(np.isin(sim_ids, val_sims))
        test_idx  = np.flatnonzero(np.isin(sim_ids, test_sims))
        if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
            raise RuntimeError(
                f"Simulation-level split produced an empty subset "
                f"(train={train_idx.size}, val={val_idx.size}, "
                f"test={test_idx.size}); refusing to continue with a "
                f"degenerate split."
            )

        train_subset = Subset(full_dataset, train_idx.tolist())
        val_subset   = Subset(full_dataset, val_idx.tolist())
        test_subset  = Subset(full_dataset, test_idx.tolist())
        if full_dataset.train_indices is not None:
            print("  [warn] persisted window-level split ignored "
                  "(leaks near-duplicate windows across splits with STRIDE=1)")
        print(f"  simulation-level split: "
              f"train={len(train_sims)}/{n_sims} sims ({len(train_subset)} windows)  "
              f"val={len(val_sims)} sims ({len(val_subset)})  "
              f"test={len(test_sims)} sims ({len(test_subset)})  "
              f"(seed={split_seed})")
    else:
        # No window-level fallback (post-audit guard, 2026-09-20): a
        # random WINDOW-level split leaks near-duplicate samples across
        # splits with STRIDE=1 (adjacent windows share W-1 frames), which
        # silently invalidates val/test metrics. If the per-simulation ids
        # cannot be recovered, fail loudly instead of training on a leaky
        # split.
        if sim_ids.size:
            raise RuntimeError(
                f"Only {n_sims} simulations available; need >= 3 for a "
                f"simulation-level split. Generate more simulations."
            )
        raise RuntimeError(
            f"Cannot safely split {npz_path}: per-window simulation ids "
            f"could not be recovered (no mass table and no sidecar "
            f"'n_windows_per_sim'). A window-level random split leaks "
            f"near-duplicate samples across splits with STRIDE=1; "
            f"re-export the dataset with 3d_export_pipeline.py."
        )

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
