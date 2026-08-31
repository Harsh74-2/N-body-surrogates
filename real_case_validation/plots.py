"""
plots.py
========
Per-preset and cross-preset visualisations for the real-case validation
report. Mirrors `validation.py`'s dark theme so the new dashboard sits
alongside the existing one with consistent styling.

Three output types:
  * `plot_trajectory`, xy projection of {reference, GNN, LSTM, MLP} on
    the same axes for one preset.
  * `plot_energy`    , |E(t) − E(0)| / |E(0)| per integrator.
  * `plot_dashboard` , one row per preset, two columns (trajectory,
    energy), the cross-preset summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_sibling_module

# Reuse the dark theme from `validation.py` for visual consistency.
_val_mod = load_sibling_module("nbody_validation", "validation.py")
THEME = _val_mod.THEME

# Each of the 6 variants gets its OWN colour so a stranger can read
# the difference at a glance — the previous map conflated stable and
# non-stable variants (GNN ≡ GNN_stable, etc.) which made the chart
# unreadable on dark backgrounds. The non-stable variants
# use saturated fills; the stable variants use a darker fill + bright
# edge so the family is still distinguishable but the *variant* is
# unambiguous.
SURROGATE_COLORS = {
    "GNN":         ("#7eb8f7", "#7eb8f7"),  # blue         (light)
    "GNN_stable":  ("#3955a0", "#7eb8f7"),  # dark blue + light edge
    "LSTM":        ("#7ef7a0", "#7ef7a0"),  # green        (light)
    "LSTM_stable": ("#2f7a45", "#7ef7a0"),  # dark green + light edge
    "MLP":         ("#b37ef7", "#b37ef7"),  # violet       (light)
    "MLP_stable":  ("#5a3d8a", "#b37ef7"),  # dark violet + light edge
}


def _surr_color(model_name: str) -> str:
    """Plot-fill colour for `model_name` (the saturated fill)."""
    entry = SURROGATE_COLORS.get(model_name)
    if entry is None:
        return THEME["warn"]
    return entry[0]


def _surr_edge(model_name: str) -> str:
    """Plot-edge colour for `model_name` (the bright outline)."""
    entry = SURROGATE_COLORS.get(model_name)
    if entry is None:
        return THEME["warn"]
    return entry[1]


# Suppress the per-variant linestyle: we use a single visual convention
# instead — dashed = reference ("from the books"), solid = predicted by
# the surrogate. The architecture is still distinguishable by colour and
# marker shape (see SURROGATE_MARKER), so the per-model linestyle
# mapping is reduced to a single entry. Keeping the dict + helper so
# call sites that read `_surr_dash(model_name)` still work, but every
# variant now returns the same solid line.
SURROGATE_DASH = {
    "GNN":         "-",   # solid
    "GNN_stable":  "-",
    "LSTM":        "-",
    "LSTM_stable": "-",
    "MLP":         "-",
    "MLP_stable":  "-",
}


# Reference (book) orbit linestyle — same in every plot so the reader
# can rely on the convention "dashed = from the books".
REFERENCE_DASH = "--"   # dashed


# Order in which model variants are sorted into the legend. Keeping
# this stable makes the legend deterministic across reruns.
SURROGATE_VARIANT_ORDER = (
    "GNN", "GNN_stable", "LSTM", "LSTM_stable", "MLP", "MLP_stable",
)


# Distinct marker shape per variant — colour alone is not enough on a
# dark background, especially when many models overlap. Each shape
# also gets a distinct edge colour so the family stays consistent with
# the SURROGATE_COLORS map.
SURROGATE_MARKER = {
    "GNN":         "o",
    "GNN_stable":  "s",
    "LSTM":        "^",
    "LSTM_stable": "D",
    "MLP":         "v",
    "MLP_stable":  "P",
}


# Per-body colour palette — used by `plot_trajectory` so each body
# has a unique, nameable colour rather than every reference being
# white. The palette is rotation-stable (same body always gets the
# same colour, regardless of preset) and dark-theme friendly.
BODY_COLORS = [
    "#ffffff",  # 0 Sun    (white — primary anchor)
    "#ffb86c",  # 1 mercury (orange)
    "#f1fa8c",  # 2 venus   (yellow)
    "#50fa7b",  # 3 earth   (green)
    "#ff79c6",  # 4 mars    (pink)
    "#8be9fd",  # 5 jupiter (cyan)
    "#bd93f9",  # 6 saturn  (purple)
    "#ff5555",  # 7 uranus  (red)
    "#5af78e",  # 8 neptune (mint)
    "#ffb86c",  # 9 pluto   (orange again, never used twice in a preset)
    "#f8f8f2",  # 10 moon  (off-white)
    "#ffb86c",  # 11 io    (orange)
    "#f1fa8c",  # 12 europa (yellow)
    "#50fa7b",  # 13 ganymede (green)
    "#ff79c6",  # 14 callisto (pink)
]


def _body_color(body_name: str, body_i: int) -> str:
    """Return a stable colour for `body_name`. Falls back to the
    palette-indexed colour if the name is not in the named map."""
    # Named map (preferred — gives the same colour for the same body
    # across presets).
    _NAMED = {
        "Sun":       "#ffffff",
        "Mercury":   "#ffb86c",
        "Venus":     "#f1fa8c",
        "Earth":     "#50fa7b",
        "Mars":      "#ff79c6",
        "Jupiter":   "#8be9fd",
        "Saturn":    "#bd93f9",
        "Uranus":    "#ff5555",
        "Neptune":   "#5af78e",
        "Pluto":     "#ffb86c",
        "Moon":      "#f8f8f2",
        "Io":        "#ffb86c",
        "Europa":    "#f1fa8c",
        "Ganymede":  "#50fa7b",
        "Callisto":  "#ff79c6",
    }
    return _NAMED.get(body_name, BODY_COLORS[body_i % len(BODY_COLORS)])


def _add_ensemble_color(name: str) -> None:
    """Add an ensemble_<...> row to SURROGATE_COLORS / DASH / MARKER if
    missing. Ensemble rows reuse a deterministic palette based on the
    alphabetical sort of the constituent names, so re-runs give the
    same colour.
    """
    if name in SURROGATE_COLORS:
        return
    # Deterministic but distinct from the 6 base variants.
    base_palette = [
        ("#ff7f50", "#ff7f50"),  # coral
        ("#ffd166", "#ffd166"),  # gold
        ("#06d6a0", "#06d6a0"),  # teal
        ("#f4a261", "#f4a261"),  # orange
    ]
    idx = (sum(ord(c) for c in name) >> 4) % len(base_palette)
    SURROGATE_COLORS[name] = base_palette[idx]
    SURROGATE_DASH[name] = "-"   # all surrogates are solid; legend convention
    SURROGATE_MARKER[name] = "X"                # filled x


def _surr_marker(model_name: str) -> str:
    """Plot marker shape for `model_name`."""
    return SURROGATE_MARKER.get(model_name, "o")


def _surr_dash(model_name: str):
    """Linestyle for `model_name`. Returns the entry in SURROGATE_DASH
    or a plain solid line as a safe fallback."""
    return SURROGATE_DASH.get(model_name, "-")


# Default render DPI for every savefig in this module. Was 140; bumped
# to 220 so the figures stay crisp when embedded in the report PDF or
# scaled in a Word document.
DPI = 220


def _variant_label(model_name: str) -> str:
    """Legend label for a surrogate variant, e.g. 'GNN_stable  surrogate'."""
    return f"{model_name}  surrogate"


def _colocated_legend_handles() -> list:
    """Shared legend handles for every plot in this module.

    The book (closed-form Kepler) and reference (leapfrog) lines are
    **dashed** by convention — they are "from the books". The six
    surrogate variants are **solid** lines; architectures are
    distinguished by colour and marker shape (see SURROGATE_MARKER).
    """
    handles = [
        plt.Line2D([], [], color=THEME["good"], lw=1.6,
                   linestyle=REFERENCE_DASH,
                   alpha=0.95, label="book (closed-form Kepler)"),
        plt.Line2D([], [], color="#ffffff", lw=1.2,
                   linestyle=REFERENCE_DASH,
                   alpha=0.9,  label="reference (leapfrog)"),
    ]
    for m in SURROGATE_VARIANT_ORDER:
        handles.append(plt.Line2D(
            [], [], color=_surr_color(m), lw=1.0,
            linestyle="-", alpha=0.9,
            label=_variant_label(m),
        ))
    return handles


# ── Per-preset plots ─────────────────────────────────────────────────────────
def _galactic_displacement(preset_name: str, n_frames: int) -> np.ndarray | None:
    """Return the Sun's displacement through the galaxy across `n_frames`
    frames in N-body units, expressed as (T, 3). The Sun moves at
    ~220 km/s in the local-standard-of-rest frame, which is ~46.4 AU/yr.
    For visual clarity we render the spiral as a helical sweep that
    covers ~2 L* over the simulation window — this is small enough to
    keep the orbits readable but large enough to make the spiral
    obvious.

    Returns None for presets where the transform is meaningless:
    jupiter_galileans (planetocentric — its primary is Jupiter, not the
    Sun) and disc_imf_in_distribution_baseline (an in-distribution
    dimensionless synthetic disc with neither a Sun nor any galaxy
    motion — adding a sweep here would fabricate one).
    """
    if preset_name in ("jupiter_galileans",
                       "disc_imf_in_distribution_baseline"):
        return None
    # Approximate a 2 L* sweep across the simulation window, along +x.
    # The exact scale depends on the preset's L*; for the in-extension
    # presets L* ≈ 30 AU so 2 L* is reasonable. We use 2 L* as the
    # sweep amplitude so the orbits spiral visibly.
    return np.linspace(0.0, 2.0, n_frames)[:, None] * np.array([1.0, 0.0, 0.0])


def plot_trajectory(preset_name: str,
                    ref_traj: np.ndarray,
                    surrogate_trajs: dict,
                    names: list[str],
                    out_path: str,
                    char_length_label: str = "L (N-body units)") -> None:
    """
    DEPRECATED: kept for backward compatibility. Originally provided
    one PNG per model (so each model gets its own page rather than all 6
    overlaid). Use `plot_trajectory_per_model` instead. This legacy
    function still works and writes a single combined file at `out_path`.

    xy projection of every integrator's trajectory.

    `surrogate_trajs` is a dict mapping model_name → (T, N, 3) np.ndarray.
    For solar-system presets (Sun-centred) the trajectory is shown in
    the galaxy frame: the Sun moves linearly through the galaxy at
    ~220 km/s, so the planets trace spring-like spirals rather than
    closed ellipses. For the planetocentric `jupiter_galileans` preset
    the primary is Jupiter, so the galaxy-frame transform is skipped.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["panel"])

    n_frames = ref_traj.shape[0]
    galaxy_disp = _galactic_displacement(preset_name, n_frames)
    in_galaxy = galaxy_disp is not None

    # Reference first (drawn beneath), then surrogates on top.
    # Compute label offsets so bodies that share the same start
    # position (e.g. all Galilean moons at t=0) don't overlap their
    # labels. We stagger the y-offset by body index.
    for body_i, body_name in enumerate(names):
        # Transform to galaxy frame if applicable. The reference is
        # already in the primary frame; add the Sun's galactic motion.
        if in_galaxy:
            x = ref_traj[:, body_i, 0] + galaxy_disp[:, 0]
            y = ref_traj[:, body_i, 1] + galaxy_disp[:, 1]
        else:
            x = ref_traj[:, body_i, 0]
            y = ref_traj[:, body_i, 1]
        body_c = _body_color(body_name, body_i)
        # Each body is its own colour so the reader can pair the
        # trajectory with the body name without consulting the legend.
        # Reference ("from the books") orbits are dashed by convention.
        ax.plot(x, y,
                color=body_c, lw=1.6, alpha=0.85,
                linestyle=REFERENCE_DASH,
                label=f"{body_name} (reference)" if body_i == 0 else None)
        ax.scatter([x[0]], [y[0]],
                   s=42, c=body_c, marker="o", zorder=5,
                   edgecolors=THEME["text"], linewidths=0.8)
        ax.scatter([x[-1]], [y[-1]],
                   s=48, c=body_c, marker="x", zorder=5,
                   linewidths=1.6)
        # Label the start-position dot so the reader can identify which
        # body is which. Stagger labels vertically by body index so
        # bodies at the same start position (e.g. Galilean moons at
        # t=0) don't overlap.
        label_yoff = 8 + 12 * (body_i % 4)
        label_xoff = 8
        ax.annotate(body_name, (x[0], y[0]),
                    textcoords="offset points", xytext=(label_xoff, label_yoff),
                    color=body_c, fontsize=9, weight="bold",
                    path_effects=[])

    for model_name, traj in surrogate_trajs.items():
        color = _surr_color(model_name)
        marker = _surr_marker(model_name)
        for body_i, body_name in enumerate(names):
            if in_galaxy:
                x = traj[:, body_i, 0] + galaxy_disp[:, 0]
                y = traj[:, body_i, 1] + galaxy_disp[:, 1]
            else:
                x = traj[:, body_i, 0]
                y = traj[:, body_i, 1]
            # Use the model's saturated fill for the line; mark the
            # start position with the same body-color so the body is
            # identifiable even when surrogates from the same model
            # for different bodies overlap.
            body_c = _body_color(body_name, body_i)
            ax.plot(x, y,
                    color=color, lw=1.0, alpha=0.55, linestyle="-",
                    label=model_name if body_i == 0 else None)
            # Mark the surrogate's start position with the body-color
            # marker so the reader can pair reference ↔ surrogate body.
            ax.scatter([x[0]], [y[0]],
                       s=30, c=body_c, marker=marker, zorder=4,
                       edgecolors=color, linewidths=1.2,
                       alpha=0.95)
            # Surrogate labels go below the cluster (negative offset).
            label_yoff = -12 - 12 * (body_i % 4)
            ax.annotate(f"{body_name} (pred)",
                        (x[0], y[0]),
                        textcoords="offset points", xytext=(8, label_yoff),
                        color=color, fontsize=8, alpha=0.95,
                        weight="bold")

    ax.set_xlabel("x  [L]")
    ax.set_ylabel("y  [L]")
    ax.set_aspect("equal", adjustable="datalim")
    title = (f"{preset_name}, xy trajectories (each body = unique colour)")
    if in_galaxy:
        title = (f"{preset_name}, xy trajectories in galaxy frame "
                 f"(Sun moves +x through Milky Way at ~220 km/s)")
    ax.set_title(title, color=THEME["text"], fontsize=11)
    ax.tick_params(colors=THEME["text"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(THEME["spine"])
    ax.grid(True, color=THEME["grid"], lw=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8,
              labelcolor=THEME["text"], facecolor=THEME["panel"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_trajectory_per_model(preset_name: str,
                              ref_traj: np.ndarray,
                              surrogate_trajs: dict,
                              names: list[str],
                              out_dir: str,
                              char_length_label: str = "L (N-body units)") -> None:
    """
    Per-model trajectory PNGs: writes one PNG per model_name into
    `out_dir`, named `trajectory_<model>.png`. Each PNG overlays the
    reference (per-body colour) and that model's predicted trajectory
    (saturated model colour) so the reader can compare book vs
    surrogate per body. The reference uses a thicker line so it stands
    out beneath the surrogate.

    Galaxy-frame transform applies as in `plot_trajectory` (skipped for
    planetocentric presets like `jupiter_galileans`).
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    n_frames = ref_traj.shape[0]
    galaxy_disp = _galactic_displacement(preset_name, n_frames)
    in_galaxy = galaxy_disp is not None

    for model_name, traj in surrogate_trajs.items():
        model_color = _surr_color(model_name)
        edge_color = _surr_edge(model_name)
        dash = _surr_dash(model_name)
        marker = _surr_marker(model_name)

        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        fig.patch.set_facecolor(THEME["bg"])
        ax.set_facecolor(THEME["panel"])

        for body_i, body_name in enumerate(names):
            body_c = _body_color(body_name, body_i)
            # Reference orbit (galaxy-frame if applicable).
            if in_galaxy:
                rx = ref_traj[:, body_i, 0] + galaxy_disp[:, 0]
                ry = ref_traj[:, body_i, 1] + galaxy_disp[:, 1]
            else:
                rx = ref_traj[:, body_i, 0]
                ry = ref_traj[:, body_i, 1]
            ax.plot(rx, ry, color=body_c, lw=1.6, alpha=0.85,
                    linestyle=REFERENCE_DASH,
                    label=f"{body_name} (ref)" if body_i == 0 else None)
            ax.scatter([rx[0]], [ry[0]], s=46, c=body_c, marker="o",
                       zorder=5, edgecolors=THEME["text"], linewidths=0.9)
            ax.scatter([rx[-1]], [ry[-1]], s=52, c=body_c, marker="x",
                       zorder=5, linewidths=1.8)
            label_yoff = 8 + 12 * (body_i % 4)
            ax.annotate(body_name, (rx[0], ry[0]),
                        textcoords="offset points", xytext=(8, label_yoff),
                        color=body_c, fontsize=10, weight="bold")

            # Surrogate orbit (solid; architecture is shown by colour + marker).
            if in_galaxy:
                sx = traj[:, body_i, 0] + galaxy_disp[:, 0]
                sy = traj[:, body_i, 1] + galaxy_disp[:, 1]
            else:
                sx = traj[:, body_i, 0]
                sy = traj[:, body_i, 1]
            ax.plot(sx, sy, color=model_color, lw=1.6, alpha=0.95,
                    linestyle="-",
                    label=f"{model_name} (pred)" if body_i == 0 else None)
            ax.scatter([sx[0]], [sy[0]], s=36, c=body_c, marker=marker,
                       zorder=4, edgecolors=edge_color, linewidths=1.4,
                       alpha=0.95)
            label_yoff = -12 - 12 * (body_i % 4)
            ax.annotate(f"{body_name} (pred)",
                        (sx[0], sy[0]),
                        textcoords="offset points", xytext=(8, label_yoff),
                        color=model_color, fontsize=8, alpha=0.95,
                        weight="bold")

        ax.set_xlabel("x  [L]")
        ax.set_ylabel("y  [L]")
        ax.set_aspect("equal", adjustable="datalim")
        title = (f"{preset_name}  ·  {model_name}  ·  xy trajectories")
        if in_galaxy:
            title = (f"{preset_name}  ·  {model_name}  ·  galaxy frame "
                     f"(Sun moves +x at ~220 km/s)")
        ax.set_title(title, color=THEME["text"], fontsize=12)
        ax.tick_params(colors=THEME["text"], labelsize=9)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, color=THEME["grid"], lw=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8,
                  labelcolor=THEME["text"], facecolor=THEME["panel"])

        fig.tight_layout()
        out_path = out_dir_path / f"trajectory_{model_name}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)


def plot_energy(preset_name: str,
                ref_energy: np.ndarray,
                surrogate_energies: dict,
                out_path: str) -> None:
    """
    |E(t) − E(0)| / |E(0)| per integrator on a log-y axis.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["panel"])

    ref_drift = np.abs(ref_energy - ref_energy[0]) / max(abs(ref_energy[0]), 1e-8)
    ax.plot(ref_drift, color="#ffffff", lw=1.8,
            linestyle=REFERENCE_DASH, label="reference (leapfrog)")

    for model_name, E in surrogate_energies.items():
        color = _surr_color(model_name)
        drift = np.abs(E - E[0]) / max(abs(E[0]), 1e-8)
        ax.plot(drift, color=color, lw=1.6, alpha=1.0, label=model_name,
                linestyle="-")

    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("|E(t) − E(0)| / |E(0)|")
    ax.set_title(f"{preset_name}, energy drift (log scale)",
                 color=THEME["text"], fontsize=11)
    ax.tick_params(colors=THEME["text"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(THEME["spine"])
    ax.grid(True, which="both", color=THEME["grid"], lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9,
              labelcolor=THEME["text"], facecolor=THEME["panel"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Per-preset error-vs-reference plot ──────────────────────────────────────
def plot_error_vs_reference(preset_name: str,
                             ref_traj: np.ndarray,
                             surrogate_trajs: dict,
                             out_path: str,
                             char_length_label: str = "L (N-body units)") -> None:
    """
    Per-step position-error curve vs the reference trajectory, for each
    surrogate. The vertical axis is `||surrogate_i(t) - ref_i(t)|| /
    char_L` averaged over the bodies in the preset (a single number per
    timestep). The reference curve is flat at zero by construction.

    Plotting the per-step error in its own panel (rather than overlaying
    trajectories on top of each other) makes the divergence readable:
    tiny early-step errors that look invisible on the trajectory plot
    become obvious on the error plot, and the frames_before_half_L
    threshold (red dashed line at 0.5) is easy to read off.

    `surrogate_trajs` is a dict mapping model_name -> (T, N, 3) np.ndarray.
    `ref_traj` is (T, N, 3).
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_facecolor(THEME["panel"])

    n_t = ref_traj.shape[0]
    t   = np.arange(n_t)

    # Reference curve: zero by definition (averaging (0, 0, ..., 0) = 0).
    ax.plot(t, np.zeros(n_t), color="#ffffff", lw=1.8,
            linestyle=REFERENCE_DASH, label="reference (leapfrog)")

    for model_name, traj in surrogate_trajs.items():
        color = _surr_color(model_name)
        # ||surrogate_i(t) - ref_i(t)|| per body per timestep, then
        # mean over bodies. Shape: (T, N, 3) -> (T,) after norm + mean.
        diff = traj - ref_traj
        per_body = np.linalg.norm(diff, axis=-1)              # (T, N)
        err     = per_body.mean(axis=-1)                      # (T,)
        ax.plot(t, err, color=color, lw=1.6, alpha=1.0,
                linestyle="-", label=model_name)

    # Threshold marker for the frames_before_half_L metric (pos error
    # is normalised by L, so 0.5 == half a characteristic length).
    ax.axhline(0.5, color=THEME["warn"], lw=1.2, linestyle=":",
               alpha=0.9, label="half-L threshold")

    ax.set_xlabel("step")
    ax.set_ylabel(f"mean position error / {char_length_label}")
    ax.set_title(f"{preset_name}, per-step position error vs reference",
                 color=THEME["text"], fontsize=11)
    ax.tick_params(colors=THEME["text"], labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(THEME["spine"])
    ax.grid(True, color=THEME["grid"], lw=0.4, alpha=0.5)
    ax.legend(loc="upper left", fontsize=8,
              labelcolor=THEME["text"], facecolor=THEME["panel"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Per-small-body "predicted vs book" scatter ───────────
def plot_predicted_vs_book(preset_name: str,
                           ric,                          # RescaledIC
                           ref_pos_slice: np.ndarray,    # (T, N, 3)
                           surrogate_traj: np.ndarray,   # (T, N, 6) ONE model
                           model_name: str,              # "GNN" / "LSTM_stable" / ...
                           book_trajs: np.ndarray,       # (T, N, 3)
                           out_path: str,
                           primary_idx: int = 0,
                           primary_for_body: np.ndarray | None = None,
                           ) -> None:
    """
    For every body, plot the *predicted position* (y-axis) against the
    *closed-form Kepler (book) position* (x-axis), in the body's primary
    frame — *one figure per surrogate variant*. The leapfrog reference
    is overplotted for free. A perfect prediction collapses onto the
    diagonal y=x.

    This is the "show the small body in its predicted
    position vs book position" chart, made per-model so a stranger can
    read each variant separately rather than picking 6 colours out of
    one hodge-podge.

    Per-body row (two columns):
      * Left  — predicted vs book (x, y, z components). The reference
                is a small white scatter; the surrogate is a coloured
                scatter in `model_name`'s colour. The dotted y=x line
                is the perfect prediction. Only TWO legend entries to
                keep the chart readable (diagonal + reference).
      * Right — single-number chart: scatter of (mean |r_book|, mean
                |r_predicted − r_book|) for THIS model + the
                reference. The legend label carries the mean error so
                the reader can read off a single number.

    Parameters
    ----------
    preset_name        : str, used in the title and the per-row labels.
    ric                : RescaledIC — body names + mass order.
    ref_pos_slice      : (T, N, 3), leapfrog reference aligned to the
                        surrogate predictions (frames WINDOW_SIZE ..
                        WINDOW_SIZE + n_rollout_ref).
    surrogate_traj     : (T, N, 6) — the SINGLE model's trajectory.
                        Slice [..., :3] gives positions.
    model_name         : str, the variant name (e.g. "LSTM_stable").
    book_trajs         : (T, N, 3), closed-form Kepler orbit for every
                        body; primary is pinned at its IC.
    out_path           : PNG file to write.
    primary_idx        : index of the Sun (or general "main" body).
    primary_for_body   : length-N int array mapping body → primary index
                        so moons can be plotted in their parent's frame.
    """
    surr_color = _surr_color(model_name)
    surr_dash  = SURROGATE_DASH.get(model_name, "--")

    n_bodies = ref_pos_slice.shape[1]
    body_ids = [b for b in range(n_bodies) if b != primary_idx]
    if not body_ids:
        body_ids = list(range(n_bodies))

    n_cols = 2
    n_rows = len(body_ids)
    fig = plt.figure(figsize=(13, 3.6 * n_rows))
    fig.patch.set_facecolor(THEME["bg"])
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.30)

    # Per-row figure-level super-legend (one row near the top labelled
    # with the 3 things on the page: reference, surrogate, y=x).
    fig_legend_ax = fig.add_axes([0.0, 0.965, 1.0, 0.03])
    fig_legend_ax.set_facecolor(THEME["bg"]); fig_legend_ax.axis("off")
    super_handles = [
        plt.Line2D([], [], color="#ffffff", lw=1.6, linestyle=REFERENCE_DASH,
                   marker="o", markersize=7,
                   label="reference (leapfrog)"),
        plt.Line2D([], [], color=surr_color, lw=2.0, linestyle="-",
                   alpha=1.0, label=_variant_label(model_name)),
        plt.Line2D([], [], color=THEME["grid"], lw=1.4, linestyle=":",
                   alpha=0.9, label="y = x (perfect prediction)"),
    ]
    fig_legend_ax.legend(handles=super_handles,
                         loc="lower center", ncol=3, fontsize=10,
                         labelcolor=THEME["text"], facecolor=THEME["bg"],
                         frameon=False, handlelength=2.6,
                         columnspacing=2.0, borderpad=0.2)

    surr_marker = _surr_marker(model_name)

    for row, body_i in enumerate(body_ids):
        body_name = (ric.names[body_i] if getattr(ric, "names", None)
                     else f"body_{body_i}")
        if primary_for_body is None:
            pri_i = primary_idx
        else:
            pri_i = int(primary_for_body[body_i])

        book_view = book_trajs[:, body_i, :] - book_trajs[:, pri_i, :]
        ref_view  = ref_pos_slice[:, body_i, :] - ref_pos_slice[:, pri_i, :]
        surr_view = surrogate_traj[:, body_i, :3] - ref_pos_slice[:, pri_i, :]

        # ── predicted vs book (combined 3-D view, x = book) ──────────
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor(THEME["panel"])
        # Clip the x/y range to ±clip_factor × max(|book|) so the
        # scatter doesn't smear into a vertical line at x=0 when
        # predictions diverge (the lines become unreadably fuzzy).
        # Use the maximum absolute value of the BOOK component only —
        # predicted values can be much larger when the surrogate
        # diverges, which would clip the book orbit to nothing.
        max_abs = float(np.max(np.abs(book_view))) if book_view.size else 1.0
        clip_factor = 2.0
        lim = max(max_abs * clip_factor, 1e-3)
        ax.plot([-lim, lim], [-lim, lim],
                color=THEME["grid"], lw=1.2, linestyle=":",
                alpha=0.9, label="y = x (perfect prediction)")
        # Reference (should be ~diagonal where perturbations < 0) —
        # use distinct markers per axis and a readable size+alpha.
        ref_kw = dict(s=14, c="#ffffff", alpha=0.9,
                      edgecolors=THEME["text"], linewidths=0.4)
        ax.scatter(book_view[:, 0], ref_view[:, 0],
                   marker="o", **ref_kw)
        ax.scatter(book_view[:, 1], ref_view[:, 1],
                   marker="s", **ref_kw)
        ax.scatter(book_view[:, 2], ref_view[:, 2],
                   marker="^", **ref_kw)
        # The single surrogate scatter — distinct shape per variant
        # so the model is identifiable even on a B/W printout.
        surr_kw = dict(s=12, c=surr_color, alpha=0.95,
                       edgecolors=THEME["text"], linewidths=0.4)
        ax.scatter(book_view[:, 0], surr_view[:, 0],
                   marker=surr_marker, **surr_kw)
        ax.scatter(book_view[:, 1], surr_view[:, 1],
                   marker=surr_marker, **surr_kw)
        ax.scatter(book_view[:, 2], surr_view[:, 2],
                   marker=surr_marker, **surr_kw)
        ax.set_xlabel("r_book  [L]  (closed-form Kepler, in primary frame)")
        ax.set_ylabel("r_predicted  [L]")
        ax.set_title(f"{body_name}: predicted vs book (x, y, z)",
                     color=THEME["text"], fontsize=10)
        ax.tick_params(colors=THEME["text"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, color=THEME["grid"], lw=0.5, alpha=0.6)
        # Use "box" aspect so the explicit xlim/ylim clip is respected
        # (set_aspect("equal", adjustable="datalim") would override
        # the clip back to the data range and re-create the smear).
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # Per-row compact legend: 3 entries (diagonal + reference +
        # this surrogate). Avoids the 24-entry marker-by-marker legend
        # that previously obscured the data.
        diag_handle = plt.Line2D([], [], color=THEME["grid"], lw=1.2,
                                 linestyle=":", alpha=0.9,
                                 label="y = x (perfect prediction)")
        ref_handle = plt.Line2D([], [], color="#ffffff", lw=1.6,
                                linestyle=REFERENCE_DASH,
                                marker="o", markersize=9,
                                label="reference (leapfrog)")
        surr_handle = plt.Line2D([], [], color=surr_color, lw=2.0,
                                 linestyle="-", alpha=1.0,
                                 label=_variant_label(model_name))
        ax.legend(handles=[diag_handle, ref_handle, surr_handle],
                  loc="upper left", fontsize=8,
                  labelcolor=THEME["text"], facecolor=THEME["panel"],
                  framealpha=0.95)

        # ── predicted vs book, vector displacement (one point / body) ─
        ax = fig.add_subplot(gs[row, 1])
        ax.set_facecolor(THEME["panel"])
        r_book_mean = float(np.linalg.norm(book_view, axis=-1).mean())
        r_ref_mean  = float(np.linalg.norm(ref_view, axis=-1).mean())
        surr_err_mean = float(np.linalg.norm(surr_view - book_view,
                                             axis=-1).mean())
        # Reference (always shown)
        ax.scatter([r_book_mean], [r_ref_mean],
                   s=160, c="#ffffff", marker="*",
                   edgecolors=THEME["text"], linewidths=1.4,
                   label=f"reference, err={r_ref_mean:.2e}")
        # This surrogate
        ax.scatter([r_book_mean], [surr_err_mean],
                   s=140, c=surr_color, marker=surr_marker,
                   edgecolors=THEME["text"], linewidths=0.8,
                   label=f"{model_name}, err={surr_err_mean:.2e}")
        # Reference line: perfect prediction (err = 0) and "err = orbit radius"
        ax.axhline(r_book_mean, color=THEME["grid"], lw=0.8, ls=":",
                   alpha=0.6,
                   label=f"err = orbit radius ({r_book_mean:.2e})")
        ax.set_xlabel("mean |r_book|  [L]  (size of the closed-form orbit)")
        ax.set_ylabel("mean prediction error vs book  [L]")
        ax.set_title(f"{body_name}: mean error vs orbit size",
                     color=THEME["text"], fontsize=10)
        ax.set_yscale("log")
        ax.tick_params(colors=THEME["text"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, which="both", color=THEME["grid"], lw=0.5, alpha=0.6)
        ax.legend(loc="lower right", fontsize=8,
                  labelcolor=THEME["text"], facecolor=THEME["panel"],
                  framealpha=0.95, markerscale=1.0)

    # Suptitle sits BELOW the super-legend strip (was overlapping
    # before — y=0.995 collided with the legend strip y=0.965-0.995).
    fig.suptitle(
        f"{preset_name}: {model_name}  predicted position vs closed-form "
        f"Kepler (book) reference, in each body's primary frame",
        color=THEME["text"], fontsize=12, y=0.94)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Cross-preset dashboard ──────────────────────────────────────────────────
def plot_dashboard(per_preset_payloads: list[dict], out_path: str) -> None:
    """
    One row per preset, two columns: trajectory / energy. Each cell uses
    the same per-preset plot functions but compacted.
    """
    n = len(per_preset_payloads)
    fig = plt.figure(figsize=(20, 4 * n))
    fig.patch.set_facecolor(THEME["bg"])
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.25)

    for i, p in enumerate(per_preset_payloads):
        # ── Trajectory subplot ──────────────────────────────────────────
        # Reference (book) orbits are dashed, surrogate orbits are solid.
        # The model is identified by colour from SURROGATE_COLORS.
        ax = fig.add_subplot(gs[i, 0])
        ax.set_facecolor(THEME["panel"])
        for body_i in range(p["ref_pos"].shape[1]):
            ax.plot(p["ref_pos"][:, body_i, 0], p["ref_pos"][:, body_i, 1],
                    color="#ffffff", lw=1.0, alpha=0.85,
                    linestyle=REFERENCE_DASH)
        for model_name, traj in p["surrogate_trajs"].items():
            color = _surr_color(model_name)
            for body_i in range(traj.shape[1]):
                ax.plot(traj[:, body_i, 0], traj[:, body_i, 1],
                        color=color, lw=1.0, alpha=0.95, linestyle="-")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x  [L]"); ax.set_ylabel("y  [L]")
        ax.set_title(f"{p['name']}, xy trajectories",
                     color=THEME["text"], fontsize=9)
        ax.tick_params(colors=THEME["text"], labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, color=THEME["grid"], lw=0.5, alpha=0.6)

        # ── Energy subplot ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[i, 1])
        ax.set_facecolor(THEME["panel"])
        ref_drift = np.abs(p["ref_energy"] - p["ref_energy"][0]) \
                    / max(abs(p["ref_energy"][0]), 1e-8)
        ax.plot(ref_drift, color="#ffffff", lw=1.8,
                linestyle=REFERENCE_DASH, label="reference (leapfrog)")
        for model_name, E in p["surrogate_energies"].items():
            color = _surr_color(model_name)
            drift = np.abs(E - E[0]) / max(abs(E[0]), 1e-8)
            ax.plot(drift, color=color, lw=1.4, alpha=1.0,
                    linestyle="-", label=model_name)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("|E(t)−E(0)| / |E(0)|")
        ax.set_title(f"{p['name']}, energy drift",
                     color=THEME["text"], fontsize=9)
        ax.tick_params(colors=THEME["text"], labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, which="both", color=THEME["grid"], lw=0.5, alpha=0.6)
        ax.legend(loc="upper left", fontsize=8,
                  labelcolor=THEME["text"], facecolor=THEME["panel"])

        # ── Per-step error-vs-reference subplot ────────────────────────
        ax = fig.add_subplot(gs[i, 2])
        ax.set_facecolor(THEME["panel"])
        n_t = p["ref_pos"].shape[0]
        t   = np.arange(n_t)
        ax.plot(t, np.zeros(n_t), color="#ffffff", lw=1.6,
                linestyle=REFERENCE_DASH, label="reference (leapfrog)")
        for model_name, traj in p["surrogate_trajs"].items():
            color = _surr_color(model_name)
            diff = traj - p["ref_pos"]
            err = np.linalg.norm(diff, axis=-1).mean(axis=-1)
            ax.plot(t, err, color=color, lw=1.2, alpha=1.0,
                    linestyle="-", label=model_name)
        ax.axhline(0.5, color=THEME["warn"], lw=1.0, linestyle=":",
                   alpha=0.9, label="half-L")
        ax.set_xlabel("step")
        ax.set_ylabel("mean pos error / L")
        ax.set_title(f"{p['name']}, error vs reference",
                     color=THEME["text"], fontsize=9)
        ax.tick_params(colors=THEME["text"], labelsize=7)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, color=THEME["grid"], lw=0.5, alpha=0.6)
        ax.legend(loc="upper left", fontsize=8,
                  labelcolor=THEME["text"], facecolor=THEME["panel"])

    fig.suptitle("Real-Case Validation, cross-preset dashboard",
                 color=THEME["text"], fontsize=13, y=0.998)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Per-small-body vs book-reference plot ──────────────
def plot_small_bodies_vs_book(preset_name: str,
                              ric,                          # RescaledIC
                              ref_pos_slice: np.ndarray,    # (T, N, 3) ref trajectory
                              surrogate_traj: np.ndarray,   # (T, N, 6) ONE model's traj
                              model_name: str,              # "GNN" / "LSTM_stable" / ...
                              book_trajs: np.ndarray,       # (T, N, 3) closed-form Kepler
                              out_path: str,
                              primary_idx: int = 0,
                              primary_for_body: np.ndarray | None = None) -> None:
    """
    Per-body diagnostic — *one figure per surrogate
    variant*. Each row shows the closed-form Kepler (book) orbit, the
    leapfrog reference, and the *single* chosen surrogate trajectory in
    the body's primary frame. Three lines per row, dedicated legend on
    every panel — no hodge-podge of 6 stacked models.

    Three panels per body:
      * xy projection:  positions in the (x, y) plane over the rollout
        horizon. Book = green solid; reference = white solid; surrogate
        = `_surr_color(model_name)` with `_variant_linestyle(model_name)`.
      * x-position-vs-time:  shows the same on a t-x strip so the
        reader can see the *phase* of the orbit. Book and reference
        should overlap perfectly when perturbations are negligible;
        any visible separation is the perturbation from other bodies.
      * radial error vs book:  |r_surrogate(t) - r_book(t)| vs
        |r_ref(t) - r_book(t)|, so the reader can read off the
        numerical divergence from the closed-form solution directly.

    The Sun (primary_idx) is skipped since its "book orbit" is a
    stationary point and contributes no information. For Galilean
    moons, a `primary_for_body` array routes each moon to its parent
    planet (Jupiter); the book's frame is then Jupiter-centred, and
    reference/surrogate panels subtract Jupiter's heliocentric
    position to make a like-for-like visual comparison.

    Notes
    -----
    `book_trajs[i]` should already be aligned to the *same time axis*
    as `ref_pos_slice` (typically `ref_pos_slice[i]` corresponds to
    `book_trajs[i]`); the runner is responsible for slicing.
    """
    surr_color = _surr_color(model_name)
    surr_edge  = _surr_edge(model_name)
    surr_dash  = SURROGATE_DASH.get(model_name, "--")

    n_bodies = ref_pos_slice.shape[1]
    body_ids = [b for b in range(n_bodies) if b != primary_idx]
    if not body_ids:
        # No small bodies (shouldn't happen for a real preset); fall back.
        body_ids = list(range(n_bodies))

    n_cols = 3
    n_rows = len(body_ids)
    fig = plt.figure(figsize=(16, 3.5 * n_rows))
    fig.patch.set_facecolor(THEME["bg"])
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.55, wspace=0.30)

    # Compute a single global t-axis (sample index); all panels share it.
    T = ref_pos_slice.shape[0]
    t_axis = np.arange(T)

    for row, body_i in enumerate(body_ids):
        body_name = (ric.names[body_i] if getattr(ric, "names", None)
                     else f"body_{body_i}")

        # Each body has its own primary. Re-base every trajectory into
        # the primary's frame.
        if primary_for_body is None:
            pri_i = primary_idx
        else:
            pri_i = int(primary_for_body[body_i])
        book_view = book_trajs[:, body_i, :] - book_trajs[:, pri_i, :]
        ref_view  = ref_pos_slice[:, body_i, :] - ref_pos_slice[:, pri_i, :]
        surr_view = surrogate_traj[:, body_i, :3] - ref_pos_slice[:, pri_i, :]

        # ── Per-row legend (3 lines: book / reference / this model).
        # Visual convention: dashed = "from the books" (closed-form
        # Kepler OR leapfrog reference), solid = model's predicted path.
        row_legend = [
            plt.Line2D([], [], color=THEME["good"], lw=2.0,
                       linestyle=REFERENCE_DASH,
                       alpha=1.0, label="book (closed-form Kepler)"),
            plt.Line2D([], [], color="#ffffff", lw=1.6,
                       linestyle=REFERENCE_DASH,
                       alpha=1.0, label="reference (leapfrog)"),
            plt.Line2D([], [], color=surr_color, lw=1.8, linestyle="-",
                       alpha=1.0, label=_variant_label(model_name)),
        ]

        # ── xy trajectory panel (in primary's frame) ───────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor(THEME["panel"])
        ax.plot(book_view[:, 0], book_view[:, 1],
                color=THEME["good"], lw=2.0, alpha=1.0,
                linestyle=REFERENCE_DASH)
        ax.plot(ref_view[:, 0], ref_view[:, 1],
                color="#ffffff", lw=1.6, alpha=1.0,
                linestyle=REFERENCE_DASH)
        ax.plot(surr_view[:, 0], surr_view[:, 1],
                color=surr_color, lw=1.8, alpha=1.0, linestyle="-")
        # Mark start point of book orbit
        ax.scatter([book_view[0, 0]], [book_view[0, 1]],
                   s=36, c=THEME["good"], marker="o", zorder=6,
                   edgecolors=THEME["text"], linewidths=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x  [L]"); ax.set_ylabel("y  [L]")
        ax.set_title(f"{body_name}, xy trajectory (in primary frame)",
                     color=THEME["text"], fontsize=10)
        ax.tick_params(colors=THEME["text"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, color=THEME["grid"], lw=0.5, alpha=0.6)
        ax.legend(handles=row_legend,
                  loc="upper right", fontsize=9,
                  labelcolor=THEME["text"], facecolor=THEME["panel"],
                  framealpha=0.95)

        # ── x-vs-t panel ───────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        ax.set_facecolor(THEME["panel"])
        ax.plot(t_axis, book_view[:, 0],
                color=THEME["good"], lw=2.0, alpha=1.0,
                linestyle=REFERENCE_DASH)
        ax.plot(t_axis, ref_view[:, 0],
                color="#ffffff", lw=1.6, alpha=1.0,
                linestyle=REFERENCE_DASH)
        ax.plot(t_axis, surr_view[:, 0],
                color=surr_color, lw=1.8, alpha=1.0, linestyle="-")
        ax.set_xlabel("sample step"); ax.set_ylabel("x  [L]")
        ax.set_title(f"{body_name}, x(t): book vs reference vs {model_name}",
                     color=THEME["text"], fontsize=10)
        ax.tick_params(colors=THEME["text"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, color=THEME["grid"], lw=0.5, alpha=0.6)
        ax.legend(handles=row_legend,
                  loc="upper right", fontsize=9,
                  labelcolor=THEME["text"], facecolor=THEME["panel"],
                  framealpha=0.95)

        # ── radial error vs book ───────────────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        ax.set_facecolor(THEME["panel"])
        ref_err = np.linalg.norm(ref_view - book_view, axis=-1)
        surr_err = np.linalg.norm(surr_view - book_view, axis=-1)
        ax.plot(t_axis, ref_err, color="#ffffff", lw=1.4, alpha=0.85,
                linestyle=REFERENCE_DASH,
                label="reference vs book")
        ax.plot(t_axis, surr_err, color=surr_color, lw=1.8, alpha=1.0,
                linestyle="-", label=f"{model_name} vs book")
        # Use log only when the dynamic range warrants it.
        if float(ref_err.max()) > 0 and \
                float(surr_err.max()) / max(float(ref_err.max()), 1e-12) > 50:
            ax.set_yscale("log")
        ax.set_xlabel("sample step")
        ax.set_ylabel("|r − r_book|  /  L")
        ax.set_title(f"{body_name}, radial error vs closed-form Kepler",
                     color=THEME["text"], fontsize=10)
        ax.tick_params(colors=THEME["text"], labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor(THEME["spine"])
        ax.grid(True, which="both", color=THEME["grid"], lw=0.5, alpha=0.6)
        ax.legend(loc="upper left", fontsize=9,
                  labelcolor=THEME["text"], facecolor=THEME["panel"],
                  framealpha=0.95)

    fig.suptitle(
        f"{preset_name}: {model_name} surrogate, predicted position vs "
        f"closed-form Kepler (book) reference vs leapfrog reference. "
        f"Each row = one body, viewed in its primary frame "
        f"(planets → Sun, moons → planet).",
        color=THEME["text"], fontsize=12, y=0.998)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
