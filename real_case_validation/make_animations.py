"""
make_animations.py
==================

Trajectory-animations for the real-case validation. Renders the closed-form
Kepler reference (book) and the surrogate rollout for each persisted model as
two side-by-side panels per body (one row per non-primary body):

  1. ORIGINAL / book panel (left): the closed-form Kepler reference as a
     faint dotted orbit, plus the central body drawn at its physical radius
     (Jupiter for jupiter_galileans, Sun otherwise), plus the small body
     animated at its book position with a running label.
  2. PREDICTED / surrogate panel (right): the surrogate trajectory as a
     faint dashed orbit, plus the same central body, plus the small body
     animated at its predicted position with a running trailing line.
  3. Radial-error panel (rightmost): |r_surrogate − r_book| in N-body
     units running over frame index.

The book-vs-predicted panels share the same axis bounding box so the gap
between the two is visually unmistakable. The central body is drawn at
its physical radius (in N-body units, with a minimum visibility floor) so
the small body's orbit is visually anchored to a real body rather than a
bare coordinate system.

The script reads `preds.npy` + `book_pos.npy` + `preds_meta.json` from
`real_case_validation/report_N{n}/preset_<NAME>/` (a per-N dir, defaulting
to the numerically smallest `report_N*` sibling it can find) that the
runner writes when `--dump-preds` is set.

Pure CPU. Uses matplotlib's bundled `FFMpegWriter` + `PillowWriter` for
.gif fallback. No new pip installs (no `imageio`).

CLI
---
    python -m real_case_validation.make_animations \\
        --preset jupiter_galileans --model LSTM \\
        --fps 30 --frames 1460 --out <dir> --format mp4 --view 3d

Notes
-----
* Default preset dir: the first `report_N*/preset_<NAME>/` sibling found
  under `real_case_validation/`. Set `--report-dir` to override.
* For presets with `in_distribution = True` (the disc baseline), the
  surrogate is essentially noise-free, so the animation is still
  useful as a sanity check but the predicted panel will look like a
  single curve.
* The book orbit is in dimensionless N-body units, so the radial-error
  panel also uses N-body units (matches the existing `predicted_vs_book`
  plot's units).
* **Galaxy-frame mode** is enabled automatically for Sun-centred
  presets (every preset except `jupiter_galileans` and the
  in-distribution disc baseline). A constant +x drift is added to
  every heliocentric position to visualise the Sun's motion through
  the Milky Way (~220 km/s local standard of rest, rendered as a
  fixed drift in code units); the planets trace spring-like spirals
  and the primary body (Sun) moves across the panel each frame. The
  surrogate-vs-book *difference* is invariant to the common shift,
  but the plotted radial error |‖s+g‖ − ‖b+g‖| is not exactly equal
  to |‖s‖ − ‖b‖| — the invariant quantity is shown as the gap between
  the two trajectory lines in each body panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable both as `python -m
# real_case_validation.make_animations` (the canonical form) and as
# `python real_case_validation/make_animations.py`. The shim must run
# at module import time so `from utils import …` works on the direct
# invocation path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from utils import load_sibling_module

# Reuse the dark theme + per-surrogate colours from the plot module.
_val_mod = load_sibling_module("nbody_validation", "validation.py")
THEME = _val_mod.THEME
# Constants for converting physical body radii -> N-body units.
from pipeline_config import AU_M as _AU_M  # metres per AU
_KM_TO_AU = 1000.0 / _AU_M
# Direct file import (not via load_sibling_module — that helper uses
# utils.py's anchor, which is the repo root, not this script's dir).
import importlib.util as _ilu
def _load_sibling(name: str, filename: str):
    p = Path(__file__).with_name(filename)
    spec = _ilu.spec_from_file_location(name, p)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
_plots_mod = _load_sibling("real_case_validation.plots", "plots.py")
SURROGATE_COLORS = _plots_mod.SURROGATE_COLORS
SURROGATE_DASH = _plots_mod.SURROGATE_DASH
_surr_color = _plots_mod._surr_color
_surr_dash = lambda n: SURROGATE_DASH.get(n, "-")
_surr_marker = _plots_mod._surr_marker
_add_ensemble_color = _plots_mod._add_ensemble_color


# Body-name resolution — the persisted arrays don't carry names, so we
# inspect the preset's bodies list to label the per-body subplots.
def _body_names(preset: dict) -> list[str]:
    """Return ['Sun', 'Mercury', 'Venus', ...] per preset."""
    names = []
    bodies = preset.get("bodies") or []
    for b in bodies:
        nm = b.get("name", "")
        if not nm:
            # Fallback: "body_0", "body_1", ...
            names.append("body")
        else:
            names.append(nm)
    return names


def _discover_presets(report_dir: Path) -> list[str]:
    """Return the preset names available under `report_dir`."""
    out = []
    for p in sorted(report_dir.iterdir()):
        if p.is_dir() and p.name.startswith("preset_"):
            if (p / "preds.npy").exists() and (p / "preds_meta.json").exists():
                out.append(p.name[len("preset_"):])
    return out


def _load_preset(preset_name: str, report_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (preds, book_pos, meta) for one preset."""
    preset_dir = report_dir / f"preset_{preset_name}"
    preds = np.load(preset_dir / "preds.npy")          # (M, T, N, 6)
    book_pos = np.load(preset_dir / "book_pos.npy")    # (T, N, 3)
    with open(preset_dir / "preds_meta.json") as f:
        meta = json.load(f)
    return preds, book_pos, meta


def _resolve_preset_module(preset_name: str):
    """Import the presets module and return its `get_preset` function."""
    import importlib
    pm = importlib.import_module("real_case_validation.presets")
    return pm.get_preset(preset_name)


def _view_for_body(preds: np.ndarray, book_pos: np.ndarray,
                   primary_idx: int = 0, body_i: int = 1) -> np.ndarray:
    """Return (T, 3) for `body_i` in the primary's frame, using book_pos."""
    return book_pos[:, body_i, :] - book_pos[:, primary_idx, :]


def _colors_for_model(model_name: str) -> tuple[str, str]:
    """Return (fill, edge) colour tuple for `model_name`."""
    _add_ensemble_color(model_name)
    return SURROGATE_COLORS.get(model_name, (THEME["warn"], THEME["warn"]))


# Visual radii (in metres) for the bodies that can appear as a primary
# in our real-case presets. Used to draw the central body at its
# physical scale (or a tiny exaggeration if the camera is far away).
# Source: NASA Planetary Fact Sheet.
_VISUAL_RADIUS_M = {
    "Sun":     6.957e8,
    "Mercury": 2.439e6,
    "Venus":   6.052e6,
    "Earth":   6.371e6,
    "Mars":    3.390e6,
    "Jupiter": 6.991e7,
    "Saturn":  5.823e7,
    "Uranus":  2.536e7,
    "Neptune": 2.462e7,
    "Pluto":   1.188e6,
    "Moon":    1.737e6,
    "Io":      1.822e6,
    "Europa":  1.561e6,
    "Ganymede":2.634e6,
    "Callisto":2.410e6,
}


def _primary_radius_in_nbody(primary_name: str, preset: dict) -> float:
    """Return the primary's body radius in N-body units (assuming the
    preset's `characteristic_radius_au` is the L*). Falls back to a
    sensible default if the preset doesn't carry a radius.
    """
    r_m = _VISUAL_RADIUS_M.get(primary_name, 0.0)
    if r_m <= 0.0:
        return 0.0
    r_au = r_m * _KM_TO_AU / 1e3  # m -> AU
    L_au = preset.get("characteristic_radius_au")
    # Many presets don't set `characteristic_radius_au`; infer a
    # reasonable L* from the orbital period so the Sun is visible.
    # For sun_earth_only the period is ~1 year → L* ≈ 1 AU.
    # For jupiter_galileans the period is ~12 years → L* ≈ 0.05 AU.
    if not L_au or L_au <= 0:
        duration_years = preset.get("duration_years", 10.0)
        sample_per_year = preset.get("sample_per_year", 60)
        # Heuristic: first body's orbital radius is the L*. For the
        # presets without an explicit L*, fall back to a 1 AU default
        # (works for sun_earth_only, inner_planets, full_solar_system).
        # For planetocentric presets (jupiter_galileans) L* is much
        # smaller (~0.005 AU) but Jupiter's radius is also larger in
        # absolute terms, so the clamp on r_n still gives a visible
        # body.
        L_au = 1.0
    r_n = r_au / L_au
    # The Sun's physical radius is ~0.005 L* (Earth orbit) which is
    # sub-pixel at most camera distances. Exaggerate to a visible
    # 5% of the orbit so the Sun is anchored on screen. Other bodies
    # (Jupiter, planets) keep their real scale.
    if primary_name == "Sun":
        return max(r_n, 0.05)
    return max(r_n, 1e-4)

def _make_3d_panel(ax, body_name: str, primary_name: str,
                   book_view: np.ndarray, ref_traj: np.ndarray | None,
                   surr_traj: np.ndarray,
                   model_name: str,
                   primary_radius: float = 0.0,
                   show_primary: bool = True,
                   trail: int = 60, is_3d: bool = True) -> None:
    """Draw the static background (book orbit, ref, surrogate already
    plotted up to `frame_i` minus the trailing line).

    Adds:
    - The primary body (e.g. Jupiter) as a filled circle at the origin so
      the small body's orbit is visually anchored to a real central body.
    - The book orbit as a faint dotted "rail" (full curve).
    - The reference (leapfrog) trajectory as a white solid line if present.
    """
    ax.set_facecolor(THEME["panel"])
    ax.set_title(f"{body_name} (primary: {primary_name})", color=THEME["text"])
    ax.set_xlabel("x [L*]", color=THEME["text"])
    ax.set_ylabel("y [L*]", color=THEME["text"])
    if is_3d:
        ax.set_zlabel("z [L*]", color=THEME["text"])
    # Book orbit as a faint dotted "rail" (full curve).
    if is_3d:
        ax.plot(book_view[:, 0], book_view[:, 1], book_view[:, 2],
                color=THEME["grid"], lw=1.0, alpha=0.5, linestyle=":",
                label="book (closed-form Kepler)")
    else:
        ax.plot(book_view[:, 0], book_view[:, 1],
                color=THEME["grid"], lw=1.0, alpha=0.5, linestyle=":",
                label="book (closed-form Kepler)")
    # Reference (leapfrog) trajectory: full curve white, lw=1.4.
    if ref_traj is not None:
        if is_3d:
            ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2],
                    color=THEME["text"], lw=1.4, alpha=0.7,
                    label="reference (leapfrog)")
        else:
            ax.plot(ref_traj[:, 0], ref_traj[:, 1],
                    color=THEME["text"], lw=1.4, alpha=0.7,
                    label="reference (leapfrog)")
    # Central body marker at origin (primary's frame). Radius is the
    # canonical body radius in plotted units; in 3D this is a 3D sphere
    # drawn with a scatter; in 2D a flat disc.
    if show_primary and primary_radius > 0.0:
        if is_3d:
            # Approximate a sphere with a scatter; pick a coarse mesh
            # so render remains fast.
            u = np.linspace(0, 2*np.pi, 16)
            v = np.linspace(0, np.pi, 8)
            xs = primary_radius * np.outer(np.cos(u), np.sin(v))
            ys = primary_radius * np.outer(np.sin(u), np.sin(v))
            zs = primary_radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(xs, ys, zs, color=THEME["accent"],
                            alpha=0.85, edgecolor=THEME["text"],
                            linewidth=0.2, shade=True)
        else:
            circ = plt.Circle((0, 0), primary_radius, color=THEME["accent"],
                              alpha=0.85, ec=THEME["text"], lw=0.6,
                              label=f"{primary_name} (radius ×)"
                                            f"{primary_radius:.2e}")
            ax.add_patch(circ)
    ax.tick_params(colors=THEME["text"])
    # Equal aspect: BBox driven by the book orbit AND the primary radius
    # so the central body is always visible.
    bbox = np.array([
        book_view.min(axis=0), book_view.max(axis=0)
    ])
    centre = bbox.mean(axis=0)
    orbit_half = max((bbox[1] - bbox[0]).max() / 2.0, 1e-9)
    # If a primary is drawn, ensure at least 1.3 * primary_radius is
    # visible around the origin — otherwise the panel just shows the body.
    vis_half = max(orbit_half, primary_radius * 1.3 if show_primary else 0.0)
    ax.set_xlim(centre[0] - vis_half, centre[0] + vis_half)
    ax.set_ylim(centre[1] - vis_half, centre[1] + vis_half)
    if is_3d:
        ax.set_zlim(centre[2] - vis_half, centre[2] + vis_half)
        # Subtle grid.
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
    else:
        ax.set_aspect("equal", adjustable="box")


def _draw_primary(ax, primary_name: str, primary_radius: float,
                  is_3d: bool, pos: np.ndarray | None = None) -> list:
    """Draw the central body (primary) at the origin (or at `pos` in
    galaxy-frame mode where the Sun moves through the galaxy).

    Returns the list of matplotlib artists created so the caller can
    remove() them on subsequent frames. This matters in galaxy-frame
    mode where the primary position changes every frame — without
    explicit cleanup the figure accumulates one sphere/circle + label
    per frame and the .mp4 file balloons (1500 frames × N cells × 2
    artists).
    """
    if primary_radius <= 0.0:
        return []
    if pos is None:
        pos = np.zeros(3)
    artists = []
    if is_3d:
        u = np.linspace(0, 2*np.pi, 16)
        v = np.linspace(0, np.pi, 8)
        xs = pos[0] + primary_radius * np.outer(np.cos(u), np.sin(v))
        ys = pos[1] + primary_radius * np.outer(np.sin(u), np.sin(v))
        zs = pos[2] + primary_radius * np.outer(np.ones_like(u), np.cos(v))
        artists.append(ax.plot_surface(xs, ys, zs, color=THEME["accent"],
                                       alpha=0.85, edgecolor=THEME["text"],
                                       linewidth=0.2, shade=True))
    else:
        circ = plt.Circle((pos[0], pos[1]), primary_radius, color=THEME["accent"],
                          alpha=0.85, ec=THEME["text"], lw=0.6)
        ax.add_patch(circ)
        artists.append(circ)
    # Label the primary at its current position so the reader knows
    # which body is in the centre.
    if is_3d:
        artists.append(ax.text(pos[0], pos[1], pos[2] + primary_radius * 1.4,
                               primary_name, color=THEME["text"], fontsize=9,
                               ha="center", va="center", weight="bold"))
    else:
        artists.append(ax.text(pos[0], pos[1] + primary_radius * 1.4,
                               primary_name, color=THEME["text"], fontsize=9,
                               ha="center", va="center", weight="bold"))
    return artists


def _make_3d_panel_book(ax, body_name: str, primary_name: str,
                        book_view: np.ndarray,
                        primary_radius: float, is_3d: bool,
                        sun_pos: np.ndarray | None = None) -> None:
    """Left panel: ORIGINAL system (closed-form Kepler)."""
    ax.set_facecolor(THEME["panel"])
    ax.set_title(f"ORIGINAL  ·  {body_name} (primary: {primary_name})",
                 color=THEME["good"], fontsize=11)
    ax.set_xlabel("x [L*]", color=THEME["text"])
    ax.set_ylabel("y [L*]", color=THEME["text"])
    if is_3d:
        ax.set_zlabel("z [L*]", color=THEME["text"])
    # Full book orbit (faint dashed rail — by convention "dashed = from
    # the books" matches the static trajectory plots).
    if is_3d:
        ax.plot(book_view[:, 0], book_view[:, 1], book_view[:, 2],
                color=THEME["good"], lw=1.0, alpha=0.55, linestyle="--",
                label="book orbit (ref)")
    else:
        ax.plot(book_view[:, 0], book_view[:, 1],
                color=THEME["good"], lw=1.0, alpha=0.55, linestyle="--",
                label="book orbit (ref)")
    _draw_primary(ax, primary_name, primary_radius, is_3d, pos=sun_pos)
    # Axis bounds: union of book-extent and primary radius.
    bbox = np.array([book_view.min(axis=0), book_view.max(axis=0)])
    centre = bbox.mean(axis=0)
    orbit_half = max((bbox[1] - bbox[0]).max() / 2.0, 1e-9)
    vis_half = max(orbit_half, primary_radius * 1.3)
    ax.set_xlim(centre[0] - vis_half, centre[0] + vis_half)
    ax.set_ylim(centre[1] - vis_half, centre[1] + vis_half)
    if is_3d:
        ax.set_zlim(centre[2] - vis_half, centre[2] + vis_half)
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=THEME["text"])


def _make_3d_panel_surr(ax, body_name: str, primary_name: str,
                        surr_traj: np.ndarray,
                        model_name: str, primary_radius: float,
                        is_3d: bool,
                        sun_pos: np.ndarray | None = None) -> None:
    """Right panel: PREDICTED system (surrogate)."""
    ax.set_facecolor(THEME["panel"])
    ax.set_title(f"PREDICTED  ·  {body_name} ({model_name})",
                 color=THEME["accent"], fontsize=11)
    ax.set_xlabel("x [L*]", color=THEME["text"])
    ax.set_ylabel("y [L*]", color=THEME["text"])
    if is_3d:
        ax.set_zlabel("z [L*]", color=THEME["text"])
    # Full surrogate trajectory (faint dashed rail).
    fill, _ = _colors_for_model(model_name)
    if is_3d:
        ax.plot(surr_traj[:, 0], surr_traj[:, 1], surr_traj[:, 2],
                color=fill, lw=1.0, alpha=0.55,
                linestyle=_surr_dash(model_name),
                label=f"{model_name} trajectory")
    else:
        ax.plot(surr_traj[:, 0], surr_traj[:, 1],
                color=fill, lw=1.0, alpha=0.55,
                linestyle=_surr_dash(model_name),
                label=f"{model_name} trajectory")
    _draw_primary(ax, primary_name, primary_radius, is_3d, pos=sun_pos)
    # Axis bounds: union of surrogate-extent and primary radius.
    bbox = np.array([surr_traj.min(axis=0), surr_traj.max(axis=0)])
    centre = bbox.mean(axis=0)
    orbit_half = max((bbox[1] - bbox[0]).max() / 2.0, 1e-9)
    vis_half = max(orbit_half, primary_radius * 1.3)
    ax.set_xlim(centre[0] - vis_half, centre[0] + vis_half)
    ax.set_ylim(centre[1] - vis_half, centre[1] + vis_half)
    if is_3d:
        ax.set_zlim(centre[2] - vis_half, centre[2] + vis_half)
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=THEME["text"])


def _make_radial_panel(ax, body_name: str, primary_name: str,
                       book_view: np.ndarray, surr_traj: np.ndarray,
                       model_name: str,
                       err: np.ndarray) -> None:
    """Draw the static background of the radial-error panel."""
    ax.set_facecolor(THEME["panel"])
    ax.set_title(f"Radial error  |r_surrogate − r_book| / L*",
                 color=THEME["text"])
    ax.set_xlabel("frame", color=THEME["text"])
    ax.set_ylabel(f"|r_surrogate − r_book| [L*]", color=THEME["text"])
    ax.grid(True, color=THEME["grid"], lw=0.3, alpha=0.5)
    # Plot the *full* curve very faintly so the running line has context.
    fill, _ = _colors_for_model(model_name)
    ax.plot(err, color=fill, lw=0.7, alpha=0.30)
    ax.set_xlim(0, len(err))
    # Choose log/linear based on dynamic range.
    if err.max() > 0 and err.min() >= 0:
        rng = err.max() / max(err[err > 0].min() if (err > 0).any() else 1e-12, 1e-12)
    else:
        rng = 1.0
    if rng > 1e3:
        ax.set_yscale("log")
    ax.tick_params(colors=THEME["text"])
    ax.legend(loc="upper left",
              handles=[plt.Line2D([0], [0], color=fill, lw=1.8,
                                  label=f"{body_name} ({model_name})")],
              framealpha=0.95, fontsize=9, labelcolor=THEME["text"])


def _make_combined_panel(ax, body_name: str, primary_name: str,
                          book_view: np.ndarray, surr_traj: np.ndarray,
                          model_name: str,
                          primary_radius: float, is_3d: bool,
                          sun_pos: np.ndarray | None = None) -> None:
    """One panel per body: book orbit (faint dotted) + surrogate trailing
    line / head markers drawn live in _update. Aligned to the BOOK Sun
    so the predicted-vs-book gap is the surrogate's own error, not the
    Sun's drift."""
    ax.set_facecolor(THEME["panel"])
    ax.set_title(f"{body_name} (primary: {primary_name})",
                 color=THEME["text"], fontsize=10)
    ax.set_xlabel("x [L*]", color=THEME["text"], fontsize=8)
    ax.set_ylabel("y [L*]", color=THEME["text"], fontsize=8)
    if is_3d:
        ax.set_zlabel("z [L*]", color=THEME["text"], fontsize=8)
    # Book orbit (faint dashed rail — "from the books" convention) in
    # the body colour so the orbit matches the per-body colour scheme.
    body_c = _plots_mod._body_color(body_name, 0)
    if is_3d:
        ax.plot(book_view[:, 0], book_view[:, 1], book_view[:, 2],
                color=body_c, lw=0.9, alpha=0.45, linestyle="--",
                label="book (ref)")
    else:
        ax.plot(book_view[:, 0], book_view[:, 1],
                color=body_c, lw=0.9, alpha=0.45, linestyle="--",
                label="book (ref)")
    _draw_primary(ax, primary_name, primary_radius, is_3d, pos=sun_pos)
    # Axis bounds: union of book + surrogate extents + primary radius.
    bbox = np.array([
        np.minimum(book_view.min(axis=0), surr_traj.min(axis=0)),
        np.maximum(book_view.max(axis=0), surr_traj.max(axis=0)),
    ])
    centre = bbox.mean(axis=0)
    orbit_half = max((bbox[1] - bbox[0]).max() / 2.0, 1e-9)
    vis_half = max(orbit_half, primary_radius * 1.3)
    ax.set_xlim(centre[0] - vis_half, centre[0] + vis_half)
    ax.set_ylim(centre[1] - vis_half, centre[1] + vis_half)
    if is_3d:
        ax.set_zlim(centre[2] - vis_half, centre[2] + vis_half)
        ax.xaxis.pane.set_alpha(0.05)
        ax.yaxis.pane.set_alpha(0.05)
        ax.zaxis.pane.set_alpha(0.05)
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=THEME["text"], labelsize=7)


def _make_global_error_panel(ax, axes, model_name: str) -> None:
    """Single full-width radial-error panel at the bottom of the figure:
    one running curve per body, each in its own per-body colour."""
    ax.set_facecolor(THEME["panel"])
    ax.set_title(f"Radial error  |r_surrogate − r_book| / L*  (one curve per body)",
                 color=THEME["text"], fontsize=10)
    ax.set_xlabel("frame", color=THEME["text"])
    ax.set_ylabel(f"|r_surrogate − r_book| [L*]", color=THEME["text"])
    ax.grid(True, color=THEME["grid"], lw=0.3, alpha=0.5)
    # Faint full-curve background per body so the running line has context.
    for cell_i, (ax_body, body_i, surr_traj, err, body_name,
                 is_3d_c, book_view) in enumerate(axes):
        body_c = _plots_mod._body_color(body_name, body_i)
        ax.plot(err, color=body_c, lw=0.7, alpha=0.25)
    # x-axis limit based on the longest err.
    if axes:
        max_T = max(len(axes[c][3]) for c in range(len(axes)))
        ax.set_xlim(0, max_T)
    # Log scale if dynamic range is large.
    if axes:
        all_err = np.concatenate([a[3] for a in axes])
        if all_err.max() > 0:
            pos = all_err[all_err > 0]
            if pos.size > 0:
                rng = all_err.max() / pos.min()
                if rng > 1e3:
                    ax.set_yscale("log")
    ax.tick_params(colors=THEME["text"], labelsize=8)


def _animate_one(preset_name: str, model_name: str,
                 report_dir: Path, out_dir: Path,
                 view: str, frames: int, fps: int,
                 fmt: str, trail: int = 60) -> list[Path]:
    """Render one (preset, model) animation. Returns list of paths written."""
    preds, book_pos, meta = _load_preset(preset_name, report_dir)
    model_names = meta["models"]
    if model_name not in model_names:
        print(f"  [skip] {model_name} not in {model_name}'s model list "
              f"{model_names}", file=sys.stderr)
        return []
    model_idx = model_names.index(model_name)
    surr_full = preds[model_idx, :, :, :]                # (T, N, 6)
    T, n_bodies, _ = surr_full.shape
    # Primary index is 0 (Sun or Jupiter) per the runner's convention.
    primary_idx = 0
    if frames is None or frames >= T:
        frames = T
    # Resolve body names from the presets module.
    preset = _resolve_preset_module(preset_name)
    body_names = _body_names(preset)
    # In-distribution presets carry `bodies=None` (populated by the loader
    # at runtime, not persisted). Fall back to synthetic placeholder
    # names so the per-body subplots still label cleanly.
    if not body_names:
        body_names = [f"body_{i}" for i in range(n_bodies)]
    if len(body_names) < n_bodies:
        # Pad with placeholder names; shouldn't happen for the current
        # presets but stay defensive.
        body_names = body_names + [f"body_{i}" for i in range(len(body_names), n_bodies)]
    primary_name = body_names[primary_idx]
    primary_radius = _primary_radius_in_nbody(primary_name, preset)

    # Galaxy-frame: the Sun moves through the Milky Way at ~220 km/s
    # (~46.4 AU/yr). Every planet traces a spring-like spiral rather
    # than a closed ellipse. We add a constant velocity to all
    # heliocentric positions. The amplitude is scaled so the spiral
    # sweep is clearly visible inside the orbit but not so dominant
    # that the orbit becomes unreadable.
    galaxy_disp = _plots_mod._galactic_displacement(preset_name, T)
    in_galaxy = galaxy_disp is not None
    galaxy_subtitle = "  ·  galaxy frame (Sun moves +x at ~220 km/s)" if in_galaxy else ""

    # Build a subplot grid: 2 rows × 4 cols of body panels + 1 row of
    # 4 global radial-error subplots at the bottom. Each body panel
    # overlays the book orbit (faint grey) with the surrogate's trailing
    # trajectory (model colour). One panel per
    # body so the comparison reads at a glance.
    bodies_to_show = [i for i in range(n_bodies) if i != primary_idx]
    n_bodies_show = len(bodies_to_show)
    if n_bodies_show == 0:
        return []
    n_body_cols = 4
    n_body_rows = (n_bodies_show + n_body_cols - 1) // n_body_cols  # ceil
    fig_height = 3.6 * n_body_rows + 2.4  # extra row for global error
    fig = plt.figure(figsize=(16.0, fig_height), facecolor=THEME["bg"])
    fig.suptitle(f"{preset_name}  ·  {model_name}  ·  {fmt.upper()}{galaxy_subtitle}",
                 color=THEME["text"], fontsize=14)
    # Frame-rate / sampling caption. The trajectories are integrated at a
    # finite dt and the animation samples one frame per integration step,
    # so the visible segments are the integration-sampling choice — not a
    # rendering artefact. Dash = reference (book / leapfrog), solid = pre-
    # dicted (surrogate). For Sun-centred presets the surrogate's reference
    # frame may shift because the network sees a zero-momentum state; the
    # body appears to wrap but the radial error (right panel) is invariant.
    fig.text(0.5, 0.93,
             "segments = integration-sampling (one frame per step);  "
             "dashed = from the books, solid = predicted",
             ha="center", va="top", color=THEME["text"], fontsize=9,
             alpha=0.75)
    is_3d = (view == "3d")
    gs = fig.add_gridspec(n_body_rows + 1, n_body_cols,
                          height_ratios=[3.6] * n_body_rows + [2.4],
                          hspace=0.45, wspace=0.30)
    axes = []
    for cell_i, body_i in enumerate(bodies_to_show):
        row_i = cell_i // n_body_cols
        col_i = cell_i % n_body_cols
        body_name = body_names[body_i]
        # Both panels anchored to BOOK Sun's position so the observed
        # gap reflects the surrogate's prediction error, not Sun-drift.
        book_view = book_pos[:, body_i, :] - book_pos[:, primary_idx, :]
        surr_view = surr_full[:, body_i, :3] - book_pos[:, primary_idx, :3]
        if in_galaxy:
            book_view = book_view + galaxy_disp
            surr_view = surr_view + galaxy_disp
        surr_traj = surr_view
        r_book = np.linalg.norm(book_view, axis=-1)
        r_surr = np.linalg.norm(surr_traj, axis=-1)
        err = np.abs(r_surr - r_book)
        if is_3d:
            ax_body = fig.add_subplot(gs[row_i, col_i], projection="3d")
        else:
            ax_body = fig.add_subplot(gs[row_i, col_i])
        # Combined book-vs-pred panel: book orbit (faint grey) +
        # surrogate trailing line + head markers for both. In galaxy
        # frame the primary is drawn per-frame inside _update (its
        # position moves), so the static setup draw would leave a
        # ghost primary at the frame-0 position for the whole clip.
        _make_combined_panel(ax_body, body_name, primary_name,
                             book_view, surr_traj, model_name=model_name,
                             primary_radius=primary_radius,
                             is_3d=is_3d, sun_pos=None)
        axes.append((ax_body, body_i, surr_traj, err, body_name, is_3d, book_view))
    # Global radial-error subplot spanning full width at the bottom.
    ax_err = fig.add_subplot(gs[n_body_rows, :])
    _make_global_error_panel(ax_err, axes, model_name)
    # State for the FuncAnimation.
    state = {
        "book_heads":    [],
        "surr_heads":    [],
        "surr_trails":   [],
        "book_suns":     [],
        "book_sun_artists": [],   # matplotlib artists for the redrawn
                                  # primary in galaxy-frame mode; removed
                                  # on the next frame to avoid leaks.
        "err_lines":     [],   # global error panel: one line per body
    }
    # Pre-size every per-body state list so the first frame's indexed
    # assignment (e.g. state["book_sun_artists"][cell_i] in the
    # galaxy-frame path) can never hit an empty list. The grow-then-assign
    # block at the end of the cell loop only runs AFTER that assignment.
    for _k in state:
        state[_k] = [None] * len(axes)

    def _init():
        return []

    def _update(frame_i: int):
        nonlocal trail
        artists = []
        fill, _ = _colors_for_model(model_name)
        head_marker = _surr_marker(model_name)
        for cell_i, (ax_body, body_i, surr_traj, err,
                     body_name, is_3d_c, book_view) in enumerate(axes):
            body_c = _plots_mod._body_color(body_name, body_i)
            # Remove prior frame's artists.
            for key in ("book_heads", "surr_heads", "surr_trails",
                        "book_suns", "book_sun_artists", "err_lines"):
                if cell_i < len(state[key]):
                    old = state[key][cell_i]
                    if old is not None:
                        try:
                            # Check for a plain list/tuple of artists
                            # BEFORE hasattr(old, 'remove'): list HAS a
                            # .remove() method (removes by value), so
                            # calling old.remove() here raises TypeError
                            # (missing argument), the outer except
                            # swallows it, and the artists leak — one
                            # sphere + label per body per frame, making
                            # late frames quadratic to render.
                            if isinstance(old, (list, tuple)):
                                for a in old:
                                    try:
                                        a.remove()
                                    except Exception:
                                        pass
                            elif hasattr(old, 'remove'):
                                old.remove()
                        except Exception:
                            pass
            book_now = book_view[frame_i]
            surr_now = surr_traj[frame_i]
            # Galaxy-frame: redraw the primary at its current galaxy-frame
            # position and remember the artists so the next frame's cleanup
            # pass can remove them. Without this the figure accumulates one
            # sphere/circle + label per frame (1500 frames * N cells * 2
            # artists), inflating the .mp4 and slowing the FuncAnimation.
            if in_galaxy:
                sun_pos = galaxy_disp[frame_i]
                primary_artists = _draw_primary(
                    ax_body, primary_name, primary_radius,
                    is_3d, pos=sun_pos,
                )
                state["book_sun_artists"][cell_i] = primary_artists
            else:
                sun_pos = None
            # Trailing segment (last K frames) for the surrogate.
            lo = max(0, frame_i - trail)
            seg = surr_traj[lo:frame_i + 1]
            # Combined panel: book head + surrogate trail + surrogate head.
            if is_3d:
                book_head, = ax_body.plot([book_now[0]], [book_now[1]],
                                          [book_now[2]],
                                          color=body_c, marker="o",
                                          markersize=8,
                                          markerfacecolor=body_c,
                                          markeredgecolor=THEME["text"],
                                          markeredgewidth=1.0,
                                          label="book")
                surr_trail, = ax_body.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                                           color=fill, lw=1.6, alpha=0.95,
                                           linestyle=_surr_dash(model_name))
                surr_head, = ax_body.plot([surr_now[0]], [surr_now[1]],
                                          [surr_now[2]],
                                          color=body_c, marker=head_marker,
                                          markersize=10,
                                          markeredgecolor=fill,
                                          markeredgewidth=1.2,
                                          label=f"{model_name}")
            else:
                book_head, = ax_body.plot([book_now[0]], [book_now[1]],
                                          color=body_c, marker="o",
                                          markersize=8,
                                          markerfacecolor=body_c,
                                          markeredgecolor=THEME["text"],
                                          markeredgewidth=1.0,
                                          label="book")
                surr_trail, = ax_body.plot(seg[:, 0], seg[:, 1],
                                           color=fill, lw=1.6, alpha=0.95,
                                           linestyle=_surr_dash(model_name))
                surr_head, = ax_body.plot([surr_now[0]], [surr_now[1]],
                                          color=body_c, marker=head_marker,
                                          markersize=10,
                                          markeredgecolor=fill,
                                          markeredgewidth=1.2,
                                          label=f"{model_name}")
            # Body-name label on first frame.
            if frame_i == 0:
                if is_3d:
                    ax_body.text(book_now[0], book_now[1], book_now[2],
                                 body_name, color=body_c, fontsize=8,
                                 ha="left", va="bottom", weight="bold")
                else:
                    ax_body.annotate(body_name, (book_now[0], book_now[1]),
                                     textcoords="offset points",
                                     xytext=(6, 6),
                                     color=body_c, fontsize=8,
                                     weight="bold")
            # Per-cell legend once.
            if frame_i == 0 and cell_i == 0:
                ax_body.legend(loc="upper right", fontsize=7,
                               framealpha=0.92,
                               labelcolor=THEME["text"],
                               facecolor=THEME["panel"])
            # Store.
            for key in ("book_heads", "surr_heads", "surr_trails",
                        "book_suns", "book_sun_artists", "err_lines"):
                if len(state[key]) <= cell_i:
                    state[key].append(None)
            state["book_heads"][cell_i] = book_head
            state["surr_heads"][cell_i] = surr_head
            state["surr_trails"][cell_i] = surr_trail
            state["book_suns"][cell_i] = sun_pos
            artists.extend([book_head, surr_head, surr_trail])
        # Global radial-error panel: update one curve per body.
        # Remove all prior error lines, then re-add fresh ones up to frame_i.
        for old in list(ax_err.lines):
            try:
                old.remove()
            except Exception:
                pass
        t_axis = np.arange(frame_i + 1)
        for cell_i, (ax_body, body_i, surr_traj, err,
                     body_name, is_3d_c, book_view) in enumerate(axes):
            body_c = _plots_mod._body_color(body_name, body_i)
            line, = ax_err.plot(t_axis, err[:frame_i + 1],
                                color=body_c, lw=1.4, alpha=0.95)
            artists.append(line)
        # Update legend once (per-body colour in legend).
        if frame_i == 0:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], color=_plots_mod._body_color(name, body_i),
                       lw=2, label=name)
                for body_i, name in [(b_i, body_names[b_i])
                                     for b_i in bodies_to_show]
            ]
            ax_err.legend(handles=legend_handles, loc="upper left",
                          fontsize=7, framealpha=0.92,
                          labelcolor=THEME["text"],
                          facecolor=THEME["panel"])
        # Per-frame title.
        first_row = axes[0]
        first_gap = float(np.linalg.norm(
            first_row[2][frame_i] - first_row[6][frame_i]))
        fig.suptitle(
            f"{preset_name}  ·  {model_name}  ·  frame {frame_i}/{T-1}  ·  "
            f"gap({first_row[4]}) = {first_gap:.2e} L*",
            color=THEME["text"], fontsize=13)
        return artists

    anim = FuncAnimation(fig, _update, init_func=_init,
                        frames=frames, interval=1000 / fps, blit=False,
                        repeat=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    label = f"preset_{preset_name}_{model_name}_{view}"
    if fmt in ("mp4", "both"):
        out_path = out_dir / f"{label}.mp4"
        # Resolve ffmpeg: if imageio-ffmpeg is installed, it ships the
        # binary; otherwise fall back to whatever FFMpegWriter can find.
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_path = None
        writer_kwargs = {"fps": fps, "bitrate": 2400, "codec": "h264",
                         "metadata": {"preset": preset_name,
                                      "model": model_name}}
        if ffmpeg_path:
            writer = FFMpegWriter(**writer_kwargs)
            # Patch the binary into the writer's subprocess env.
            import os
            os.environ["FFMPEG_BINARY"] = ffmpeg_path
            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        else:
            writer = FFMpegWriter(**writer_kwargs)
        # Save to a temp name first, then rename: anim.save() writes the
        # mp4 progressively, so an interrupted encode would leave a
        # partial file that size-based "already rendered" checks would
        # wrongly treat as finished. The rename is atomic once ffmpeg's
        # final flush completes (moov atom included).
        tmp_path = out_path.with_suffix(".tmp.mp4")
        anim.save(tmp_path, writer=writer)
        tmp_path.replace(out_path)
        written.append(out_path)
    if fmt in ("gif", "both"):
        out_path = out_dir / f"{label}.gif"
        writer = PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        written.append(out_path)
    plt.close(fig)
    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=None,
                   help="Preset name (e.g. jupiter_galileans). Default: all "
                        "with --dump-preds persisted.")
    p.add_argument("--model", default=None,
                   help="Model name (e.g. LSTM, LSTM_stable, ensemble_LSTM_GNN). "
                        "Default: all in preds_meta.")
    p.add_argument("--frames", type=int, default=None,
                   help="Number of frames to animate (default = full length).")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None,
                   help="Output directory. Default: real_case_validation/animations/")
    p.add_argument("--format", choices=("mp4", "gif", "both"), default="mp4")
    p.add_argument("--view", choices=("3d", "2d", "both"), default="3d")
    p.add_argument("--report-dir", default=None,
                   help="Directory with preset_<NAME>/ subdirs. Default: "
                        "the first valid report_N* under "
                        "real_case_validation/")
    p.add_argument("--trail", type=int, default=60,
                   help="Trailing-line length in frames (default 60).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    default_report_dir = repo_root / "real_case_validation" / "report_N50"
    report_dir = Path(args.report_dir) if args.report_dir else default_report_dir
    if not report_dir.exists():
        # Fall back to the numerically smallest per-N report_N{N}/ that
        # has preds (lexicographic would put N100 ahead of N25).
        candidates = sorted(
            (repo_root / "real_case_validation").glob("report_N*"),
            key=lambda p: (len(p.name), p.name),
        )
        for c in candidates:
            if any((c / f"preset_{n}").exists() for n in _discover_presets(c)):
                report_dir = c
                break
        else:
            print(f"[anim] no report dir with preds under {report_dir}",
                  file=sys.stderr)
            return 1
    out_dir = Path(args.out) if args.out else (
        repo_root / "real_case_validation" / "animations")
    out_dir.mkdir(parents=True, exist_ok=True)

    presets = [args.preset] if args.preset else _discover_presets(report_dir)
    if not presets:
        print(f"[anim] no presets with preds.npy under {report_dir}",
              file=sys.stderr)
        return 1
    total = 0
    for preset_name in presets:
        try:
            preds, book_pos, meta = _load_preset(preset_name, report_dir)
        except Exception as e:
            print(f"[anim] {preset_name}: cannot load ({e})", file=sys.stderr)
            continue
        models = [args.model] if args.model else list(meta["models"])
        for m in models:
            print(f"[anim] {preset_name} / {m}  ...", end=" ", flush=True)
            try:
                # Honor --view both by rendering 2 passes.
                views = ("3d", "2d") if args.view == "both" else (args.view,)
                all_written = []
                for v in views:
                    written = _animate_one(preset_name, m, report_dir,
                                           out_dir,
                                           view=v, frames=args.frames,
                                           fps=args.fps, fmt=args.format,
                                           trail=args.trail)
                    all_written.extend(written)
                for w in all_written:
                    print(f"{w.name}", end=" ")
                print(f"({len(all_written)} files)")
                total += len(all_written)
            except Exception as e:
                print(f"FAILED ({e})", file=sys.stderr)
                import traceback
                traceback.print_exc()
    print(f"[anim] wrote {total} files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
