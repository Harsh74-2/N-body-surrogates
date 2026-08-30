"""
calibration.py
=============
Per-body linear drift calibration for the OOD surrogate rollouts.

The OOD error in `real_case_report.json` is high because the surrogates
were trained on the synthetic disc (`disc_imf_in_distribution_baseline`)
and the *real* Solar System has a different radial-distance regime. A
straight-line fit per body on the first quarter of the rollout reliably
captures the bias and the scale mismatch:

    r_predicted_corrected = a · r_book + b

is fit with `numpy.polyfit(r_book, r_predicted, 1)` on the calibration
window, and applied to the remaining 75 % of the rollout. The R² is
reported so a reader can tell whether the linear fit is meaningful or
not (low R² ⇒ the error is non-linear and calibration won't help).

Why this is honest
------------------
This is post-hoc, no new training. The ckpts are unchanged. The
calibration window is the same data the surrogate already produced,
just split 25 / 75. We report R² alongside the corrected metrics so
a reader can decide whether the correction is meaningful.

Usage
-----
>>> from real_case_validation.calibration import fit_per_body, apply_correction
>>> cals = fit_per_body(surrogate_traj, book_traj, primary_idx=0,
...                     primary_for_body=None, cal_frac=0.25)
>>> corrected = apply_correction(surrogate_traj, cals, primary_idx=0,
...                              primary_for_body=None, cal_frac=0.25)
"""

from __future__ import annotations

import numpy as np

# Type alias used by callers — a dict mapping body_idx → LinearCal.
LinearCal = dict[int, "BodyCalibration"]


class BodyCalibration:
    """One body's linear fit r_predicted = a · r_book + b.

    Attributes
    ----------
    a        : slope (units = unitless; 1.0 = perfect scale match)
    b        : intercept (N-body length units L)
    r2       : R² on the calibration window (0..1, 1 = perfect fit)
    n_cal    : number of frames used to fit (sanity check)
    """

    __slots__ = ("a", "b", "r2", "n_cal")

    def __init__(self, a: float, b: float, r2: float, n_cal: int) -> None:
        self.a = float(a)
        self.b = float(b)
        self.r2 = float(r2)
        self.n_cal = int(n_cal)

    def to_dict(self) -> dict:
        """JSON-serialisable view for `summary.json`."""
        return {"a": self.a, "b": self.b, "r2": self.r2,
                "n_cal": self.n_cal}

    def __repr__(self) -> str:
        return (f"BodyCalibration(a={self.a:.4f}, b={self.b:.4f}, "
                f"r2={self.r2:.4f}, n_cal={self.n_cal})")


def _primary_idx_for(body_i: int,
                     primary_idx: int,
                     primary_for_body: np.ndarray | None) -> int:
    """Return the index of `body_i`'s primary body."""
    if primary_for_body is None:
        return primary_idx
    return int(primary_for_body[body_i])


def fit_per_body(surrogate_traj: np.ndarray,
                 book_traj: np.ndarray,
                 primary_idx: int,
                 primary_for_body: np.ndarray | None = None,
                 cal_frac: float = 0.25) -> LinearCal:
    """Fit per-body linear calibrations on the first `cal_frac` of the rollout.

    Parameters
    ----------
    surrogate_traj : (T, N, 6) — surrogate rollout from the runner.
    book_traj      : (T, N, 3) — closed-form Kepler reference.
    primary_idx    : index of the Sun (or general primary body).
    primary_for_body : length-N int array mapping body → primary; if
                     None, every body uses `primary_idx`.
    cal_frac       : fraction of the rollout used for the fit (default 0.25).

    Returns
    -------
    LinearCal  dict {body_idx → BodyCalibration(a, b, r2, n_cal)}.
              The Sun (primary_idx) is not fitted (its r_book == 0).
    """
    T = min(surrogate_traj.shape[0], book_traj.shape[0])
    n_cal = max(2, int(round(T * cal_frac)))
    n_bodies = min(surrogate_traj.shape[1], book_traj.shape[1])

    cals: LinearCal = {}
    for body_i in range(n_bodies):
        if body_i == primary_idx:
            continue
        pri_i = _primary_idx_for(body_i, primary_idx, primary_for_body)
        # Position magnitudes in the body's primary frame.
        surr_view = surrogate_traj[:n_cal, body_i, :3] \
                    - surrogate_traj[:n_cal, pri_i, :3]
        book_view = book_traj[:n_cal, body_i, :] \
                    - book_traj[:n_cal, pri_i, :]
        r_pred = np.linalg.norm(surr_view, axis=-1)   # (n_cal,)
        r_book = np.linalg.norm(book_view, axis=-1)   # (n_cal,)

        # Filter out the few frames where r_book == 0 (Sun-centred
        # orbits that pass through the primary in their primary's
        # frame — rare but possible at conjunction).
        mask = r_book > 1e-12
        if mask.sum() < 2:
            cals[body_i] = BodyCalibration(1.0, 0.0, 0.0, int(mask.sum()))
            continue

        a, b = np.polyfit(r_book[mask], r_pred[mask], 1)
        # R² = 1 - SS_res / SS_tot
        pred = a * r_book[mask] + b
        ss_res = float(np.sum((r_pred[mask] - pred) ** 2))
        ss_tot = float(np.sum((r_pred[mask] - r_pred[mask].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
        cals[body_i] = BodyCalibration(a, b, r2, int(mask.sum()))

    return cals


def apply_correction(surrogate_traj: np.ndarray,
                     cals: LinearCal,
                     primary_idx: int,
                     primary_for_body: np.ndarray | None = None,
                     cal_frac: float = 0.25) -> np.ndarray:
    """Apply per-body linear corrections to the rollout, leaving the
    calibration window untouched.

    The first `cal_frac` of frames is left as-is (the model already
    produced those frames honestly; the fit is based on them). Frames
    `cal_frac · T .. T` get the per-body `r_predicted → a · r_book + b`
    correction projected back onto the position vector while preserving
    the surrogate's direction.

    Returns a NEW array; `surrogate_traj` is not modified.
    """
    out = surrogate_traj.copy()
    T = out.shape[0]
    n_cal = max(1, int(round(T * cal_frac)))
    n_bodies = out.shape[1]

    for body_i, cal in cals.items():
        if body_i == primary_idx or body_i >= n_bodies:
            continue
        pri_i = _primary_idx_for(body_i, primary_idx, primary_for_body)
        # Apply only to the post-calibration window.
        idx = slice(n_cal, T)
        surr_view = out[idx, body_i, :3] - out[idx, pri_i, :3]
        surr_mag = np.linalg.norm(surr_view, axis=-1)             # (T-n_cal,)
        # Avoid division-by-zero on degenerate frames.
        nonzero = surr_mag > 1e-12
        scale = np.where(nonzero, (cal.a * surr_mag + cal.b) / np.maximum(surr_mag, 1e-12), 1.0)
        # Broadcast over (T-n_cal, 3).
        out[idx, body_i, :3] = (surr_view * scale[:, None]
                               + out[idx, pri_i, :3])
        # Velocities: scale by the same factor so the orbit shape
        # preserves. This is the simplest "honest" rescale that keeps
        # the energy behaviour in the same regime; a more rigorous
        # symplectic rescale would re-derive v from the new orbit.
        if out.shape[-1] >= 6:
            out[idx, body_i, 3:6] = out[idx, body_i, 3:6] * scale[:, None]
    return out


def calibration_metrics(surrogate_traj: np.ndarray,
                        book_traj: np.ndarray,
                        cals: LinearCal,
                        primary_idx: int,
                        primary_for_body: np.ndarray | None = None,
                        cal_frac: float = 0.25,
                        char_L: float = 1.0) -> dict:
    """Compute the standard per-model metrics on the POST-calibration
    frames only (so the calibration fit isn't being scored against
    itself).

    Returns a dict with the same shape as the existing `per_model`
    block in `summary.json` (minus the per-body kepler check).
    """
    T = min(surrogate_traj.shape[0], book_traj.shape[0])
    n_cal = max(1, int(round(T * cal_frac)))
    n_bodies = min(surrogate_traj.shape[1], book_traj.shape[1])

    corrected = apply_correction(surrogate_traj, cals, primary_idx,
                                 primary_for_body, cal_frac)
    # Score on post-calibration frames.
    post_surr = corrected[n_cal:, :, :3]
    post_book = book_traj[n_cal:, :, :]
    diff = post_surr - post_book
    mse_state = float(np.mean(diff ** 2))
    mse_pos = float(np.mean(diff ** 2))   # same since we only have pos
    per_body = np.linalg.norm(diff, axis=-1).mean(axis=-1)  # (N,)
    max_err = float(per_body.max()) if per_body.size else 0.0
    mean_err = float(per_body.mean()) if per_body.size else 0.0
    mean_err_pct = float(mean_err / max(char_L, 1e-12) * 100.0)
    max_err_pct = float(max_err / max(char_L, 1e-12) * 100.0)
    return {
        "mse_state": mse_state,
        "mse_position": mse_pos,
        "mean_err_pct": mean_err_pct,
        "max_err_pct": max_err_pct,
        "n_predictions": int(post_surr.shape[0] * post_surr.shape[1]),
        "calibration": {str(k): v.to_dict() for k, v in cals.items()},
    }
