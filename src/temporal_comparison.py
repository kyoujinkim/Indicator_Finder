"""
Temporal Comparison — measures how indicator importance changed over time.

Strategy: rolling-window comparison.
  Current window : [today - window_weeks,  today]
  Past window    : [today - window_weeks - shift_weeks,  today - shift_weeks]

Both windows are the same length so scores are directly comparable.
The difference in final_score / rank between windows isolates the
marginal contribution of the most recent `shift_weeks` of data.

Public API
----------
  run_pipeline(feature_matrix, target_col) -> pd.DataFrame
      Run all four analysis paths on a single feature matrix slice.

  compare_periods(feature_matrix, target_col, window_weeks, shift_weeks)
      -> dict with keys: "current", "past", "delta"
"""

import warnings
import numpy as np
import pandas as pd

from .continuous_analysis import run_continuous_analysis
from .irf_analysis import run_irf_analysis
from .shock_analysis import run_shock_analysis
from .threshold_analysis import run_threshold_analysis
from .feature_selector import run_feature_selection
from .relation_scorer import compute_relation_scores


# --------------------------------------------------------------------------- #
# Pipeline runner for a single window
# --------------------------------------------------------------------------- #

_SCORE_COLS = [
    "rank", "final_score",
    "continuous_score", "irf_score", "shock_score", "threshold_score",
]

def run_pipeline(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
) -> pd.DataFrame:
    """
    Run all four analysis paths on the given feature matrix slice and
    return the final scored/ranked DataFrame.
    """
    feat_cols = [c for c in feature_matrix.columns if c not in ("price", "eps")]
    if len(feat_cols) == 0 or len(feature_matrix) < 15:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        continuous  = run_continuous_analysis(feature_matrix, target_col=target_col)
        irf         = run_irf_analysis(feature_matrix, target_col=target_col)
        shock       = run_shock_analysis(feature_matrix, target_col=target_col)
        threshold   = run_threshold_analysis(feature_matrix, target_col=target_col)
        fs          = run_feature_selection(feature_matrix, target_col=target_col)
        scores      = compute_relation_scores(continuous, irf, shock, threshold, fs)

    return scores


# --------------------------------------------------------------------------- #
# Period comparison
# --------------------------------------------------------------------------- #

def compare_periods(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
    window_weeks: int = 52,
    shift_weeks: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Compare indicator importance between two rolling windows.

    Parameters
    ----------
    feature_matrix : full aligned feature matrix from build_feature_matrix()
    target_col     : 'price' or 'eps'
    window_weeks   : length of each analysis window in weeks
    shift_weeks    : how many weeks to shift back for the "past" window
                     (4 ≈ 1 month, 13 ≈ 1 quarter)
    verbose        : print progress

    Returns
    -------
    dict with keys:
        "current"     : scored DataFrame for the current window
        "past"        : scored DataFrame for the past window
        "delta"       : comparison DataFrame (one row per feature), columns:
                            rank_current, rank_past, rank_delta (+= rose),
                            score_current, score_past, score_delta,
                            cont_delta, irf_delta, shock_delta, thresh_delta,
                            dominant_change  (which path changed most),
                            status           (risen / fallen / stable / new / dropped)
        "window_current": (start, end) dates
        "window_past"   : (start, end) dates
    """
    idx = feature_matrix.index
    current_end   = idx.max()
    current_start = current_end - pd.DateOffset(weeks=window_weeks)
    past_end      = current_end - pd.DateOffset(weeks=shift_weeks)
    past_start    = past_end   - pd.DateOffset(weeks=window_weeks)

    fm_current = feature_matrix.loc[
        (idx >= current_start) & (idx <= current_end)
    ]
    fm_past = feature_matrix.loc[
        (idx >= past_start) & (idx <= past_end)
    ]

    if verbose:
        print(f"  Current window : {fm_current.index.min().date()} → {fm_current.index.max().date()} ({len(fm_current)} rows)")
        print(f"  Past window    : {fm_past.index.min().date()} → {fm_past.index.max().date()} ({len(fm_past)} rows)")

    if verbose:
        print("  Running pipeline on current window...")
    scores_current = run_pipeline(fm_current, target_col=target_col)

    if verbose:
        print("  Running pipeline on past window...")
    scores_past = run_pipeline(fm_past, target_col=target_col)

    delta = _build_delta(scores_current, scores_past)

    return {
        "current": scores_current,
        "past": scores_past,
        "delta": delta,
        "window_current": (fm_current.index.min(), fm_current.index.max()),
        "window_past": (fm_past.index.min(), fm_past.index.max()),
    }


def _build_delta(
    current: pd.DataFrame,
    past: pd.DataFrame,
) -> pd.DataFrame:
    """Merge current and past scores, compute deltas and classify changes."""
    if current.empty or past.empty:
        return pd.DataFrame()

    # Keep only scoring columns to avoid conflicts on the many raw-metric cols
    c = current[_SCORE_COLS].copy()
    p = past[_SCORE_COLS].copy()

    merged = c.join(p, how="outer", lsuffix="_current", rsuffix="_past")

    # Score and rank deltas
    merged["score_delta"] = merged["final_score_current"] - merged["final_score_past"]
    # Positive rank_delta = feature moved up (lower rank number = more important)
    merged["rank_delta"]  = merged["rank_past"].fillna(999) - merged["rank_current"].fillna(999)

    # Per-path deltas
    for path in ("continuous", "irf", "shock", "threshold"):
        cur_col  = f"{path}_score_current"
        past_col = f"{path}_score_past"
        merged[f"{path}_delta"] = merged[cur_col].fillna(0) - merged[past_col].fillna(0)

    # Which path changed most in absolute terms
    path_delta_cols = ["continuous_delta", "irf_delta", "shock_delta", "threshold_delta"]
    path_labels     = ["Continuous", "IRF", "Shock", "Threshold"]

    def _dominant(row):
        vals = [abs(row.get(c, 0) or 0) for c in path_delta_cols]
        return path_labels[int(np.argmax(vals))]

    merged["dominant_change"] = merged.apply(_dominant, axis=1)

    # Status classification
    def _status(row):
        if pd.isna(row.get("final_score_past")):
            return "new"
        if pd.isna(row.get("final_score_current")):
            return "dropped"
        delta = row["score_delta"]
        if delta > 0.05:
            return "risen"
        if delta < -0.05:
            return "fallen"
        return "stable"

    merged["status"] = merged.apply(_status, axis=1)

    return merged.sort_values("rank_current").reset_index()
