"""
Event Impact Analysis.

For each named macro event in data/impacts.txt:
  1. Equity Impact   - price/EPS response at pre/post-event windows
  2. Indicator Response - how each macro indicator moved after the event (z-score delta)

Recurring events (same name, multiple dates) are shown per-occurrence
and summarised as a mean across occurrences.

Format of impacts.txt:
  event_name : YYYY-MM-DD
"""

from pathlib import Path
import numpy as np
import pandas as pd

IMPACTS_FILE = Path(__file__).parent.parent / "data" / "impacts.txt"

# Post-event windows to track (weeks)
RESPONSE_WINDOWS = [1, 2, 4, 8]
# Pre-event baseline length for abnormal-return calc
BASELINE_WINDOW = 12  # weeks


# --------------------------------------------------------------------------- #
# 1. Load events
# --------------------------------------------------------------------------- #

def load_events(path: Path = IMPACTS_FILE) -> pd.DataFrame:
    """
    Parse data/impacts.txt → DataFrame[event_name, event_date].
    Lines with format "event_name : YYYY-MM-DD"; blank lines ignored.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            try:
                date = pd.to_datetime(parts[1].strip())
            except Exception:
                continue
            rows.append({"event_name": name, "event_date": date})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Equity Impact
# --------------------------------------------------------------------------- #

def _nearest_loc(index: pd.DatetimeIndex, date: pd.Timestamp, max_gap_days: int = 14) -> int | None:
    if index.empty or pd.isna(date):
        return None

    # 1. 가장 가까운 인덱스 위치를 이진 탐색 기반으로 찾음 (매우 빠름)
    loc = index.get_indexer([date], method='nearest')[0]

    # get_indexer는 찾지 못하면 -1을 반환함
    if loc == -1:
        return None

    # 2. 거리 계산 및 제약 조건 확인
    gap = abs(index[loc] - date)

    if gap > pd.Timedelta(days=max_gap_days):
        return None

    return int(loc)


def equity_event_impact(
    events: pd.DataFrame,
    target: pd.Series,
    windows: list[int] | None = None,
    baseline_window: int = BASELINE_WINDOW,
) -> pd.DataFrame:
    """
    For each event occurrence, compute equity (price/EPS) response.

    Returned columns
    ----------------
    event_name, event_date, matched_date, baseline_mean, baseline_std, event_value,
    pre_-4w, pre_-2w   : pre-event z-score vs baseline (context)
    ret_+Nw            : % return from event value at +N weeks
    ar_+Nw             : abnormal return = (post_val - baseline_mean) / baseline_std
    """
    if windows is None:
        windows = RESPONSE_WINDOWS

    rows = []
    idx = target.index

    for _, ev in events.iterrows():
        loc = _nearest_loc(idx, ev["event_date"])
        if loc is None:
            continue

        matched_date = idx[loc]
        ev_val = float(target.iloc[loc])

        bl_start = max(0, loc - baseline_window)
        baseline = target.iloc[bl_start:loc]
        bl_mean = float(baseline.mean()) if len(baseline) > 0 else np.nan
        bl_std  = float(baseline.std())  if len(baseline) > 1 else np.nan

        row: dict = {
            "event_name":    ev["event_name"],
            "event_date":    ev["event_date"],
            "matched_date":  matched_date,
            "baseline_mean": bl_mean,
            "baseline_std":  bl_std,
            "event_value":   ev_val,
        }

        # Pre-event context (z-score vs baseline)
        for pre in [4, 2]:
            pre_loc = max(0, loc - pre)
            pre_val = float(target.iloc[pre_loc])
            row[f"pre_-{pre}w"] = (
                (pre_val - bl_mean) / (bl_std + 1e-12)
                if not np.isnan(bl_std) else np.nan
            )

        # Post-event returns and abnormal returns
        for w in windows:
            post_loc = min(len(target) - 1, loc + w)
            post_val = float(target.iloc[post_loc])
            row[f"ret_+{w}w"] = (post_val - ev_val) / (abs(ev_val) + 1e-12)

            win_bl = target.pct_change(w).iloc[bl_start:loc].dropna()
            win_bl_mean = float(win_bl.mean()) if len(win_bl) > 0 else np.nan
            win_bl_std  = float(win_bl.std())  if len(win_bl) > 1 else np.nan
            win_post_val = float(target.pct_change(w).iloc[post_loc]) if post_loc < len(target) else np.nan
            row[f"ar_+{w}w"]  = (
                (win_post_val - win_bl_mean) / (win_bl_std + 1e-12)
                if not np.isnan(bl_std) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def equity_event_summary(
    equity_impact: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Group recurring events by name and compute mean response across occurrences.
    """
    if equity_impact.empty:
        return pd.DataFrame()
    if windows is None:
        windows = RESPONSE_WINDOWS

    agg_cols = (
        [f"ret_+{w}w" for w in windows]
        + [f"ar_+{w}w"  for w in windows]
        + ["pre_-4w", "pre_-2w"]
    )
    agg_cols = [c for c in agg_cols if c in equity_impact.columns]

    summary = equity_impact.groupby("event_name")[agg_cols].mean().reset_index()
    counts  = equity_impact.groupby("event_name").size().rename("n_occurrences").reset_index()
    summary = summary.merge(counts, on="event_name")
    return summary


# --------------------------------------------------------------------------- #
# 3. Indicator Response
# --------------------------------------------------------------------------- #

def indicator_event_response(
    events: pd.DataFrame,
    features: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    For each event occurrence × indicator, compute z-score change post-event.

    z-scoring uses full-series mean/std so values are comparable across events.

    Returned columns
    ----------------
    event_name, event_date, matched_date, feature, event_z,
    z_+Nw      : z-score of indicator N weeks after the event
    delta_+Nw  : z_+Nw - event_z  (change from event-date level)
    """
    if windows is None:
        windows = RESPONSE_WINDOWS

    idx        = features.index
    feat_cols  = list(features.columns)
    feat_means = features.mean()
    feat_stds  = features.std().replace(0, np.nan)

    # Resolve event locations once
    event_locs: list[int | None] = [
        _nearest_loc(idx, ev["event_date"])
        for _, ev in events.iterrows()
    ]

    rows = []
    for col in feat_cols:
        series = features[col].values
        mean_v = float(feat_means[col])
        std_v  = float(feat_stds[col]) if not np.isnan(feat_stds[col]) else np.nan

        for i, (_, ev) in enumerate(events.iterrows()):
            loc = event_locs[i]
            if loc is None:
                continue

            ev_val = float(series[loc])
            ev_z   = (ev_val - mean_v) / std_v if not np.isnan(std_v) else np.nan

            row: dict = {
                "event_name":   ev["event_name"],
                "event_date":   ev["event_date"],
                "matched_date": idx[loc],
                "feature":      col,
                "event_z":      ev_z,
            }

            for w in windows:
                post_loc  = min(len(series) - 1, loc + w)
                post_val  = float(series[post_loc])
                post_z    = (post_val - mean_v) / std_v if not np.isnan(std_v) else np.nan
                delta     = (post_z - ev_z) if not (np.isnan(post_z) or np.isnan(ev_z)) else np.nan
                row[f"z_+{w}w"]     = post_z
                row[f"delta_+{w}w"] = delta

            rows.append(row)

    return pd.DataFrame(rows)


def indicator_event_summary(
    indicator_response: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Average per-occurrence responses to get per-event-name × feature means.
    Adds a mean_abs_delta column for quick ranking.
    """
    if indicator_response.empty:
        return pd.DataFrame()
    if windows is None:
        windows = RESPONSE_WINDOWS

    agg_cols = [f"delta_+{w}w" for w in windows if f"delta_+{w}w" in indicator_response.columns]
    if not agg_cols:
        return pd.DataFrame()

    summary = (
        indicator_response
        .groupby(["event_name", "feature"])[agg_cols]
        .mean()
        .reset_index()
    )
    summary["mean_abs_delta"] = summary[agg_cols].abs().mean(axis=1)
    return summary.sort_values(["event_name", "mean_abs_delta"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# 4. Feature Event Score
# --------------------------------------------------------------------------- #

def feature_event_score(
    indicator_response: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.Series:
    """
    Score each feature by mean absolute z-score delta across all events and windows.
    Higher → indicator reacts more strongly to the defined macro events.
    """
    if indicator_response.empty:
        return pd.Series(dtype=float)
    if windows is None:
        windows = RESPONSE_WINDOWS

    delta_cols = [f"delta_+{w}w" for w in windows if f"delta_+{w}w" in indicator_response.columns]
    if not delta_cols:
        return pd.Series(dtype=float)

    tmp = indicator_response.copy()
    tmp["_mad"] = tmp[delta_cols].abs().mean(axis=1)
    return tmp.groupby("feature")["_mad"].mean().sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 5. Batch Runner
# --------------------------------------------------------------------------- #

def run_event_impact(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
) -> dict:
    """
    Run full event impact analysis.

    Returns
    -------
    dict with keys:
      events             : raw events DataFrame
      equity_impact      : per-occurrence equity response
      equity_summary     : per-event-name mean equity response
      indicator_response : long-form per-occurrence × per-feature z-score response
      indicator_summary  : per-event-name × per-feature mean response
      feature_score      : Series(feature → event-responsiveness score)
    """
    empty = {
        "events":             pd.DataFrame(),
        "equity_impact":      pd.DataFrame(),
        "equity_summary":     pd.DataFrame(),
        "indicator_response": pd.DataFrame(),
        "indicator_summary":  pd.DataFrame(),
        "feature_score":      pd.Series(dtype=float),
    }

    if not IMPACTS_FILE.exists():
        return empty

    events = load_events()
    if events.empty:
        return {**empty, "events": events}

    target    = feature_matrix[target_col].dropna()
    feat_cols = [c for c in feature_matrix.columns if c not in ("target", target_col)]
    features  = feature_matrix[feat_cols].copy()

    eq_impact  = equity_event_impact(events, target)
    eq_summary = equity_event_summary(eq_impact)

    ind_resp    = indicator_event_response(events, features)
    ind_summary = indicator_event_summary(ind_resp)
    feat_score  = feature_event_score(ind_resp)

    return {
        "events":             events,
        "equity_impact":      eq_impact,
        "equity_summary":     eq_summary,
        "indicator_response": ind_resp,
        "indicator_summary":  ind_summary,
        "feature_score":      feat_score,
    }