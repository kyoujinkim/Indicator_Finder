"""
Threshold (Indicator Function) Analysis - Path C.

Some indicators act as threshold functions: they trigger a discrete, directional
impact on the target only when they cross a critical level (e.g. PMI crossing 50,
yield curve inversion, sentiment entering extreme territory).

This module:
  1. Optimal Threshold Search  - find the level T that maximises signal quality
                                  via grid search over the indicator's empirical
                                  distribution (quantile sweep).
  2. Crossing Detection        - identify up-crossings and down-crossings of T.
  3. Conditional Forward Returns - measure target returns at fixed horizons
                                  after each crossing event.
  4. Signal Quality Metrics    - hit-rate, average forward return, information
                                  ratio, and directional asymmetry (up vs. down).
  5. Batch Runner              - iterate over all features and return a summary.
"""

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# 1. Optimal Threshold Search
# --------------------------------------------------------------------------- #

def find_optimal_threshold(
    x: pd.Series,
    y: pd.Series,
    n_quantiles: int = 20,
    horizon: int = 1,
) -> dict:
    """
    Grid-search for the threshold level T in x that maximises the absolute
    difference in mean forward returns of y between the two regimes:
        above-regime : x_t >= T
        below-regime : x_t <  T

    Parameters
    ----------
    x          : indicator series
    y          : target return series (already stationary)
    n_quantiles: number of candidate threshold levels to evaluate
    horizon    : forward-return look-ahead in periods

    Returns
    -------
    dict with keys:
        threshold   : optimal threshold value
        quantile    : quantile of that threshold in the indicator distribution
        regime_diff : mean(y | x>=T) - mean(y | x<T)
        pvalue      : Welch t-test p-value for regime separation
        above_mean  : mean forward return when x >= T
        below_mean  : mean forward return when x < T
    """
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(data) < n_quantiles * 2:
        return _empty_threshold()

    y_fwd = data["y"].shift(-horizon)
    combined = pd.concat([data["x"], y_fwd.rename("y_fwd")], axis=1).dropna()
    if len(combined) < 10:
        return _empty_threshold()

    quantile_levels = np.linspace(0.1, 0.9, n_quantiles)
    thresholds = combined["x"].quantile(quantile_levels).values

    best = None
    best_diff = -np.inf

    for T in thresholds:
        above = combined.loc[combined["x"] >= T, "y_fwd"]
        below = combined.loc[combined["x"] < T, "y_fwd"]
        if len(above) < 5 or len(below) < 5:
            continue
        diff = abs(above.mean() - below.mean())
        if diff > best_diff:
            best_diff = diff
            best = (T, above, below)

    if best is None:
        return _empty_threshold()

    T_opt, above, below = best
    try:
        _, pval = stats.ttest_ind(above, below, equal_var=False)
    except Exception:
        pval = 1.0

    q = float(stats.percentileofscore(combined["x"].values, T_opt) / 100)

    return {
        "threshold": float(T_opt),
        "quantile": q,
        "regime_diff": float(above.mean() - below.mean()),
        "pvalue": float(pval),
        "above_mean": float(above.mean()),
        "below_mean": float(below.mean()),
    }


def _empty_threshold() -> dict:
    return {
        "threshold": np.nan,
        "quantile": np.nan,
        "regime_diff": 0.0,
        "pvalue": 1.0,
        "above_mean": np.nan,
        "below_mean": np.nan,
    }


# --------------------------------------------------------------------------- #
# 2. Crossing Detection
# --------------------------------------------------------------------------- #

def detect_crossings(
    x: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """
    Detect up-crossings (x crosses above threshold) and
    down-crossings (x crosses below threshold).

    Returns a DataFrame with columns:
        date      : crossing date
        direction : +1 (up-crossing) or -1 (down-crossing)
    """
    above = (x >= threshold).astype(int)
    diff = above.diff()
    up_dates = x.index[diff == 1].tolist()
    down_dates = x.index[diff == -1].tolist()

    rows = (
        [{"date": d, "direction": 1} for d in up_dates]
        + [{"date": d, "direction": -1} for d in down_dates]
    )
    if not rows:
        return pd.DataFrame(columns=["date", "direction"])

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. Conditional Forward Returns
# --------------------------------------------------------------------------- #

def crossing_forward_returns(
    crossings: pd.DataFrame,
    price_returns: pd.Series,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute target forward returns at each horizon after each crossing.

    Parameters
    ----------
    crossings     : output of detect_crossings()
    price_returns : stationary target return series
    horizons      : list of forward-look periods (default [1, 3, 5, 10])

    Returns
    -------
    DataFrame indexed by crossing date with columns:
        direction, fwd_1, fwd_3, fwd_5, fwd_10 (or whichever horizons given)
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    if crossings.empty:
        return pd.DataFrame()

    records = []
    pr_index = price_returns.index

    for _, row in crossings.iterrows():
        ev_date = row["date"]
        direction = row["direction"]
        if ev_date not in pr_index:
            continue
        loc = pr_index.get_loc(ev_date)
        rec = {"date": ev_date, "direction": direction}
        for h in horizons:
            end = min(loc + h + 1, len(pr_index))
            if end > loc + 1:
                fwd = price_returns.iloc[loc + 1 : end].sum()
                rec[f"fwd_{h}"] = float(fwd)
            else:
                rec[f"fwd_{h}"] = np.nan
        records.append(rec)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("date")
    return result


# --------------------------------------------------------------------------- #
# 4. Signal Quality Metrics
# --------------------------------------------------------------------------- #

def signal_quality(
    fwd_returns: pd.DataFrame,
    horizons: list[int] | None = None,
) -> dict:
    """
    Summarize signal quality across crossing events.

    Returns a dict with per-horizon metrics plus aggregate scores:
        hit_rate_{h}    : fraction of crossings where direction × fwd_return > 0
        avg_return_{h}  : mean of direction-signed forward return
        info_ratio_{h}  : avg_return / std(signed forward return)
        asymmetry       : difference in hit-rate between up- and down-crossings
        trigger_score   : composite signal quality score [0, 1]
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    if fwd_returns.empty or "direction" not in fwd_returns.columns:
        return _empty_quality(horizons)

    result = {}
    info_ratios = []

    for h in horizons:
        col = f"fwd_{h}"
        if col not in fwd_returns.columns:
            continue
        sub = fwd_returns[["direction", col]].dropna()
        if len(sub) < 3:
            continue

        signed = sub["direction"] * sub[col]
        hit = float((signed > 0).mean())
        avg = float(signed.mean())
        std = float(signed.std())
        ir = avg / std if std > 0 else 0.0

        result[f"hit_rate_{h}"] = hit
        result[f"avg_return_{h}"] = avg
        result[f"info_ratio_{h}"] = ir
        info_ratios.append(ir)

    # Directional asymmetry (up-cross vs down-cross hit rate at h=5 or best)
    best_col = f"fwd_{horizons[len(horizons) // 2]}"
    if best_col in fwd_returns.columns:
        sub = fwd_returns[["direction", best_col]].dropna()
        up = sub[sub["direction"] == 1]
        dn = sub[sub["direction"] == -1]
        up_hr = float((up["direction"] * up[best_col] > 0).mean()) if len(up) > 2 else 0.5
        dn_hr = float((dn["direction"] * dn[best_col] > 0).mean()) if len(dn) > 2 else 0.5
        result["asymmetry"] = abs(up_hr - dn_hr)
    else:
        result["asymmetry"] = 0.0

    # Composite trigger score: mean of normalized info-ratios capped at [-2, 2]
    if info_ratios:
        clipped = np.clip(info_ratios, -2, 2)
        # Map [-2, 2] -> [0, 1]
        trigger_score = float(np.mean((clipped + 2) / 4))
    else:
        trigger_score = 0.0

    result["trigger_score"] = trigger_score
    result["n_crossings"] = int(len(fwd_returns))
    return result


def _empty_quality(horizons: list[int]) -> dict:
    result = {f"hit_rate_{h}": np.nan for h in horizons}
    result.update({f"avg_return_{h}": np.nan for h in horizons})
    result.update({f"info_ratio_{h}": np.nan for h in horizons})
    result["asymmetry"] = 0.0
    result["trigger_score"] = 0.0
    result["n_crossings"] = 0
    return result


# --------------------------------------------------------------------------- #
# 5. Batch Runner
# --------------------------------------------------------------------------- #

def run_threshold_analysis(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
    horizons: list[int] | None = None,
    n_quantiles: int = 20,
) -> pd.DataFrame:
    """
    Run threshold (indicator function) analysis for every feature.

    Returns a summary DataFrame with one row per feature and columns:
        threshold, threshold_quantile, regime_diff, threshold_pval,
        trigger_score, n_crossings, hit_rate_5, avg_return_5, info_ratio_5,
        asymmetry
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    feat_cols = [c for c in feature_matrix.columns if c not in ("target", target_col)]
    target = feature_matrix["target"].dropna()

    rows = []
    for col in feat_cols:
        x = feature_matrix[col].dropna()
        x_mean = x.mean()
        x_std = x.std()
        x_z = (x - x_mean) / x_std if x_std > 0 else x - x_mean
        common = x_z.index.intersection(target.index)
        x_c = x_z.loc[common]
        y_c = target.loc[common]

        th_info = find_optimal_threshold(x_c, y_c, n_quantiles=n_quantiles)

        if np.isnan(th_info["threshold"]):
            sq = _empty_quality(horizons)
        else:
            crossings = detect_crossings(x_c, th_info["threshold"])
            fwd = crossing_forward_returns(crossings, y_c, horizons=horizons)
            sq = signal_quality(fwd, horizons=horizons)

        # un-standardize threshold for reporting
        if x.std() > 0 and not np.isnan(th_info["threshold"]):
            th_info["threshold"] = th_info["threshold"] * x_std + x_mean
            th_info["regime_diff"] = th_info["regime_diff"] * x_std  # scale back to original units
            th_info["above_mean"] = th_info["above_mean"] * x_std + x_mean
            th_info["below_mean"] = th_info["below_mean"] * x_std + x_mean

        row = {"feature": col}
        row.update({
            "threshold": th_info["threshold"],
            "threshold_quantile": th_info["quantile"],
            "regime_diff": th_info["regime_diff"],
            "threshold_pval": th_info["pvalue"],
        })
        row.update(sq)
        rows.append(row)

    return pd.DataFrame(rows).set_index("feature")
