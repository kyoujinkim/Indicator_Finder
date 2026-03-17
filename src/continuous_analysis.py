"""
Continuous Impact Analysis (Path A).

Techniques:
  1. Granger Causality   - does indicator X Granger-cause target Y?
  2. Transfer Entropy    - directional information flow X -> Y
  3. ElasticNet          - sparse regression for variable selection
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# 1. Granger Causality
# --------------------------------------------------------------------------- #

def granger_causality_score(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 6,
    significance: float = 0.05,
) -> dict:
    """
    Test whether x Granger-causes y.

    Returns a dict with:
      best_lag   : lag with lowest p-value
      min_pval   : minimum p-value across lags
      significant: bool
      f_stats    : dict of {lag: f_stat}
    """
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) < max_lag * 3 + 10:
        return {"best_lag": None, "min_pval": 1.0, "significant": False, "f_stats": {}}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
    except Exception:
        return {"best_lag": None, "min_pval": 1.0, "significant": False, "f_stats": {}}

    f_stats = {}
    pvals = {}
    for lag, res in results.items():
        pval = res[0]["ssr_ftest"][1]
        fstat = res[0]["ssr_ftest"][0]
        pvals[lag] = pval
        f_stats[lag] = fstat

    best_lag = min(pvals, key=pvals.get)
    min_pval = pvals[best_lag]
    return {
        "best_lag": best_lag,
        "min_pval": min_pval,
        "significant": min_pval < significance,
        "f_stats": f_stats,
    }


# --------------------------------------------------------------------------- #
# 2. Transfer Entropy (binning estimator)
# --------------------------------------------------------------------------- #

def _entropy(pk: np.ndarray) -> float:
    """Shannon entropy from a probability array."""
    pk = pk[pk > 0]
    return -np.sum(pk * np.log2(pk))


def transfer_entropy(
    x: pd.Series,
    y: pd.Series,
    lag: int = 1,
    bins: int = 6,
) -> float:
    """
    Estimate Transfer Entropy TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

    Uses histogram binning. Returns TE in bits (>0 means X informs Y).
    """
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < lag + bins * 2:
        return 0.0

    x_vals = data.iloc[:, 0].values
    y_vals = data.iloc[:, 1].values

    y_t = y_vals[lag:]
    y_lag = y_vals[:-lag]
    x_lag = x_vals[:-lag]

    # Discretize
    def _digitize(arr: np.ndarray) -> np.ndarray:
        edges = np.percentile(arr, np.linspace(0, 100, bins + 1))
        edges = np.unique(edges)
        return np.digitize(arr, edges[1:-1])

    y_t_d = _digitize(y_t)
    y_lag_d = _digitize(y_lag)
    x_lag_d = _digitize(x_lag)

    n = len(y_t_d)

    # H(Y_t | Y_{t-1})
    joint_yy = np.zeros((bins, bins))
    for yt, yl in zip(y_t_d, y_lag_d):
        joint_yy[yt - 1, yl - 1] += 1
    joint_yy /= n
    marg_yl = joint_yy.sum(axis=0)

    h_yt_given_yl = 0.0
    for yl_idx in range(bins):
        if marg_yl[yl_idx] > 0:
            cond = joint_yy[:, yl_idx] / marg_yl[yl_idx]
            h_yt_given_yl += marg_yl[yl_idx] * _entropy(cond)

    # H(Y_t | Y_{t-1}, X_{t-lag})
    joint_yyx = np.zeros((bins, bins, bins))
    for yt, yl, xl in zip(y_t_d, y_lag_d, x_lag_d):
        joint_yyx[yt - 1, yl - 1, xl - 1] += 1
    joint_yyx /= n
    marg_ylxl = joint_yyx.sum(axis=0)

    h_yt_given_yl_xl = 0.0
    for yl_idx in range(bins):
        for xl_idx in range(bins):
            if marg_ylxl[yl_idx, xl_idx] > 0:
                cond = joint_yyx[:, yl_idx, xl_idx] / marg_ylxl[yl_idx, xl_idx]
                h_yt_given_yl_xl += marg_ylxl[yl_idx, xl_idx] * _entropy(cond)

    return max(0.0, h_yt_given_yl - h_yt_given_yl_xl)


# --------------------------------------------------------------------------- #
# 3. ElasticNet Regression
# --------------------------------------------------------------------------- #

def elasticnet_importance(
    features: pd.DataFrame,
    target: pd.Series,
    cv: int = 5,
) -> pd.Series:
    """
    Fit an ElasticNetCV model and return feature coefficients as importance scores.

    Returns a Series of absolute coefficients, sorted descending.
    """
    data = pd.concat([features, target.rename("__target__")], axis=1).dropna()
    if len(data) < cv * 3 or data.shape[1] < 2:
        return pd.Series(dtype=float)

    X = data.drop(columns="__target__").values
    y = data["__target__"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(cv=cv, max_iter=10000, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    coef = pd.Series(
        np.abs(model.coef_),
        index=data.drop(columns="__target__").columns,
    )
    return coef.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #

def run_continuous_analysis(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Run all continuous-impact analyses for every feature against the target.

    Returns a summary DataFrame with one row per feature and columns:
      granger_pval, granger_lag, granger_sig, transfer_entropy, elasticnet_coef
    """
    feat_cols = [c for c in feature_matrix.columns if c not in ("price", "eps")]
    target = feature_matrix[target_col].dropna()

    rows = []
    for col in feat_cols:
        x = feature_matrix[col].dropna()

        gc = granger_causality_score(x, target, max_lag=max_lag)
        te = transfer_entropy(x, target, lag=gc["best_lag"] or 1)

        rows.append(
            {
                "feature": col,
                "granger_pval": gc["min_pval"],
                "granger_lag": gc["best_lag"],
                "granger_sig": gc["significant"],
                "transfer_entropy": te,
            }
        )

    summary = pd.DataFrame(rows).set_index("feature")

    # ElasticNet on all features at once
    en_coef = elasticnet_importance(feature_matrix[feat_cols], target)
    summary["elasticnet_coef"] = en_coef
    summary["elasticnet_coef"] = summary["elasticnet_coef"].fillna(0.0)

    return summary
