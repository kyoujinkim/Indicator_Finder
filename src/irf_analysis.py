"""
Shock & Event Analysis (Path B).

Techniques:
  1. Impulse Response Function (IRF) via VAR model
  2. Event Study  - abnormal returns around indicator release dates
  3. SHAP Values  - feature contribution at each timestep (XGBoost)
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


# --------------------------------------------------------------------------- #
# 1. Impulse Response Function
# --------------------------------------------------------------------------- #

def compute_irf(
    x: pd.Series,
    y: pd.Series,
    periods: int = 10,
    max_lags: int = 4,
) -> pd.Series:
    """
    Fit a bivariate VAR(p) model on (x, y) and compute the IRF of x on y
    for `periods` steps ahead.

    Returns a Series of IRF values indexed 0..periods.
    """
    data = pd.concat([y, x], axis=1).dropna()
    if len(data) < max_lags * 4 + periods:
        return pd.Series([np.nan] * (periods + 1), name=x.name)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(data.values)
            result = model.fit(maxlags=max_lags, ic="aic")
            irf = result.irf(periods)
            # irf.irfs shape: (periods+1, k, k)  [response, shock]
            # index 0 = y (target), index 1 = x (shock)
            irf_values = irf.irfs[:, 0, 1]  # response of y to shock in x
    except Exception:
        irf_values = np.full(periods + 1, np.nan)

    return pd.Series(irf_values, name=x.name)


def irf_persistence_score(irf_series: pd.Series) -> float:
    """
    Summarise IRF persistence as the sum of absolute values weighted by lag
    (longer-lasting shocks score higher).
    """
    vals = irf_series.dropna().values
    if len(vals) == 0:
        return 0.0
    lags = np.arange(len(vals))
    return float(np.sum(np.abs(vals) * (lags + 1)) / (len(vals) * (len(vals) + 1) / 2))

# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #

def run_irf_analysis(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
    irf_periods: int = 10,
) -> pd.DataFrame:
    """
    Run all shock analyses for every feature against the target.

    Returns a summary DataFrame with columns:
      irf_persistence, event_aar, event_car, event_n, shap_value
    """
    feat_cols = [c for c in feature_matrix.columns if c not in ("price", "eps")]
    target = feature_matrix[target_col].dropna()

    rows = []
    for col in feat_cols:
        x = feature_matrix[col].dropna()

        irf_s = compute_irf(x, target)
        persistence = irf_persistence_score(irf_s)

        rows.append(
            {
                "feature": col,
                "peak_irf": float(np.nanmax(np.abs(irf_s.fillna(0)))),
                "irf_persistence": persistence,
            }
        )

    summary = pd.DataFrame(rows).set_index("feature")

    return summary
