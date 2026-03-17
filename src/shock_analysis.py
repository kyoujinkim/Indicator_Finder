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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap


# --------------------------------------------------------------------------- #
# 1. Event Study
# --------------------------------------------------------------------------- #

def _estimate_normal_return(price_returns: pd.Series, window: int = 60) -> float:
    """Simple market model: normal return = mean of estimation window."""
    return price_returns.mean()


def event_study(
    indicator: pd.Series,
    price: pd.Series,
    event_window: int = 5,
    estimation_window: int = 52,
    threshold_pct: float = 0.10,
) -> dict:
    """
    Identify indicator release events (large moves ≥ threshold_pct change)
    and compute average abnormal return (AAR) in ±event_window around each event.

    Returns:
      aar        : average abnormal return
      car        : cumulative abnormal return (sum of AAR)
      n_events   : number of events found
      event_dates: list of event dates
    """
    ind_z = (indicator - indicator.mean()) / indicator.std()
    price_z = price.copy()

    common = ind_z.index.intersection(price_z.index)
    ind_z = ind_z.loc[common]
    price_z = price_z.loc[common]

    # Identify events: |indicator change| >= threshold
    abs_threshold = ind_z.abs().quantile(1 - threshold_pct)
    event_dates = ind_z[ind_z.abs() >= abs_threshold].index.tolist()

    if not event_dates:
        return {"aar": np.nan, "car": np.nan, "n_events": 0, "event_dates": []}

    abnormal_returns = []
    for ev_date in event_dates:
        loc = price_z.index.get_loc(ev_date) if ev_date in price_z.index else None
        if loc is None or loc < estimation_window:
            continue

        estim = price_z.iloc[loc - estimation_window : loc]
        normal_ret = _estimate_normal_return(estim)

        start = max(0, loc - event_window)
        end = min(len(price_z), loc + event_window + 1)
        window_rets = price_z.iloc[start:end]
        ar = (window_rets - normal_ret).values
        abnormal_returns.append(ar)

    if not abnormal_returns:
        return {"aar": np.nan, "car": np.nan, "n_events": 0, "event_dates": event_dates}

    max_len = max(len(a) for a in abnormal_returns)
    padded = np.array(
        [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in abnormal_returns]
    )
    aar_series = np.nanmean(padded, axis=0)
    car = float(np.nansum(aar_series))
    aar = float(np.nanmean(aar_series))

    return {
        "aar": aar,
        "car": car,
        "n_events": len(abnormal_returns),
        "event_dates": event_dates,
    }


# --------------------------------------------------------------------------- #
# 2. SHAP Values (XGBoost time-series)
# --------------------------------------------------------------------------- #

def shap_importance(
    features: pd.DataFrame,
    target: pd.Series,
    n_estimators: int = 100,
) -> pd.Series:
    """
    Train an XGBoost regressor and compute mean |SHAP| per feature.

    Returns a Series of mean absolute SHAP values, sorted descending.
    """
    data = pd.concat([features, target.rename("__target__")], axis=1).dropna()
    if len(data) < 20:
        return pd.Series(dtype=float)

    X = data.drop(columns="__target__")
    y = data["__target__"].values

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        verbosity=0,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    )
    return mean_abs_shap.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #

def run_shock_analysis(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
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

        ev = event_study(x, target)

        rows.append(
            {
                "feature": col,
                "event_aar": ev["aar"],
                "event_car": ev["car"],
                "event_n": ev["n_events"],
            }
        )

    summary = pd.DataFrame(rows).set_index("feature")

    # SHAP on all features at once
    shap_vals = shap_importance(feature_matrix[feat_cols], target)
    summary["shap_value"] = shap_vals
    summary["shap_value"] = summary["shap_value"].fillna(0.0)

    return summary
