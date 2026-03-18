"""
Feature Selection (multi-stage filtering).

A. Statistical & Causal Filtering
   - Mutual Information (sklearn)
   - Transfer Entropy (from continuous_analysis)

B. Iterative Selection
   - Boruta Algorithm (boruta package)
   - Regime-Aware Selection (HMM-style regime detection + per-regime importance)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    from boruta import BorutaPy
    _BORUTA_AVAILABLE = True
except ImportError:
    _BORUTA_AVAILABLE = False

from .continuous_analysis import transfer_entropy


# --------------------------------------------------------------------------- #
# A. Statistical & Causal Filtering
# --------------------------------------------------------------------------- #

def mutual_information_scores(
    features: pd.DataFrame,
    target: pd.Series,
    n_neighbors: int = 5,
) -> pd.Series:
    """
    Compute Mutual Information between each feature and the target.
    Returns a Series sorted descending.
    """
    data = pd.concat([features, target.rename("__target__")], axis=1).dropna()
    if len(data) < 20:
        return pd.Series(dtype=float)

    X = data.drop(columns="__target__").fillna(0)
    y = data["__target__"].values

    mi = mutual_info_regression(X.values, y, n_neighbors=n_neighbors, random_state=42)
    result = pd.Series(mi, index=X.columns)
    return result.sort_values(ascending=False)


def transfer_entropy_scores(
    features: pd.DataFrame,
    target: pd.Series,
    lag: int = 1,
) -> pd.Series:
    """
    Compute Transfer Entropy from each feature to the target.
    Returns a Series sorted descending.
    """
    scores = {}
    for col in features.columns:
        x = features[col].dropna()
        te = transfer_entropy(x, target, lag=lag)
        scores[col] = te
    result = pd.Series(scores)
    return result.sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# B. Iterative Selection
# --------------------------------------------------------------------------- #

def boruta_selection(
    features: pd.DataFrame,
    target: pd.Series,
    n_estimators: int = 100,
    max_iter: int = 50,
    alpha: float = 0.05,
) -> dict:
    """
    Run Boruta feature selection.

    Returns:
      accepted  : list of confirmed-important features
      tentative : list of tentative features
      rejected  : list of rejected features
    """
    if not _BORUTA_AVAILABLE:
        return {"accepted": [], "tentative": [], "rejected": list(features.columns), "error": "boruta not installed"}

    data = pd.concat([features, target.rename("__target__")], axis=1).dropna()
    if len(data) < 30:
        return {"accepted": [], "tentative": [], "rejected": list(features.columns)}

    X = data.drop(columns="__target__").fillna(0).values
    y = data["__target__"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, max_depth=7, random_state=42
    )
    selector = BorutaPy(rf, n_estimators="auto", alpha=alpha, max_iter=max_iter, random_state=42, verbose=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X_scaled, y)

    col_names = data.drop(columns="__target__").columns.tolist()
    accepted = [col_names[i] for i, s in enumerate(selector.support_) if s]
    tentative = [col_names[i] for i, s in enumerate(selector.support_weak_) if s]
    rejected = [c for c in col_names if c not in accepted and c not in tentative]

    return {"accepted": accepted, "tentative": tentative, "rejected": rejected}


def detect_regimes(target: pd.Series, n_regimes: int = 3, method: str = "simple") -> pd.Series:
    """
    Simple regime detection using rolling z-score of price returns.
    Labels: 0 = bear, 1 = neutral, 2 = bull.
    """
    if method == "simple":
        tgt = target.dropna()
        roll_mean = tgt.mean()
        roll_std = tgt.std()

        if roll_std == 0 or np.isnan(roll_std):
            return pd.Series(1, index=tgt.index, name="regime")  # all neutral if no variation
        z = (tgt - roll_mean) / roll_std

        bins = np.linspace(-2, 2, n_regimes - 1)
        labels = np.digitize(z.fillna(0).values, bins)
        return pd.Series(labels, index=z.index, name="regime")
    elif method == "balanced":
        tgt = target.dropna()
        quantiles = np.quantile(tgt, [0.33, 0.66])
        labels = np.digitize(tgt.values, quantiles)
        return pd.Series(labels, index=tgt.index, name="regime")
    else:
        raise ValueError(f"Unknown regime detection method: {method}")



def regime_aware_importance(
    features: pd.DataFrame,
    target: pd.Series,
    n_estimators: int = 50,
) -> pd.DataFrame:
    """
    Compute feature importance separately per market regime.

    Returns a DataFrame with one column per regime and rows = features.
    """
    regimes = detect_regimes(target, method="balanced")
    regime_values = sorted(regimes.unique())

    result = {}
    for regime in regime_values:
        regime_dates = regimes[regimes == regime].index
        feat_r = features.loc[features.index.intersection(regime_dates)]
        target_r = target.loc[target.index.intersection(regime_dates)]

        data = pd.concat([feat_r, target_r.rename("__target__")], axis=1).dropna()
        if len(data) < 15:
            continue

        X = data.drop(columns="__target__").fillna(0).values
        y = data["__target__"].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1, max_depth=5, random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf.fit(X_scaled, y)

        importances = pd.Series(
            rf.feature_importances_,
            index=data.drop(columns="__target__").columns,
        )
        result[f"regime_{regime}"] = importances

    return pd.DataFrame(result).fillna(0.0)


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #

def run_feature_selection(
    feature_matrix: pd.DataFrame,
    target_col: str = "price",
) -> dict:
    """
    Run full feature selection pipeline.

    Returns a dict with keys:
      mi_scores         : pd.Series
      te_scores         : pd.Series
      boruta            : dict (accepted/tentative/rejected)
      regime_importance : pd.DataFrame
    """
    feat_cols = [c for c in feature_matrix.columns if c not in ("target", target_col)]
    features = feature_matrix[feat_cols].fillna(0)
    target = feature_matrix["target"].dropna()

    mi = mutual_information_scores(features, target)
    te = transfer_entropy_scores(features, target)
    boruta = boruta_selection(features, target)
    regime = regime_aware_importance(features, target)

    return {
        "mi_scores": mi,
        "te_scores": te,
        "boruta": boruta,
        "regime_importance": regime,
    }
