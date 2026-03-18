"""
Preprocessing Engine for MFMEA.

Aligns multi-frequency economic indicator data to a common daily index
using two strategies:
  - step_fill: forward-fill lower-frequency values to match high-frequency dates
  - midas: Mixed-Data Sampling weighting (Almon polynomial weights)

Also computes log-returns / YoY changes to make series stationary where applicable.
"""
import re

import numpy as np
import pandas as pd
from typing import Literal

from dateutil.relativedelta import relativedelta


def align_to_equity(
    indicator: pd.Series,
    equity_index: pd.DatetimeIndex,
    method: Literal["step_fill", "midas"] = "step_fill",
    midas_lags: int = 12,
) -> pd.Series:
    """
    Aligns a single indicator series to the equity daily index.

    Parameters
    ----------
    indicator : pd.Series  (any frequency)
    equity_index : pd.DatetimeIndex  (daily)
    method : 'step_fill' or 'midas'
    midas_lags : number of lags for MIDAS (used only when method='midas')

    Returns
    -------
    pd.Series aligned to equity_index
    """
    if method == "step_fill":
        return _step_fill(indicator, equity_index)
    elif method == "midas":
        return _midas(indicator, equity_index, lags=midas_lags)
    else:
        raise ValueError(f"Unknown method: {method}")


def _step_fill(indicator: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill lower-frequency data onto a higher-frequency index."""
    combined = indicator.reindex(indicator.index.union(target_index))
    combined = combined.sort_index().ffill()
    return combined.reindex(target_index)


def _midas(
    indicator: pd.Series, target_index: pd.DatetimeIndex, lags: int = 12
) -> pd.Series:
    """
    Simplified MIDAS: build a weighted average of the last `lags` observations
    of the low-frequency indicator using Almon polynomial weights (beta distribution).
    Returns a daily series where each day's value is the MIDAS-weighted index value.
    """
    # Use exponentially decaying weights as a simple Almon approximation
    weights = np.exp(-np.arange(lags) * 0.3)
    weights /= weights.sum()

    step = _step_fill(indicator, target_index)
    values = step.values.astype(float)
    out = np.empty(len(values))
    out[:] = np.nan

    for i in range(lags - 1, len(values)):
        window = values[i - lags + 1 : i + 1][::-1]  # most recent first
        if not np.any(np.isnan(window)):
            out[i] = np.dot(weights, window)

    return pd.Series(out, index=target_index, name=indicator.name)


def make_stationary(series: pd.Series, method: str = "log_return") -> pd.Series:
    """
    Transform a price-like series to stationary.

    method:
      'log_return'  : log(s_t / s_{t-1})
      'pct_change'  : (s_t - s_{t-1}) / s_{t-1}
      'diff'        : s_t - s_{t-1}
      'yoy'         : (s_t - s_{t-12}) / s_{t-12}  (monthly)
    """
    s = series.replace(0, np.nan)
    if method == "log_return":
        return np.log(s / s.shift(1))
    elif method == "pct_change":
        return s.pct_change()
    elif method == "diff":
        return s.diff()
    elif method == "yoy":
        return s.pct_change(12)
    elif method == 'base100':
        return s - 100
    else:
        raise ValueError(f"Unknown method: {method}")


def build_feature_matrix(
    target: str,
    indicators: dict[str, pd.DataFrame],
    equity: pd.DataFrame,
    frequency: str = 'w',
    align_method: Literal["step_fill", "midas"] = "step_fill",
    stationary: bool = True,
    threshold: bool = False,
) -> pd.DataFrame:
    """
    Build a single aligned feature matrix (columns = indicators, rows = equity dates).
    Optionally applies stationarity transformation on each series.

    Returns a DataFrame with equity 'price' and 'eps' as last two columns.
    """
    eq = equity.copy()
    eq = eq.resample(frequency).last()
    target_index = eq.index
    frames: list[pd.Series] = []

    for sheet_name, df in indicators.items():
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 5:
                continue
            # Find Date Indexers and Numbers for naming
            date_indexer = re.search(r"\{(.*?)\}", col)
            if date_indexer:
                numbers = re.findall(r"\d+", date_indexer.group())
                series.index = series.index - pd.offsets.MonthBegin() + pd.DateOffset(months=int(numbers[0]), day=int(numbers[1]))

            aligned = align_to_equity(series, target_index, method=align_method)
            # Add Caption
            if '[1d]' in col:
                col += "(차분)"
            elif '[1%]' in col:
                col += "(변동률)"
            elif '[100]' in col:
                col += "(100기준)"
            else:
                pass
            # Determine stationarity method based on column name
            if stationary:
                if '[1d]' in col:
                    aligned = make_stationary(aligned, method="diff")
                elif '[1%]' in col:
                    aligned = make_stationary(aligned, method="pct_change")
                elif '[100]' in col:
                    aligned = make_stationary(aligned, method="base100")
                else:
                    pass
            # Apply threshold-based transformation if requested (for shock analysis)
            elif threshold:
                if '[1d]' in col or '[100]' in col:
                    aligned = make_stationary(aligned, method="diff")
                elif '[1%]' in col:
                    aligned = make_stationary(aligned, method="pct_change")
                else:
                    aligned = make_stationary(aligned, method="diff")

            col = re.sub(r"\[(.*?)\]", '', col).strip()
            col = re.sub(r"\{(.*?)\}", '', col).strip()
            aligned.name = f"{sheet_name}__{col}"
            frames.append(aligned)

    feature_df = pd.concat(frames, axis=1)

    # Attach target columns
    if stationary:
        eq["target"] = make_stationary(eq[target], 'pct_change')

    result = pd.concat([feature_df, eq[["target", target]]], axis=1)
    # Drop rows where all features are NaN
    feat_cols = [c for c in result.columns if c not in ("target", target)]
    result = result.dropna(subset=feat_cols, how="all")
    return result
