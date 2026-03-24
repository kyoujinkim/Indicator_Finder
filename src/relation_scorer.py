"""
Relation Scorer - aggregates all analysis results into a unified ranking.

Scoring formula (normalized 0-1 per metric, then weighted sum):

  Continuous Impact Score =
      w_gc * (1 - granger_pval)  +  w_te * transfer_entropy  +  w_en * elasticnet_coef

  Shock Score =
      w_irf * irf_persistence  +  w_ev * |event_car|  +  w_sh * shap_value

  Threshold Score =
      w_rd * |regime_diff_norm|  +  w_ts * trigger_score  +  w_as * asymmetry_norm
      (penalised by threshold_pval so statistically weak thresholds score lower)

  Final Score = w_cont * Continuous + w_shock * Shock + w_thresh * Threshold
    + bonus if Boruta-confirmed
    + MI score contribution
"""

import numpy as np
import pandas as pd

from .config import PATH_WEIGHTS, BORUTA_BONUS, METRIC_WEIGHTS


def _normalize(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_relation_scores(
    continuous_summary: pd.DataFrame,
    irf_summary: pd.DataFrame,
    shock_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    fs_results: dict,
    path_weights: tuple[float, float, float, float] = PATH_WEIGHTS,
    boruta_bonus: float = BORUTA_BONUS,
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Merge all analysis outputs into a single ranked DataFrame.

    Parameters
    ----------
    continuous_summary  : output of run_continuous_analysis()
    irf_summary         : output of run_irf_analysis()
    shock_summary       : output of run_shock_analysis()
    threshold_summary   : output of run_threshold_analysis()
    fs_results          : output of run_feature_selection()
    path_weights        : (w_continuous, w_shock, w_threshold) — must sum to 1
    boruta_bonus        : bonus score added to Boruta-confirmed features
    weights             : optional dict to override per-metric weights within each path

    Returns
    -------
    DataFrame with columns [continuous_score, irf_score, shock_score, threshold_score,
    final_score, impact_path, rank, ...] sorted by final_score descending.
    """
    w_cont, w_irf, w_shock, w_thresh = path_weights

    if weights is None:
        weights = METRIC_WEIGHTS

    # ---------- Continuous ----------
    cont = continuous_summary.copy()
    cont["gc_score"] = _normalize(1 - cont["granger_pval"].clip(0, 1))
    cont["te_norm"] = _normalize(cont["transfer_entropy"].fillna(0))
    cont["en_norm"] = _normalize(cont["elasticnet_coef"].fillna(0))

    cont_score = (
        weights["granger"] * cont["gc_score"]
        + weights["transfer_entropy"] * cont["te_norm"]
        + weights["elasticnet"] * cont["en_norm"]
    )

    # --------- IRF ----------
    irf = irf_summary.copy()
    irf["irf_norm"] = _normalize(irf["irf_persistence"].fillna(0))
    irf_score = irf["irf_norm"] * weights["irf"]

    # ---------- Shock ----------
    shock = shock_summary.copy()
    shock["event_norm"] = _normalize(shock["event_car"].abs().fillna(0))
    shock["shap_norm"] = _normalize(shock["shap_value"].fillna(0))

    shock_score = (
        weights["event"] * shock["event_norm"]
        + weights["shap"] * shock["shap_norm"]
    )

    # ---------- Threshold ----------
    thresh = threshold_summary.copy()
    # Penalize by p-value: multiply raw scores by (1 - pval) so weak thresholds score less
    pval_weight = (1 - thresh["threshold_pval"].fillna(1).clip(0, 1))
    thresh["rd_norm"] = _normalize(thresh["regime_diff"].abs().fillna(0)) * pval_weight
    thresh["ts_norm"] = _normalize(thresh["trigger_score"].fillna(0))
    thresh["as_norm"] = _normalize(thresh["asymmetry"].fillna(0))

    thresh_score = (
        weights["regime_diff"] * thresh["rd_norm"]
        + weights["trigger_score"] * thresh["ts_norm"]
        + weights["asymmetry"] * thresh["as_norm"]
    )

    # ---------- Merge ----------
    merged = pd.DataFrame({
        "continuous_score": cont_score,
        "irf_score": irf_score,
        "shock_score": shock_score,
        "threshold_score": thresh_score,
    })
    merged["final_score"] = (
        w_cont * merged["continuous_score"]
        + w_irf * merged["irf_score"]
        + w_shock * merged["shock_score"]
        + w_thresh * merged["threshold_score"]
    )

    # Boruta bonus
    boruta_accepted = set(fs_results["boruta"].get("accepted", []))
    merged["boruta_confirmed"] = merged.index.isin(boruta_accepted)
    merged.loc[merged["boruta_confirmed"], "final_score"] += boruta_bonus

    # MI score
    mi = fs_results["mi_scores"]
    if len(mi) > 0:
        mi_norm = _normalize(mi)
        merged["mi_score"] = mi_norm
        merged["final_score"] += 0.05 * mi_norm.reindex(merged.index).fillna(0)
    else:
        merged["mi_score"] = 0.0

    # Propagate raw metrics for reporting
    merged["granger_lag"] = cont["granger_lag"]
    merged["granger_sig"] = cont["granger_sig"]
    merged["granger_pval"] = cont["granger_pval"]
    merged["transfer_entropy"] = cont["transfer_entropy"]
    merged["elasticnet_coef"] = cont["elasticnet_coef"]
    merged["irf_persistence"] = irf["irf_persistence"]
    merged["event_car"] = shock["event_car"]
    merged["shap_value"] = shock["shap_value"]
    merged["threshold"] = thresh["threshold"]
    merged["threshold_quantile"] = thresh["threshold_quantile"]
    merged["regime_diff"] = thresh["regime_diff"]
    merged["threshold_pval"] = thresh["threshold_pval"]
    merged["trigger_score"] = thresh["trigger_score"]
    merged["n_crossings"] = thresh["n_crossings"]
    for h in [1, 3, 5, 10]:
        col = f"hit_rate_{h}"
        if col in thresh.columns:
            merged[col] = thresh[col]

    merged = merged.sort_values("final_score", ascending=False)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    return merged
