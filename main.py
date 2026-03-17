"""
MFMEA - Multi-Frequency Macro-Equity Analytics
===============================================
Entry point. Orchestrates the full pipeline:

  1. Load data (data/data.xlsx, data/equity.xlsx)
  2. Preprocess (align frequencies, stationarity)
  3. Continuous Impact Analysis (Granger, Transfer Entropy, ElasticNet)
  4. Shock & Event Analysis (IRF, Event Study, SHAP)
  5. Threshold / Indicator-Function Analysis (optimal threshold, crossing events, signal quality)
  6. Feature Selection (MI, Boruta, Regime-Aware)
  7. Relation Scoring (aggregate → rank) + Report generation

Usage:
  python main.py [--target price|eps] [--top 20] [--align step_fill|midas]
"""

import argparse
import time

from src.data_loader import load_indicators, load_equity
from src.preprocessor import build_feature_matrix
from src.continuous_analysis import run_continuous_analysis
from src.irf_analysis import run_irf_analysis
from src.shock_analysis import run_shock_analysis
from src.threshold_analysis import run_threshold_analysis
from src.feature_selector import run_feature_selection
from src.relation_scorer import compute_relation_scores
from src.reporter import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="MFMEA - Macro-Equity Analytics")
    parser.add_argument("--target", choices=["price", "eps"], default="price")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--align", choices=["step_fill", "midas"], default="step_fill")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--no-save", action="store_true", help="Skip saving output files")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" MFMEA - Multi-Frequency Macro-Equity Analytics")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    t0 = time.time()
    print("\n[1/7] Loading data...")
    indicators = load_indicators()
    equity = load_equity()
    print(f"      Loaded {len(indicators)} indicator sheets, {len(equity)} equity rows")

    # ── 2. Preprocess ─────────────────────────────────────────────
    print(f"\n[2/7] Preprocessing (align={args.align})...")
    feature_matrix = build_feature_matrix(
        indicators, equity, align_method=args.align, stationary=True, threshold=False
    )
    feature_threshold_matrix = build_feature_matrix(
        indicators, equity, align_method=args.align, stationary=True, threshold=False
    )
    n_features = feature_matrix.shape[1] - 2  # minus price, eps
    print(f"      Feature matrix: {feature_matrix.shape[0]} rows × {n_features} features")

    # ── 3. Continuous Impact ──────────────────────────────────────
    print(f"\n[3/7] Continuous impact analysis (target={args.target})...")
    continuous = run_continuous_analysis(
        feature_matrix, target_col=args.target, max_lag=args.max_lag
    )
    sig_count = continuous["granger_sig"].sum()
    print(f"      {sig_count}/{len(continuous)} features Granger-significant at p<0.05")

    # ── 4. irf Analysis ─────────────────────────────────────────
    print(f"\n[4/7] Shock & event analysis...")
    irf = run_irf_analysis(feature_matrix, target_col=args.target)
    print(f"      IRF and SHAP computed for {len(irf)} features")

    # ── 4. Shock Analysis ─────────────────────────────────────────
    print(f"\n[4/7] Shock & event analysis...")
    shock = run_shock_analysis(feature_threshold_matrix, target_col=args.target)
    print(f"      IRF and SHAP computed for {len(shock)} features")

    # ── 5. Threshold Analysis ─────────────────────────────────────
    print(f"\n[5/7] Threshold (indicator-function) analysis...")
    threshold = run_threshold_analysis(feature_threshold_matrix, target_col=args.target)
    n_triggers = int((threshold["trigger_score"] > 0.55).sum())
    print(f"      {n_triggers}/{len(threshold)} features show strong threshold trigger (score>0.55)")

    # ── 6. Feature Selection ──────────────────────────────────────
    print(f"\n[6/7] Feature selection (MI, Boruta, Regime)...")
    fs = run_feature_selection(feature_matrix, target_col=args.target)
    n_accepted = len(fs["boruta"].get("accepted", []))
    print(f"      Boruta accepted {n_accepted} features")

    # ── 7. Relation Scoring ───────────────────────────────────────
    print(f"\n[7/7] Computing relation scores...")
    scores = compute_relation_scores(continuous, irf, shock, threshold, fs)
    print(f"      Scored {len(scores)} features, top score: {scores['final_score'].max():.4f}")

    # ── Report ────────────────────────────────────────────────────
    print()
    generate_report(
        scores,
        fs,
        target_col=args.target,
        top_n=args.top,
        save=not args.no_save,
    )

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
