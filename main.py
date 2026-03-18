"""
MFMEA - Multi-Frequency Macro-Equity Analytics
===============================================
Entry point. Orchestrates the full pipeline:

  1. Load data (data/data.xlsx, data/equity.xlsx)
  2. Preprocess (align frequencies, stationarity)
  3. Continuous Impact Analysis (Granger, Transfer Entropy, ElasticNet)
  4. IRF Analysis (VAR-IRF, persistence score)
  5. Shock & Event Analysis (Event Study, SHAP)
  6. Named-Event Impact Analysis (equity response + indicator tracking, data/impacts.txt)
  7. Threshold / Indicator-Function Analysis (optimal threshold, crossing events, signal quality)
  8. Feature Selection (MI, Boruta, Regime-Aware)
  9. Relation Scoring (aggregate → rank) + Report generation

  --compare: additionally run a rolling-window temporal comparison
             to measure how importance changed over the past month.

Usage:
  python main.py [--target price|eps] [--top 20] [--align step_fill|midas]
  python main.py --compare [--window 52] [--shift 4]
"""

import argparse
import time

from src.data_loader import load_indicators, load_equity
from src.preprocessor import build_feature_matrix
from src.continuous_analysis import run_continuous_analysis
from src.irf_analysis import run_irf_analysis
from src.shock_analysis import run_shock_analysis
from src.event_impact import run_event_impact
from src.threshold_analysis import run_threshold_analysis
from src.feature_selector import run_feature_selection
from src.relation_scorer import compute_relation_scores
from src.reporter import generate_report, generate_comparison_report, generate_event_impact_report
from src.temporal_comparison import compare_periods


def parse_args():
    parser = argparse.ArgumentParser(description="MFMEA - Macro-Equity Analytics")
    parser.add_argument("--target", choices=["price", "eps"], default="price")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--align", choices=["step_fill", "midas"], default="step_fill")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--no-save", action="store_true", help="Skip saving output files")
    parser.add_argument(
        "--compare", action="store_true",
        help="Run temporal comparison (current vs. past window)",
    )
    parser.add_argument(
        "--window", type=int, default=52,
        help="Rolling window size in weeks for comparison (default: 52 = 1 year)",
    )
    parser.add_argument(
        "--shift", type=int, default=4,
        help="Weeks to shift back for the past window (default: 4 ≈ 1 month)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(" MFMEA - Multi-Frequency Macro-Equity Analytics")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    t0 = time.time()
    print("\n[1/9] Loading data...")
    indicators = load_indicators()
    equity = load_equity()
    print(f"      Loaded {len(indicators)} indicator sheets, {len(equity)} equity rows")

    # ── 2. Preprocess ─────────────────────────────────────────────
    print(f"\n[2/9] Preprocessing (align={args.align})...")
    feature_matrix = build_feature_matrix(
        args.target, indicators, equity, align_method=args.align, stationary=True, threshold=False
    )
    n_features = feature_matrix.shape[1] - 2  # minus price, eps
    print(f"      Feature matrix: {feature_matrix.shape[0]} rows × {n_features} features")

    # ── 3. Continuous Impact ──────────────────────────────────────
    print(f"\n[3/9] Continuous impact analysis (target={args.target})...")
    continuous = run_continuous_analysis(
        feature_matrix, target_col=args.target, max_lag=args.max_lag
    )
    sig_count = continuous["granger_sig"].sum()
    print(f"      {sig_count}/{len(continuous)} features Granger-significant at p<0.05")

    # ── 4. IRF Analysis ───────────────────────────────────────────
    print(f"\n[4/9] IRF analysis...")
    irf = run_irf_analysis(feature_matrix, target_col=args.target)
    print(f"      IRF computed for {len(irf)} features")

    # ── 5. Shock Analysis ─────────────────────────────────────────
    print(f"\n[5/9] Shock & event analysis...")
    shock = run_shock_analysis(feature_matrix, target_col=args.target)
    print(f"      Event study and SHAP computed for {len(shock)} features")

    # ── 6. Named-Event Impact ─────────────────────────────────────
    print(f"\n[6/9] Named-event impact analysis (impacts.txt)...")
    event_impact = run_event_impact(feature_matrix, target_col=args.target)
    n_events = len(event_impact["events"])
    n_matched = len(event_impact["equity_impact"])
    print(f"      {n_matched}/{n_events} events matched in data; "
          f"{len(event_impact['feature_score'])} features tracked")

    # ── 7. Threshold Analysis ─────────────────────────────────────
    print(f"\n[7/9] Threshold (indicator-function) analysis...")
    threshold = run_threshold_analysis(feature_matrix, target_col=args.target)
    n_triggers = int((threshold["trigger_score"] > 0.55).sum())
    print(f"      {n_triggers}/{len(threshold)} features show strong threshold trigger (score>0.55)")

    # ── 8. Feature Selection ──────────────────────────────────────
    print(f"\n[8/9] Feature selection (MI, Boruta, Regime)...")
    fs = run_feature_selection(feature_matrix, target_col=args.target)
    n_accepted = len(fs["boruta"].get("accepted", []))
    print(f"      Boruta accepted {n_accepted} features")

    # ── 9. Relation Scoring ───────────────────────────────────────
    print(f"\n[9/9] Computing relation scores...")
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
    generate_event_impact_report(
        event_impact,
        target_col=args.target,
        top_n=args.top,
        save=not args.no_save,
    )

    # ── Temporal Comparison ───────────────────────────────────────
    if args.compare:
        print(f"\n{'=' * 60}")
        print(f" Temporal Comparison (window={args.window}w, shift={args.shift}w)")
        print(f"{'=' * 60}")
        comparison = compare_periods(
            feature_matrix,
            target_col=args.target,
            window_weeks=args.window,
            shift_weeks=args.shift,
            verbose=True,
        )
        print()
        generate_comparison_report(
            comparison,
            target_col=args.target,
            top_n=args.top,
            save=not args.no_save,
        )

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
