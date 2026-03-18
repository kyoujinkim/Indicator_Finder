"""
MFMEA - Future Target Forecaster
=================================
Reads pre-computed output reports and builds a two-step forecast:
  · Step 1: Long-term trend model (ElasticNet on top-ranked indicators)
  · Step 2: Short-term event adjustment (Impact Decay from impacts.txt)

After the model selects base indicators, you are prompted to rate each one
individually (-2 to +2). The rating shifts that indicator's effective current
value before prediction — the model's own coefficients determine whether a
positive outlook actually raises or lowers the forecast.

Usage:
  python forecast.py --target price --top 10
  python forecast.py --target eps   --top 8 --no-save
"""

import argparse
import time
from datetime import datetime

from src.data_loader import load_indicators, load_equity
from src.preprocessor import build_feature_matrix
from src.future_forecaster import (
    run_forecast,
    print_forecast_report,
    save_forecast_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MFMEA Future Forecaster")
    parser.add_argument(
        "--target", choices=["price", "eps"], default="price",
        help="Forecast target (default: price)",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of top base indicators to use (default: 10)",
    )
    parser.add_argument(
        "--align", choices=["step_fill", "midas"], default="step_fill",
        help="Frequency alignment method (default: step_fill)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving output Excel file",
    )
    parser.add_argument(
        "--expect", action="store_true",
        help="Add expectation on features"
    )
    parser.add_argument(
        "--ci", type=float, default=0.95, metavar="LEVEL",
        help="Confidence interval level (default: 0.95 = 95%%)",
    )
    parser.add_argument(
        "--damp", type=float, default=0.9, metavar="GAMMA",
        help="Horizon dampening exponent γ (default: 0.6). "
             "1.0=linear, 0.5=square-root. Lower values shrink long-horizon estimates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  MFMEA - Future Forecaster")
    print("=" * 65)

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    indicators = load_indicators()
    equity = load_equity()
    print(f"      {len(indicators)} indicator sheets, {len(equity)} equity rows")

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    print(f"\n[2/3] Building feature matrix (align={args.align})...")
    feature_matrix = build_feature_matrix(
        args.target, indicators, equity,
        align_method=args.align, stationary=True, threshold=False,
    )
    n_features = feature_matrix.shape[1] - 2
    print(f"      {feature_matrix.shape[0]} rows x {n_features} features")

    # ── 3. Run forecast ───────────────────────────────────────────────────────
    print(f"[3/3] Running forecast (target={args.target})...")
    result = run_forecast(
        feature_matrix,
        expect=args.expect,
        target=args.target,
        top_n=args.top,
        today=datetime.today(),
        ci_level=args.ci,
        horizon_damp=args.damp,
        verbose=True,
    )

    print_forecast_report(result)

    if not args.no_save:
        path = save_forecast_report(result, save=True)
        if path:
            print(f"  Saved -> {path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()