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
    load_report,
    load_comparison,
    select_base_features,
    fit_base_model,
    run_forecast,
    print_forecast_report,
    save_forecast_report,
)

_RATING_LABELS = {
    "-2": "very bad",
    "-1": "bad",
     "0": "neutral",
     "1": "good",
     "2": "very good",
}


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
        "--add_exp", action="store_true",
        help="Add expectation on features"
    )
    return parser.parse_args()


def _short_name(feature: str) -> str:
    """Strip sheet prefix (e.g. '일_주가__종가지수(변동률)' → '종가지수(변동률)')."""
    if "__" in feature:
        return feature.split("__", 1)[1]
    return feature


def prompt_feature_args(features: list[str], coefs: dict[str, float], add_exp: bool = False) -> dict[str, int]:
    """
    Interactively ask the user to rate each base indicator.
    Shows the model's ElasticNet coefficient so the user understands the direction.
    Pressing Enter (empty input) skips → treated as 0 (neutral).
    """
    if ~add_exp:
        feature_args = {feat: 0 for feat in features}
        return feature_args

    print("\n" + "=" * 65)
    print("  Qualitative Indicator Assessment")
    print("  Rate each indicator's current outlook:")
    print("  -2 very bad  |  -1 bad  |  0 neutral  |  +1 good  |  +2 very good")
    print("  (Press Enter to skip = neutral)")
    print("=" * 65)

    feature_args: dict[str, int] = {}

    for feat in features:
        coef = coefs.get(feat, 0.0)
        direction = "+" if coef >= 0 else "-"
        short = _short_name(feat)
        while True:
            try:
                raw = input(f"  [{direction}] {short}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  (Interrupted — all remaining set to neutral)")
                break
            if raw == "":
                feature_args[feat] = 0
                break
            if raw in _RATING_LABELS:
                feature_args[feat] = int(raw)
                break
            print("    Enter a value from -2 to 2 (or press Enter to skip).")

    print()
    return feature_args


def main():
    args = parse_args()

    print("=" * 65)
    print("  MFMEA - Future Forecaster")
    print("=" * 65)

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    indicators = load_indicators()
    equity = load_equity()
    print(f"      {len(indicators)} indicator sheets, {len(equity)} equity rows")

    # ── 2. Build feature matrix ───────────────────────────────────────────────
    print(f"\n[2/4] Building feature matrix (align={args.align})...")
    feature_matrix = build_feature_matrix(
        args.target, indicators, equity,
        align_method=args.align, stationary=True, threshold=False,
    )
    n_features = feature_matrix.shape[1] - 2
    print(f"      {feature_matrix.shape[0]} rows x {n_features} features")

    # ── 3. Select features & prompt for qualitative ratings ───────────────────
    print(f"\n[3/4] Selecting top {args.top} base indicators...")
    report = load_report(args.target)
    comparison = load_comparison(args.target)
    base_features = select_base_features(report, comparison, args.top)

    # Fit model first so we can show coefficient direction during prompting
    model_artifacts = fit_base_model(feature_matrix, args.target, base_features)

    feature_args = prompt_feature_args(base_features, model_artifacts["coefs"], args.add_exp)

    # ── 4. Run forecast ───────────────────────────────────────────────────────
    print(f"[4/4] Running forecast (target={args.target})...")
    result = run_forecast(
        feature_matrix,
        target=args.target,
        top_n=args.top,
        feature_args=feature_args,
        today=datetime.today(),
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