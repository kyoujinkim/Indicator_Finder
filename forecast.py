"""
MFMEA - Future Target Forecaster (CLI entry point)
===================================================
Thin CLI wrapper around src/future_forecaster.py.

Flow:
  1. Load raw data and build feature matrix
  2. prepare_model()  -> fits ElasticNet, returns coefs for display
  3. prompt_feature_args() [--expect only] -> interactive qualitative ratings
  4. run_forecast()   -> two-step forecast with CI
  5. print / save report

Usage:
  python forecast.py --target price --top 10
  python forecast.py --target price --top 10 --expect
  python forecast.py --target eps   --top 8  --damp 0.7 --ci 0.90 --no-save
"""

import argparse
import time
from datetime import datetime

from src.data_loader import load_indicators, load_equity
from src.preprocessor import build_feature_matrix
from src.future_forecaster import (
    prepare_model,
    run_forecast,
    print_forecast_report,
    save_forecast_report,
)
from src.config import HORIZON_DAMPENING, ARG_LABEL

# ─────────────────────────────────────────────────────────────────────────────
# Interactive prompt helpers (CLI only)
# ─────────────────────────────────────────────────────────────────────────────

# String-keyed version of ARG_LABEL for CLI input() validation
_RATING_LABELS = {str(k): v for k, v in ARG_LABEL.items()}


def _short_name(feature: str) -> str:
    """Strip sheet prefix: '일_주가__종가지수(변동률)' -> '종가지수(변동률)'."""
    return feature.split("__", 1)[1] if "__" in feature else feature


def prompt_feature_args(
    model_artifacts: dict,
    features: list[str],
) -> dict[str, int]:
    """
    Interactively rate each base indicator (-2..+2).
    Shows direction sign from ElasticNet coef and current EWMA value.
    Enter = skip = neutral (0).
    """
    coefs      = model_artifacts["coefs"]
    trail_ewma = model_artifacts["trailing_ewma"]

    print("\n" + "=" * 65)
    print("  Qualitative Indicator Assessment")
    print("  -2 very bad  |  -1 bad  |  0 neutral  |  +1 good  |  +2 very good")
    print("  (Press Enter to skip = neutral)")
    print("=" * 65)

    feature_args: dict[str, int] = {}
    for feat in features:
        direction = "+" if coefs.get(feat, 0.0) >= 0 else "-"
        short     = _short_name(feat)
        cur_val   = round(trail_ewma.get(feat, float("nan")), 4)
        while True:
            try:
                raw = input(f"  [{direction}] {short}  (now: {cur_val}): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  (Interrupted — remaining set to neutral)")
                return feature_args
            if raw == "":
                feature_args[feat] = 0
                break
            if raw in _RATING_LABELS:
                feature_args[feat] = int(raw)
                break
            print("    Enter a value from -2 to 2 (or press Enter to skip).")

    print()
    return feature_args


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MFMEA Future Forecaster")
    parser.add_argument("--target", choices=["price", "eps"], default="price",
                        help="Forecast target (default: price)")
    parser.add_argument("--top",    type=int, default=10,
                        help="Number of top base indicators (default: 10)")
    parser.add_argument("--align",  choices=["step_fill", "midas"], default="step_fill",
                        help="Frequency alignment method (default: step_fill)")
    parser.add_argument("--expect", action="store_true",
                        help="Prompt for per-indicator qualitative ratings")
    parser.add_argument("--ci",     type=float, default=0.95, metavar="LEVEL",
                        help="Confidence interval level (default: 0.95)")
    parser.add_argument("--damp",   type=float, default=HORIZON_DAMPENING, metavar="GAMMA",
                        help=f"Horizon dampening exponent γ (default: {HORIZON_DAMPENING}). "
                             "1.0=linear, 0.5=sqrt. Lower -> shorter long-horizon estimates.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving output Excel file")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  MFMEA - Future Forecaster")
    print("=" * 65)
    t0 = time.time()

    # 1. Load raw data
    print("\n[1/3] Loading data...")
    indicators = load_indicators()
    equity     = load_equity()
    print(f"      {len(indicators)} indicator sheets, {len(equity)} equity rows")

    # 2. Build feature matrix
    print(f"\n[2/3] Building feature matrix (align={args.align})...")
    feature_matrix = build_feature_matrix(
        args.target, indicators, equity,
        align_method=args.align, stationary=True, threshold=False,
    )
    print(f"      {feature_matrix.shape[0]} rows x {feature_matrix.shape[1] - 2} features")

    # 3. Fit model (done once; reused inside run_forecast)
    print(f"\n[3/3] Running forecast (target={args.target})...")
    model_artifacts, base_features = prepare_model(feature_matrix, args.target, args.top)
    print(f"      {len(base_features)} base features selected, "
          f"{len(model_artifacts['nonzero_coefs'])} non-zero coefs")

    # Optional: interactive qualitative ratings
    feature_args = (
        prompt_feature_args(model_artifacts, base_features)
        if args.expect
        else {}
    )

    result = run_forecast(
        feature_matrix,
        target          = args.target,
        feature_args    = feature_args,
        model_artifacts = model_artifacts,   # skip re-fitting
        top_n           = args.top,
        today           = datetime.today(),
        ci_level        = args.ci,
        horizon_damp    = args.damp,
        verbose         = True,
    )

    print_forecast_report(result)

    if not args.no_save:
        path = save_forecast_report(result, save=True)
        if path:
            print(f"  Saved -> {path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()