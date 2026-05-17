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
from datetime import datetime, date

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.data_loader import load_indicators, load_equity
from src.preprocessor import build_feature_matrix
from src.future_forecaster import (
    prepare_model,
    run_forecast,
    print_forecast_report,
    save_forecast_report,
)
from src.config import HORIZON_DAMPENING

# ─────────────────────────────────────────────────────────────────────────────
# Interactive prompt helpers (CLI only)
# ─────────────────────────────────────────────────────────────────────────────

_CACHE = {}


def _short_name(feature: str) -> str:
    """Strip sheet prefix: '일_주가__종가지수(변동률)' -> '종가지수(변동률)'."""
    return feature.split("__", 1)[1] if "__" in feature else feature


def prompt_feature_args(
    model_artifacts: dict,
    features: list[str],
) -> dict[str, float]:
    """
    Interactively rate each base indicator (-2..+2).
    Shows direction sign from ElasticNet coef and current EWMA value.
    Enter = skip = neutral (0).
    """
    coefs      = model_artifacts["coefs"]
    trail_ewma = model_artifacts["trailing_ewma"]

    print("\n" + "=" * 65)
    print("  Qualitative Indicator Assessment")
    print("  -2 Deep Decline  |  -1 Decline  |  0 Neutral  |  +1 Increse  |  +2 Sharp Increase")
    print("  (Press Enter to skip = neutral)")
    print("=" * 65)

    feature_args: dict[str, float] = {}
    for feat in features:
        if coefs.get(feat, 0.0) == 0.0:
            feature_args[feat] = 0
            continue
        direction = "+" if coefs.get(feat, 0.0) >= 0 else "-"
        short     = _short_name(feat)
        history   = model_artifacts['trailing_history'].get(feat)
        cur_val   = round(trail_ewma.get(feat, float("nan")), 4)
        std_val   = round(model_artifacts["trailing_feat_std"].get(feat, float("nan")), 4)
        if short in _CACHE:
            print(f"  [{direction}] {short}  (current: {cur_val}, std: {std_val}) -> {_CACHE[short]} (cached)")
            feature_args[feat] = float(_CACHE[short])
            continue
        while True:
            try:
                print("History: ", history.index.strftime('%Y%m%d').tolist(), history.values)
                raw = input(f"  [{direction}] {short}  (current: {cur_val}, std: {std_val}): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  (Interrupted — remaining set to neutral)")
                return feature_args
            if raw == "":
                feature_args[feat] = 0
                _CACHE[short] = 0
                break
            feature_args[feat] = float(raw)
            _CACHE[short] = raw
            break

    print()
    return feature_args


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MFMEA Future Forecaster")
    parser.add_argument(
        "--code", type=str, default="G0",
        help="Which code to analyze (e.g. 'G10'); if not specified, analyze last features",
    )
    parser.add_argument("--target", choices=["price", "eps"], default="eps",
                        help="Forecast target (default: price)")
    parser.add_argument("--top",    type=int, default=10,
                        help="Number of top base indicators (default: 10)")
    parser.add_argument("--align",  choices=["step_fill", "midas"], default="step_fill",
                        help="Frequency alignment method (default: step_fill)")
    parser.add_argument("--expect", action="store_true", default=True,
                        help="Prompt for per-indicator qualitative ratings")
    parser.add_argument("--ci",     type=float, default=0.95, metavar="LEVEL",
                        help="Confidence interval level (default: 0.95)")
    parser.add_argument("--damp",   type=float, default=HORIZON_DAMPENING, metavar="GAMMA",
                        help=f"Horizon dampening exponent γ (default: {HORIZON_DAMPENING}). "
                             "1.0=linear, 0.5=sqrt. Lower -> shorter long-horizon estimates.")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving output Excel file")
    parser.add_argument("--all", action="store_true",
                         help="Analyze all codes in the equity list (overrides --code)")
    parser.add_argument("--date", action="store", type=str, default=None,)
    parser.add_argument("--s_len", action="store", type=int, default=3, help="Minimum length of sector code to analyze (default: 5)")
    return parser.parse_args()


def main(args):
    print("=" * 65)
    print("  MFMEA - Future Forecaster")
    print("=" * 65)
    t0 = time.time()

    # 1. Load raw data
    print("\n[1/3] Loading data...")
    indicators = load_indicators(args.date)
    equity     = load_equity(args.target, args.code, args.date)
    if len(equity.loc[pd.to_datetime(args.date)-relativedelta(years=1):args.date]) < 40:
        print(f"      Loaded {len(equity)} equity rows. Since there is not enough data, skip analysis.")
        return False
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
    model_artifacts, base_features = prepare_model(feature_matrix, args.code, args.target, args.top)
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
        code            = args.code,
        target          = args.target,
        feature_args    = feature_args,
        model_artifacts = model_artifacts,   # skip re-fitting
        top_n           = args.top,
        today           = args.date,
        ci_level        = args.ci,
        horizon_damp    = args.damp,
        verbose         = True,
    )

    print_forecast_report(result)

    if not args.no_save:
        path = save_forecast_report(args.code, result, save=True)
        if path:
            print(f"  Saved -> {path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    args = parse_args()
    if args.date == None:
        args.date = date.today().strftime("%Y-%m-%d")

    if args.all:
        equ_list = pd.read_excel("data/equity.xlsx", sheet_name="list", header=0)
        start_length = args.s_len - 2
        total_length = args.s_len
        equ_list = equ_list[(start_length < equ_list.Length) & (equ_list.Length <= total_length)]
        equs = equ_list["Code"].tolist()
        print(f"Analyzing equities with length ≤ {total_length}: {equs}")
        for code in equs:
            args.code = code
            print(f"\n{'#' * 60}")
            print(f" Analyzing {code} (length={total_length})")
            print(f"{'#' * 60}")
            main(args)
    else:
        main(args)