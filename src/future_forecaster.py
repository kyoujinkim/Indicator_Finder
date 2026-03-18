"""
Future Forecaster for MFMEA.
============================================================
Two-step forecast architecture (see readme_future_forecast.md):

  Step 1 - Base Trend Model
    · Selects top-N indicators from report_{target} by final_score
    · Excludes 'fallen' features via comparison_{target}
    · Deduplicates by indicator name (highest score kept)
    · Fits ElasticNetCV on historical stationary returns
    · Per-feature EWMA (span=8w) represents current market state
    · Optional qualitative assessment via --add_exp flag:
        Each indicator can be rated -2..+2; the rating shifts its EWMA by
        arg × std(feat) × 0.5 before prediction. The model's own coefficient
        determines whether a positive rating raises or lowers the forecast.
    · Weekly signal is accumulated linearly across forecast windows (1w..26w)

  Step 2 - Event Volatility Adjustment
    · Parses data/impacts.txt for named macro events and dates
    · For each event, computes days_since = today - event_date
    · _active_windows(): for each forecast slot (1w/2w/4w/13w/26w), maps to
        the time-shifted ar_ lookup column using:
            w1 = abs(-(days_since + n_weeks*7) // 7)
        so the ar_ column reflects the remaining horizon from today rather
        than from the original event date.
    · Abnormal return (ar_+Xw, Z-score units) is fetched from Equity_Summary,
        multiplied by exponential decay weight (half_life=14d), then scaled by
        target_series_std (trailing BASELINE_WINDOW=12w) to convert to return units.
    · Contributions are summed across all active events per window.

Returns:
  dict with keys:
    'forecast'     : pd.DataFrame - windows x {base_return, event_adjustment,
                                                total_return, direction,
                                                magnitude_%, projected_value}
    'base_model'   : dict - ElasticNet model, coefs, EWMA values, feature stds
    'events_used'  : list[dict] - events that contributed with decay/contrib detail
    'feature_args' : dict[str, int] - per-indicator qualitative ratings used
    'windows'      : list[str]
"""

import glob
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"

# Forecast horizons: label → calendar days
WINDOWS: dict[str, int] = {
    "1w":  7,
    "2w":  14,
    "4w":  28,
    "13w": 91,
    "26w": 182,
}

# Map each window to the closest available return column in Equity_Summary
# Use ret_+ (actual % returns) not ar_+ (Z-score abnormal returns)
_WINDOW_TO_AR_COL: dict[str, str] = {
    "1w":  "ar_+1w",
    "2w":  "ar_+2w",
    "4w":  "ar_+4w",
    "13w": "ar_+8w",
    "26w": "ar_+8w",
}

# Qualitative arg → numeric nudge (fraction of trailing std)
_ARG_SCALE = 0.5  # half a trailing std per arg unit
# Pre-event baseline length for abnormal-return calc
BASELINE_WINDOW = 12  # weeks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Path:
    """Return the lexicographically last file matching an output/ glob."""
    files = sorted(glob.glob(str(OUTPUT_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No output file found for pattern: {OUTPUT_DIR / pattern}")
    return Path(files[-1])


def load_report(target: str) -> pd.DataFrame:
    path = _find_latest(f"mfmea_report_{target}_*.csv")
    return pd.read_csv(path)


def load_comparison(target: str) -> pd.DataFrame:
    path = _find_latest(f"mfmea_comparison_{target}_*.csv")
    return pd.read_csv(path)


def _load_equity_summary(target: str) -> pd.DataFrame:
    """Load Equity_Summary sheet from the latest event_impact xlsx."""
    path = _find_latest(f"mfmea_event_impact_{target}_*.xlsx")
    return pd.read_excel(path, sheet_name="Equity_Summary")


def _load_events() -> pd.DataFrame:
    """Parse data/impacts.txt → DataFrame[event_name, event_date]."""
    rows = []
    with open(DATA_DIR / "impacts.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            try:
                date = pd.to_datetime(parts[1].strip())
                rows.append({"event_name": name, "event_date": date})
            except Exception:
                continue
    return pd.DataFrame(rows)


def _active_windows(days_since: int) -> dict:
    """
    Return list of windows still active given days since event, based on simple week-based thresholds.
        For example, an event 10 days ago would still influence 1w and 2w windows, but not 4w+.
    """
    # Adjust windows to reflect elapsed time
    adjusted = {}
    for a, n in zip(['1w', '2w', '4w', '13w', '26w'], [1, 2, 4, 13, 26]):
        w1 = abs(-(days_since + n*7)//7)
        if w1 < 1:
            adjusted[a] = "1w"
        elif w1 < 2:
            adjusted[a] = "2w"
        elif w1 < 4:
            adjusted[a] = "4w"
        elif w1 < 13:
            adjusted[a] = "13w"
        elif w1 < 26:
            adjusted[a] = "26w"
        else:
            pass

    return adjusted


def get_active_windows(days_since: int) -> list[str | None]:
    """
    Returns a fixed-length list of windows (1w, 2w, 4w, 13w, 26w).
    Preserves original logic: an event 10 days ago still influences the 1w position.
    """
    # These are the 'a' values from your original loop
    window_steps = [1, 2, 4, 13, 26]
    labels = ["1w", "2w", "4w", "13w", "26w"]

    adjusted = []
    for step, label in zip(window_steps, labels):
        # This replicates your -(days_since + a*7)//7 logic
        # which determines if the event is still 'active' for this specific step
        is_active = (days_since + step * 7) // 7 > 0

        # Your logic implies that as long as the calculation is positive,
        # the window label is appended.
        if is_active and (days_since // 7) < step:
            adjusted.append(label)
        else:
            adjusted.append(None)

    return adjusted


def _event_decay_weight(days_since: int, half_life: int = 14) -> float:
    """Exponential decay: weight = exp(-λ * days_since), λ = ln2 / half_life."""
    lam = np.log(2) / half_life
    return np.exp(-lam * days_since)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Select base indicators
# ─────────────────────────────────────────────────────────────────────────────

def select_base_features(
    report: pd.DataFrame,
    comparison: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """
    Pick top_n features from report by final_score.
    Exclude features marked 'fallen' in comparison (rank degraded significantly).
    """
    # Build exclusion set from comparison
    fallen = set()
    if "status" in comparison.columns:
        fallen = set(comparison.loc[comparison["status"] == "fallen", "feature"])

    candidates = (
        report[~report["feature"].isin(fallen)]
        .sort_values("final_score", ascending=False)
        .head(top_n * 2)  # oversample, then deduplicate by indicator
    )

    # Deduplicate: keep highest-scoring feature per indicator name
    seen_indicators: set[str] = set()
    selected: list[str] = []
    for _, row in candidates.iterrows():
        ind = row.get("indicator", row["feature"])
        if ind not in seen_indicators:
            seen_indicators.add(ind)
            selected.append(row["feature"])
        if len(selected) >= top_n:
            break

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Base trend model
# ─────────────────────────────────────────────────────────────────────────────

def fit_base_model(
    feature_matrix: pd.DataFrame,
    target: str,
    base_features: list[str],
) -> dict:
    """
    Fit ElasticNetCV on the base features against the stationary target return.
    Returns model artefacts: scaler, model, selected_features, trailing_ewma.
    """
    # Intersect requested features with available columns
    available = [f for f in base_features if f in feature_matrix.columns]
    if not available:
        raise ValueError("None of the selected base features are present in the feature matrix.")

    df = feature_matrix[available + ["target"]].dropna()
    X = df[available].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
        cv=5,
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_scaled, y)

    # EWMA of each feature for "current state" extrapolation
    ewma_span = 8  # 8-week half-life for trend
    trailing_ewma = {
        feat: feature_matrix[feat].ewm(span=ewma_span).mean().iloc[-1]
        for feat in available
    }
    # Per-feature trailing std — used to scale qualitative arg nudges
    trailing_feat_std = {
        feat: float(feature_matrix[feat].dropna().std())
        for feat in available
    }

    coefs = dict(zip(available, model.coef_))
    nonzero = {f: c for f, c in coefs.items() if abs(c) > 1e-6}

    return {
        "model": model,
        "scaler": scaler,
        "features": available,
        "coefs": coefs,
        "nonzero_coefs": nonzero,
        "trailing_ewma": trailing_ewma,
        "trailing_feat_std": trailing_feat_std,
        "trailing_std": float(np.std(y)),
        "trailing_mean_y": float(np.mean(y[-8:])),  # recent 8-week mean return
    }


def _base_forecast(
    model_artifacts: dict,
    feature_args: dict[str, int] | None = None,
) -> dict[str, float]:
    """
    Project a base return estimate for each window.

    Each feature's EWMA is optionally nudged by a per-indicator qualitative
    score (feature_args[feat] in -2..+2), shifting the effective current value
    by  arg * trailing_std[feat] * _ARG_SCALE  before the model predicts.
    The model's own coefficients then determine whether a positive nudge on a
    given indicator raises or lowers the target forecast.

    Returns {window: cumulative_return}.
    """
    features = model_artifacts["features"]
    feat_std = model_artifacts["trailing_feat_std"]
    feature_args = feature_args or {}

    # Apply per-indicator arg nudge to EWMA values
    ewma_vals = np.array([
        model_artifacts["trailing_ewma"][f]
        + feature_args.get(f, 0) * _ARG_SCALE * feat_std.get(f, 0.0)
        for f in features
    ])

    ewma_scaled = model_artifacts["scaler"].transform(ewma_vals.reshape(1, -1))
    weekly_signal = float(model_artifacts["model"].predict(ewma_scaled)[0])

    result: dict[str, float] = {}
    for window, days in WINDOWS.items():
        n_weeks = days / 7.0
        result[window] = weekly_signal * n_weeks

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Event volatility adjustment
# ─────────────────────────────────────────────────────────────────────────────

def _event_adjustment(
    equity_summary: pd.DataFrame,
    today: datetime | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """
    Compute per-window event adjustment using Impact Decay.

    Returns
    -------
    adjustments : {window: cumulative_ar_adjustment}
    events_used : list of dicts with event metadata
    """
    if today is None:
        today = datetime.today()

    events_df = _load_events()
    adjustments: dict[str, float] = {w: 0.0 for w in WINDOWS}
    events_used: list[dict] = []

    for _, evt in events_df.iterrows():
        event_name: str = evt["event_name"]
        event_date: pd.Timestamp = evt["event_date"]

        days_since = (today - event_date.to_pydatetime()).days
        active = _active_windows(days_since)
        if not active:
            continue  # event too old to influence any window

        decay = _event_decay_weight(days_since)

        # Look up average abnormal return for this event name in Equity_Summary
        match = equity_summary[equity_summary["event_name"] == event_name]
        if match.empty:
            continue

        row = match.iloc[0]
        window_contrib: dict[str, float] = {}
        for _, (a, window) in enumerate(active.items()):
            ar_col = _WINDOW_TO_AR_COL[window]
            if ar_col in row.index and pd.notna(row[ar_col]):
                contrib = float(row[ar_col]) * decay
                adjustments[a] += contrib
                window_contrib[a] = contrib

        if window_contrib:
            events_used.append({
                "event_name": event_name,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "days_since": days_since,
                "decay_weight": round(decay, 4),
                "window_contrib": window_contrib,
            })

    return adjustments, events_used


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast(
    feature_matrix: pd.DataFrame,
    target: str,
    top_n: int = 10,
    feature_args: dict[str, int] | None = None,
    today: datetime | None = None,
    verbose: bool = True,
) -> dict:
    """
    Main forecasting entry point.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Output of build_feature_matrix() - stationary indicator + target columns.
    target : str
        'price' or 'eps'.
    top_n : int
        Number of base indicators to use for the trend model.
    feature_args : dict[str, int], optional
        Per-indicator qualitative scores (-2=very bad .. +2=very good).
        Each entry shifts that indicator's effective current EWMA value before
        the model predicts, so the sign of the model's own coefficient
        determines whether a positive outlook raises or lowers the forecast.
    today : datetime, optional
        Reference date for event decay (defaults to datetime.today()).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        'forecast'      : pd.DataFrame
        'base_model'    : dict
        'events_used'   : list[dict]
        'feature_args'  : dict[str, int]
        'windows'       : list[str]
    """
    if today is None:
        today = datetime.today()

    # ── Load report outputs ───────────────────────────────────────────────────
    if verbose:
        print("[Forecast] Loading output reports...")
    report = load_report(target)
    comparison = load_comparison(target)
    equity_summary = _load_equity_summary(target)

    # ── Step 1: Select base features ─────────────────────────────────────────
    base_features = select_base_features(report, comparison, top_n)
    if verbose:
        print(f"[Forecast] Base features selected ({len(base_features)}): "
              f"{base_features[:5]}{'...' if len(base_features) > 5 else ''}")

    # ── Step 2: Fit base model ────────────────────────────────────────────────
    if verbose:
        print("[Forecast] Fitting ElasticNet base model...")
    model_artifacts = fit_base_model(feature_matrix, target, base_features)
    base_returns = _base_forecast(model_artifacts, feature_args=feature_args)

    if verbose:
        n_nz = len(model_artifacts["nonzero_coefs"])
        print(f"[Forecast] Model: {n_nz}/{len(base_features)} features with non-zero coef, "
              f"alpha={model_artifacts['model'].alpha_:.4f}, weekly_signal={list(base_returns.values())[0]:.4f}")

    # ── Step 3: Event adjustment ──────────────────────────────────────────────
    if verbose:
        print("[Forecast] Computing event volatility adjustment...")
    event_adj, events_used = _event_adjustment(equity_summary, today=today)

    if verbose and events_used:
        for e in events_used:
            print(f"  · {e['event_name']} ({e['event_date']}, {e['days_since']}d ago, "
                  f"decay={e['decay_weight']:.3f})")

    # ── Combine & build output table ─────────────────────────────────────────
    target_series = feature_matrix["target"].dropna().iloc[-BASELINE_WINDOW:]
    target_series_std = float(target_series.std())
    current_price = float(feature_matrix[target].dropna().iloc[-1])
    records = []
    for window, days in WINDOWS.items():
        base_ret = base_returns[window]
        ev_adj = event_adj.get(window, 0.0) * target_series_std  # scale adjustment by target's trailing std
        total_ret = base_ret + ev_adj
        direction = "UP" if total_ret > 0 else ("DOWN" if total_ret < 0 else "FLAT")
        magnitude = abs(total_ret)
        projected = current_price * (1 + total_ret) if target == "price" else current_price + total_ret * current_price

        records.append({
            "window": window,
            "horizon_days": days,
            "base_return": round(base_ret, 6),
            "event_adjustment": round(ev_adj, 6),
            "total_return": round(total_ret, 6),
            "direction": direction,
            "magnitude_%": round(magnitude * 100, 4),
            "projected_value": round(projected, 2),
        })

    forecast_df = pd.DataFrame(records).set_index("window")

    return {
        "forecast": forecast_df,
        "base_model": model_artifacts,
        "events_used": events_used,
        "feature_args": feature_args or {},
        "windows": list(WINDOWS.keys()),
        "current_value": current_price,
        "target": target,
        "today": today,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

_ARG_LABEL = {-2: "very bad", -1: "bad", 0: "neutral", 1: "good", 2: "very good"}


def print_forecast_report(result: dict) -> None:
    """Pretty-print the forecast result to stdout."""
    target = result["target"]
    today = result["today"]
    current = result["current_value"]
    feature_args: dict[str, int] = result.get("feature_args", {})
    events = result["events_used"]

    print(f"\n{'=' * 65}")
    print(f"  MFMEA - Future {target.upper()} Forecast")
    print(f"  As of: {today.strftime('%Y-%m-%d')}  |  Current {target}: {current:,.2f}")
    print(f"{'=' * 65}")

    print(f"\n  {'Window':<8} {'Horizon':>10} {'Base Ret':>10} {'Evt Adj':>10} "
          f"{'Total Ret':>10} {'Dir':>6} {'|Mag|%':>8} {'Projected':>12}")
    print(f"  {'-' * 78}")

    for window, row in result["forecast"].iterrows():
        print(
            f"  {window:<8} {row['horizon_days']:>8}d  "
            f"{row['base_return']:>+10.4%} "
            f"{row['event_adjustment']:>+10.4%} "
            f"{row['total_return']:>+10.4%} "
            f"{row['direction']:>6} "
            f"{row['magnitude_%']:>7.3f}% "
            f"{row['projected_value']:>12,.2f}"
        )

    if events:
        print(f"\n  Active Events ({len(events)}):")
        for e in events:
            print(f"    · {e['event_name']:<28} {e['event_date']}  "
                  f"{e['days_since']:>4}d ago  decay={e['decay_weight']:.3f}")

    model = result["base_model"]
    nz = sorted(model["nonzero_coefs"].items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Base Model Features (ElasticNet alpha={model['model'].alpha_:.4f}):")
    print(f"  {'Coef':>8}  {'Arg':>5}  Feature")
    print(f"  {'-' * 60}")
    for feat, coef in nz[:10]:
        arg_val = feature_args.get(feat, 0)
        arg_str = f"{arg_val:+d} ({_ARG_LABEL.get(arg_val, '')})" if arg_val != 0 else "  0 (neutral)"
        print(f"  {coef:>+8.4f}  {arg_str:<18}  {feat}")

    print(f"{'=' * 65}\n")


def save_forecast_report(result: dict, save: bool = True) -> Path | None:
    """Save forecast table to output/ as Excel."""
    if not save:
        return None

    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows

    target = result["target"]
    today = result["today"].strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"mfmea_forecast_{target}_{today}.xlsx"

    wb = Workbook()

    # ── Sheet 1: Forecast ────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Forecast"
    df = result["forecast"].reset_index()
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        ws.append(row)
        if r_idx == 1:
            for cell in ws[r_idx]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="1F4E79")
                cell.font = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(horizontal="center")

    # Colour direction cells
    dir_col = df.columns.get_loc("direction") + 1
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=dir_col, max_col=dir_col):
        for cell in row:
            if cell.value == "UP":
                cell.fill = PatternFill("solid", fgColor="C6EFCE")
            elif cell.value == "DOWN":
                cell.fill = PatternFill("solid", fgColor="FFC7CE")

    # ── Sheet 2: Events Used ─────────────────────────────────────────────────
    ws2 = wb.create_sheet("Events_Used")
    if result["events_used"]:
        events_df = pd.DataFrame([
            {
                "event_name": e["event_name"],
                "event_date": e["event_date"],
                "days_since": e["days_since"],
                "decay_weight": e["decay_weight"],
                **{f"contrib_{w}": e["window_contrib"].get(w, 0.0) for w in WINDOWS},
            }
            for e in result["events_used"]
        ])
        for r_idx, row in enumerate(dataframe_to_rows(events_df, index=False, header=True), 1):
            ws2.append(row)
            if r_idx == 1:
                for cell in ws2[r_idx]:
                    cell.font = Font(bold=True)

    # ── Sheet 3: Model Coefs ─────────────────────────────────────────────────
    ws3 = wb.create_sheet("Model_Coefs")
    coef_df = pd.DataFrame(
        sorted(result["base_model"]["coefs"].items(), key=lambda x: abs(x[1]), reverse=True),
        columns=["feature", "coef"],
    )
    for r_idx, row in enumerate(dataframe_to_rows(coef_df, index=False, header=True), 1):
        ws3.append(row)
        if r_idx == 1:
            for cell in ws3[r_idx]:
                cell.font = Font(bold=True)

    wb.save(path)
    return path