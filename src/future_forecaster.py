"""
Future Forecaster for MFMEA.
============================================================
Pure analytics module — no interactive I/O.
See readme_future_forecast.md for architecture details.

Two-step forecast:
  Step 1 - Base Trend Model
    · select_base_features : pick top-N from report, filter fallen, deduplicate
    · fit_base_model        : ElasticNetCV on stationary returns; EWMA current state
    · _base_forecast        : weekly_signal × n_weeks^γ (dampened accumulation)

  Step 2 - Event Volatility Adjustment
    · _load_events          : parse data/impacts.txt (name, date, probability)
    · _active_windows       : time-shifted ar_ column mapping per forecast slot
    · _event_adjustment     : decay × probability × ar_ × target_std per window

Public API:
    prepare_model(feature_matrix, target, top_n)
        -> (model_artifacts, base_features)   [call before prompting user]

    run_forecast(feature_matrix, target, feature_args, top_n, today,
                 ci_level, horizon_damp, verbose)
        -> result dict

    print_forecast_report(result)
    save_forecast_report(result, save)
"""

import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from .config import (
    OUTPUT_DIR, DATA_DIR, FORECAST_DIR, IMPACTS_FILE,
    FORECAST_WINDOWS as WINDOWS,
    WINDOW_TO_AR_COL as _WINDOW_TO_AR_COL,
    EWMA_SPAN,
    ARG_SCALE as _ARG_SCALE,
    HORIZON_DAMPENING,
    BASELINE_WINDOW,
    ARG_LABEL as _ARG_LABEL,
)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers (private)
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Path:
    files = sorted(glob.glob(str(OUTPUT_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"No output file found: {OUTPUT_DIR / pattern}")
    return Path(files[-1])


def _load_tabular(pattern_csv: str, pattern_xlsx: str, sheet: str = "Delta") -> pd.DataFrame:
    """Load the latest csv; fall back to the first sheet of xlsx if csv absent."""
    try:
        return pd.read_csv(_find_latest(pattern_csv))
    except FileNotFoundError:
        return pd.read_excel(_find_latest(pattern_xlsx), sheet_name=sheet)


def load_report(code:str, target: str) -> pd.DataFrame:
    return _load_tabular(
        f"mfmea_report_{code}_{target}_*.csv",
        f"mfmea_report_{code}_{target}_*.xlsx",
        sheet="Full_Ranking",
    )


def load_comparison(code: str, target: str) -> pd.DataFrame:
    return _load_tabular(
        f"mfmea_comparison_{code}_{target}_*.csv",
        f"mfmea_comparison_{code}_{target}_*.xlsx",
        sheet="Delta",
    )


def _load_equity_summary(code:str, target: str) -> pd.DataFrame:
    path = _find_latest(f"mfmea_event_impact_{code}_{target}_*.xlsx")
    return pd.read_excel(path, sheet_name="Equity_Summary")


def _load_events() -> pd.DataFrame:
    """Parse data/impacts.txt -> DataFrame[event_name, event_date, probability]."""
    rows = []
    with open(IMPACTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            try:
                date = pd.to_datetime(parts[1].strip())
            except Exception:
                continue
            try:
                prob = float(parts[2].strip()) if len(parts) >= 3 else 1.0
            except ValueError:
                prob = 1.0
            rows.append({"event_name": name, "event_date": date, "probability": prob})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Event window helpers (private)
# ─────────────────────────────────────────────────────────────────────────────

def _active_windows(days_since: int) -> dict[str, str]:
    """
    For each forecast slot (1w/2w/4w/13w/26w) return the time-shifted ar_ key.

    Formula: w1 = abs(-(days_since + n*7) // 7)
    Maps the remaining horizon from today to the nearest available ar_ column.
    Slots whose effective w1 >= 26 are excluded (event too old for that window).
    """
    adjusted: dict[str, str] = {}
    for label, n in zip(["1w", "2w", "4w", "13w", "26w"], [1, 2, 4, 13, 26]):
        w1 = abs(-(days_since + n * 7) // 7)
        if w1 < 1:
            adjusted[label] = "1w"
        elif w1 < 2:
            adjusted[label] = "2w"
        elif w1 < 4:
            adjusted[label] = "4w"
        elif w1 < 13:
            adjusted[label] = "13w"
        elif w1 < 26:
            adjusted[label] = "26w"
        # else: slot inactive — omit from dict
    return adjusted


def _event_decay_weight(days_since: int, half_life: int = 14) -> float:
    lam = np.log(2) / half_life
    return float(np.exp(-lam * days_since))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature selection & model fitting
# ─────────────────────────────────────────────────────────────────────────────

def select_base_features(
    report: pd.DataFrame,
    comparison: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """
    Pick top_n features by final_score, excluding 'fallen' and deduplicating
    by indicator name (keeping the highest-scoring feature per indicator).
    """
    fallen: set[str] = set()
    if "status" in comparison.columns:
        fallen = set(comparison.loc[comparison["status"] == "fallen", "feature"])

    candidates = (
        report[~report["feature"].isin(fallen)]
        .sort_values("final_score", ascending=False)
        .head(top_n * 2)
    )

    seen: set[str] = set()
    selected: list[str] = []
    for _, row in candidates.iterrows():
        ind = row.get("indicator", row["feature"])
        if ind not in seen:
            seen.add(ind)
            selected.append(row["feature"])
        if len(selected) >= top_n:
            break
    return selected


def fit_base_model(
    feature_matrix: pd.DataFrame,
    target: str,
    base_features: list[str],
) -> dict:
    """
    Fit ElasticNetCV on selected features vs. stationary target returns.

    Returns a model-artifacts dict with:
      model, scaler, features, coefs, nonzero_coefs,
      trailing_ewma, trailing_feat_std, trailing_std,
      residual_std, nudge_ewma (populated after _base_forecast)
    """
    available = [f for f in base_features if f in feature_matrix.columns]
    if not available:
        raise ValueError("None of the selected base features are present in the feature matrix.")

    df = feature_matrix[available + ["target"]].dropna()
    X  = df[available].values
    y  = df["target"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
        cv=5, max_iter=5000, random_state=42,
    )
    model.fit(X_scaled, y)

    trailing_ewma = {
        f: float(feature_matrix[f].ewm(span=EWMA_SPAN).mean().iloc[-1])
        for f in available
    }
    trailing_feat_std = {
        f: float(feature_matrix[f].dropna().std())
        for f in available
    }

    coefs    = dict(zip(available, model.coef_))
    nonzero  = {f: c for f, c in coefs.items() if abs(c) > 1e-6}
    residual_std = float(np.std(y - model.predict(X_scaled)))

    return {
        "model":             model,
        "scaler":            scaler,
        "features":          available,
        "coefs":             coefs,
        "nonzero_coefs":     nonzero,
        "trailing_ewma":     trailing_ewma,
        "trailing_feat_std": trailing_feat_std,
        "trailing_std":      float(np.std(y)),
        "residual_std":      residual_std,
        "nudge_ewma":        {},   # populated by _base_forecast
    }


def prepare_model(
    feature_matrix: pd.DataFrame,
    code: str,
    target: str,
    top_n: int = 10,
) -> tuple[dict, list[str]]:
    """
    Convenience wrapper: load reports, select features, fit model.
    Returns (model_artifacts, base_features).

    Call this before prompting the user so coefs are available for display,
    then pass the collected feature_args into run_forecast.
    """
    report     = load_report(code, target)
    comparison = load_comparison(code, target)
    features   = select_base_features(report, comparison, top_n)
    artifacts  = fit_base_model(feature_matrix, target, features)
    return artifacts, features


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Base forecast with qualitative nudge
# ─────────────────────────────────────────────────────────────────────────────

def _base_forecast(
    model_artifacts: dict,
    feature_args: dict[str, int] | None = None,
    horizon_damp: float = HORIZON_DAMPENING,
) -> tuple[dict, dict[str, float]]:
    """
    Apply per-indicator qualitative nudges to EWMA values, run model prediction,
    accumulate weekly signal across windows using dampened horizon scaling.

    Returns (updated model_artifacts, {window: base_return}).
    """
    features     = model_artifacts["features"]
    feat_std     = model_artifacts["trailing_feat_std"]
    feature_args = feature_args or {}

    nudged_ewma = np.array([
        model_artifacts["trailing_ewma"][f]
        + feature_args.get(f, 0) * _ARG_SCALE * feat_std.get(f, 0.0)
        for f in features
    ])
    model_artifacts["nudge_ewma"] = dict(zip(features, nudged_ewma))

    X_scaled      = model_artifacts["scaler"].transform(nudged_ewma.reshape(1, -1))
    weekly_signal = float(model_artifacts["model"].predict(X_scaled)[0])

    returns = {
        w: weekly_signal * ((days / 7.0) ** horizon_damp)
        for w, days in WINDOWS.items()
    }
    return model_artifacts, returns


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Event volatility adjustment
# ─────────────────────────────────────────────────────────────────────────────

def _event_adjustment(
    equity_summary: pd.DataFrame,
    today: datetime | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """
    For each recent event: decay × probability × ar_+Xw × target_std.
    ar_ values are Z-scores; multiplying by target_std converts to return units.
    """
    if today is None:
        today = datetime.today()

    events_df   = _load_events()
    adjustments = {w: 0.0 for w in WINDOWS}
    events_used: list[dict] = []

    for _, evt in events_df.iterrows():
        event_name: str   = evt["event_name"]
        event_date        = evt["event_date"]
        event_prob: float = float(evt["probability"])

        days_since = (today - event_date.to_pydatetime()).days
        active     = _active_windows(days_since)
        if not active:
            continue

        decay = _event_decay_weight(days_since)
        match = equity_summary[equity_summary["event_name"] == event_name]
        if match.empty:
            continue

        row    = match.iloc[0]
        contribs: dict[str, float] = {}
        for slot, lookup_window in active.items():
            ar_col = _WINDOW_TO_AR_COL[lookup_window]
            if ar_col in row.index and pd.notna(row[ar_col]):
                c = float(row[ar_col]) * decay * event_prob
                adjustments[slot] += c
                contribs[slot]     = c

        if contribs:
            events_used.append({
                "event_name":   event_name,
                "event_date":   event_date.strftime("%Y-%m-%d"),
                "days_since":   days_since,
                "probability":  event_prob,
                "decay_weight": round(decay, 4),
                "window_contrib": contribs,
            })

    return adjustments, events_used


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_forecast(
    feature_matrix: pd.DataFrame,
    code: str,
    target: str,
    feature_args: dict[str, int] | None = None,
    model_artifacts: dict | None = None,
    top_n: int = 10,
    today: datetime | None = None,
    ci_level: float = 0.95,
    horizon_damp: float = HORIZON_DAMPENING,
    verbose: bool = True,
) -> dict:
    """
    Main forecasting entry point.

    Parameters
    ----------
    feature_matrix  : output of build_feature_matrix()
    target          : 'price' or 'eps'
    feature_args    : per-indicator qualitative scores {feature: -2..+2}
                      (collect via prompt_feature_args in forecast.py, or pass None)
    model_artifacts : pre-fitted model from prepare_model(); re-fitted if None
    top_n           : number of base indicators
    today           : reference date (defaults to datetime.today())
    ci_level        : confidence interval level (default 0.95)
    horizon_damp    : γ exponent for horizon scaling (default HORIZON_DAMPENING)
    verbose         : print progress

    Returns
    -------
    dict with keys: forecast, base_model, events_used, feature_args,
                    ci_level, windows, current_value, target, today
    """
    if today is None:
        today = datetime.today()

    if verbose:
        print("[Forecast] Loading output reports...")
    equity_summary = _load_equity_summary(code, target)

    # Reuse pre-fitted model if supplied (avoids redundant refitting)
    if model_artifacts is None:
        report, comparison = load_report(code, target), load_comparison(code, target)
        base_features      = select_base_features(report, comparison, top_n)
        if verbose:
            print(f"[Forecast] Base features selected ({len(base_features)}): "
                  f"{base_features[:3]}...")
        if verbose:
            print("[Forecast] Fitting ElasticNet base model...")
        model_artifacts = fit_base_model(feature_matrix, target, base_features)

    model_artifacts, base_returns = _base_forecast(
        model_artifacts, feature_args=feature_args, horizon_damp=horizon_damp
    )

    if verbose:
        n_nz = len(model_artifacts["nonzero_coefs"])
        print(f"[Forecast] Model: {n_nz}/{len(model_artifacts['features'])} non-zero coefs, "
              f"alpha={model_artifacts['model'].alpha_:.4f}, "
              f"1w signal={list(base_returns.values())[0]:+.4f}")

    if verbose:
        print("[Forecast] Computing event volatility adjustment...")
    event_adj, events_used = _event_adjustment(equity_summary, today=today)

    if verbose:
        for e in events_used:
            print(f"  · {e['event_name']} ({e['event_date']}, {e['days_since']}d ago, "
                  f"prob={e['probability']:.2f}, decay={e['decay_weight']:.3f})")

    # Combine base + event adjustment into forecast table
    z             = float(_norm.ppf((1 + ci_level) / 2))
    residual_std  = model_artifacts["residual_std"]
    target_std    = float(feature_matrix["target"].dropna().iloc[-BASELINE_WINDOW:].std())
    current_value = float(feature_matrix[target].dropna().iloc[-1])

    def _proj(r: float) -> float:
        return round(current_value * (1 + r), 2)

    records = []
    for window, days in WINDOWS.items():
        n_weeks   = days / 7.0
        base_ret  = base_returns[window]
        ev_adj    = event_adj.get(window, 0.0) * target_std
        total_ret = base_ret + ev_adj
        margin    = z * residual_std * np.sqrt(n_weeks)

        records.append({
            "window":           window,
            "horizon_days":     days,
            "base_return":      round(base_ret, 6),
            "event_adjustment": round(ev_adj, 6),
            "total_return":     round(total_ret, 6),
            "direction":        "UP" if total_ret > 0 else ("DOWN" if total_ret < 0 else "FLAT"),
            "magnitude_%":      round(abs(total_ret) * 100, 4),
            "projected_value":  _proj(total_ret),
            "ci_lower":         round(total_ret - margin, 6),
            "ci_upper":         round(total_ret + margin, 6),
            "proj_lower":       _proj(total_ret - margin),
            "proj_upper":       _proj(total_ret + margin),
        })

    return {
        "forecast":      pd.DataFrame(records).set_index("window"),
        "base_model":    model_artifacts,
        "events_used":   events_used,
        "feature_args":  feature_args or {},
        "ci_level":      ci_level,
        "windows":       list(WINDOWS.keys()),
        "current_value": current_value,
        "target":        target,
        "today":         today,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_forecast_report(result: dict) -> None:
    target       = result["target"]
    today        = result["today"]
    current      = result["current_value"]
    feature_args = result.get("feature_args", {})
    events       = result["events_used"]
    ci_pct       = int(result.get("ci_level", 0.95) * 100)

    print(f"\n{'=' * 85}")
    print(f"  MFMEA - Future {target.upper()} Forecast")
    print(f"  As of: {today.strftime('%Y-%m-%d')}  |  Current {target}: {current:,.2f}  |  CI: {ci_pct}%")
    print(f"{'=' * 85}")
    print(f"\n  {'Window':<6} {'Horizon':>8} {'Base':>9} {'EvtAdj':>9} "
          f"{'Total':>9} {'Dir':>5} {'Projected':>12} {'Lower':>12} {'Upper':>12}  ({ci_pct}% CI)")
    print(f"  {'-' * 95}")

    for window, row in result["forecast"].iterrows():
        dir_str = "UP" if row["direction"] == "UP" else ("DN" if row["direction"] == "DOWN" else "--")
        print(
            f"  {window:<6} {row['horizon_days']:>6}d  "
            f"{row['base_return']:>+9.3%} "
            f"{row['event_adjustment']:>+9.3%} "
            f"{row['total_return']:>+9.3%} "
            f"{dir_str:>5} "
            f"{row['projected_value']:>12,.2f} "
            f"{row['proj_lower']:>12,.2f} "
            f"{row['proj_upper']:>12,.2f}"
        )

    if events:
        print(f"\n  Active Events ({len(events)}):")
        for e in events:
            print(f"    · {e['event_name']:<30} {e['event_date']}  "
                  f"{e['days_since']:>4}d ago  prob={e['probability']:.2f}  decay={e['decay_weight']:.3f}")

    model = result["base_model"]
    nz    = sorted(model["nonzero_coefs"].items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Base Model Features (ElasticNet alpha={model['model'].alpha_:.4f}):")
    print(f"  {'Coef':>8}  {'Current EWMA':>13}  {'Expect':<18}  Feature")
    print(f"  {'-' * 80}")
    for feat, coef in nz[:10]:
        arg_val = feature_args.get(feat, 0)
        arg_str = f"{arg_val:+d} ({_ARG_LABEL.get(arg_val, '')})" if arg_val != 0 else "0"
        cur_val = model["trailing_ewma"].get(feat, float("nan"))
        print(f"  {coef:>+8.4f}  {cur_val:>13.4f}  {arg_str:<18}  {feat}")

    print(f"{'=' * 85}\n")


def save_forecast_report(code: str, result: dict, save: bool = True) -> Path | None:
    if not save:
        return None

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows

    target = result["target"]
    path   = FORECAST_DIR / f"mfmea_forecast_{code}_{target}_{result['today'].strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb     = Workbook()

    # Sheet 1: Forecast
    ws = wb.active
    ws.title = "Forecast"
    df = result["forecast"].reset_index()
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        ws.append(row)
        if r_idx == 1:
            for cell in ws[r_idx]:
                cell.font      = Font(bold=True, color="FFFFFF")
                cell.fill      = PatternFill("solid", fgColor="1F4E79")
                cell.alignment = Alignment(horizontal="center")

    dir_col = df.columns.get_loc("direction") + 1
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=dir_col, max_col=dir_col):
        for cell in row:
            if cell.value == "UP":
                cell.fill = PatternFill("solid", fgColor="C6EFCE")
            elif cell.value == "DOWN":
                cell.fill = PatternFill("solid", fgColor="FFC7CE")

    # Sheet 2: Events Used
    ws2 = wb.create_sheet("Events_Used")
    if result["events_used"]:
        events_df = pd.DataFrame([
            {
                "event_name":   e["event_name"],
                "event_date":   e["event_date"],
                "days_since":   e["days_since"],
                "probability":  e["probability"],
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

    # Sheet 3: Model Coefs
    ws3      = wb.create_sheet("Model_Coefs")
    model    = result["base_model"]
    fa       = result.get("feature_args", {})
    coef_df  = pd.DataFrame([
        {
            "feature":       feat,
            "coef":          coef,
            "current_ewma":  model["trailing_ewma"].get(feat, float("nan")),
            "nudged_ewma":   model["nudge_ewma"].get(feat, float("nan")),
            "expect":        fa.get(feat, 0),
            "expect_label":  _ARG_LABEL.get(fa.get(feat, 0), ""),
        }
        for feat, coef in sorted(model["coefs"].items(), key=lambda x: abs(x[1]), reverse=True)
    ])
    for r_idx, row in enumerate(dataframe_to_rows(coef_df, index=False, header=True), 1):
        ws3.append(row)
        if r_idx == 1:
            for cell in ws3[r_idx]:
                cell.font = Font(bold=True)

    wb.save(path)
    return path