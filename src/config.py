"""
MFMEA Project Configuration
============================
Central registry for all project-wide constants.
Edit values here; all modules import from this file.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "output/mfmea"
FORECAST_DIR = BASE_DIR / "output/forecast"

DATA_XLSX    = DATA_DIR / "data.xlsx"
EQUITY_XLSX  = DATA_DIR / "equity.xlsx"
IMPACTS_FILE = DATA_DIR / "impacts.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

# Row index (0-based) where actual data begins in both data.xlsx and equity.xlsx
DATA_START_ROW = 15

# ─────────────────────────────────────────────────────────────────────────────
# Event Analysis  (event_impact.py)
# ─────────────────────────────────────────────────────────────────────────────

# Post-event windows to track (weeks)
RESPONSE_WINDOWS = [1, 2, 4, 8, 13, 17, 21, 26]

# Pre-event baseline length for abnormal-return calculation (weeks)
BASELINE_WINDOW = 12

# ─────────────────────────────────────────────────────────────────────────────
# Relation Scoring  (relation_scorer.py)
# ─────────────────────────────────────────────────────────────────────────────

# Columns carried forward for temporal comparison and output sheets
SCORE_COLS = [
    "rank", "final_score",
    "continuous_score", "irf_score", "shock_score", "threshold_score",
]

# Top-level path weights: (continuous, irf, shock, threshold)  — must sum to 1
PATH_WEIGHTS = (0.35, 0.15, 0.35, 0.15)

# Bonus added to Boruta-confirmed features
BORUTA_BONUS = 0.15

# Per-metric sub-weights within each path
METRIC_WEIGHTS = {
    # Continuous path
    "granger":          0.35,
    "transfer_entropy": 0.35,
    "elasticnet":       0.30,
    # IRF path
    "irf":              1.00,
    # Shock path
    "event":            0.50,
    "shap":             0.50,
    # Threshold path
    "regime_diff":      0.40,
    "trigger_score":    0.40,
    "asymmetry":        0.20,
}

# ─────────────────────────────────────────────────────────────────────────────
# Forecast  (future_forecaster.py, forecast.py)
# ─────────────────────────────────────────────────────────────────────────────

# Forecast horizons: label -> calendar days
FORECAST_WINDOWS: dict[str, int] = {
    "1w" :   7,
    "2w" :   14,
    "4w" :   28,
    "8w" :   56,
    "13w":   91,
    "17w":   126,
    "21w":   147,
    "26w":   182,
}

# Maps each effective time-shifted week count to the ar_ column in Equity_Summary
WINDOW_TO_AR_COL: dict[str, str] = {
    "1w":  "ar_+1w",
    "2w":  "ar_+2w",
    "4w":  "ar_+4w",
    "8w":  "ar_+8w",
    "13w": "ar_+13w",
    "17w": "ar_+17w",
    "21w": "ar_+21w",
    "26w": "ar_+26w",
}

# EWMA span (weeks) for current market state estimation
EWMA_SPAN = 8

# EWMA nudge per qualitative rating unit  (arg × std × ARG_SCALE)
ARG_SCALE = 1.0

# Horizon dampening exponent γ: base_return = weekly_signal × n_weeks^γ
# 1.0 = linear, 0.5 = sqrt; lower values shrink long-horizon estimates
HORIZON_DAMPENING = 0.6

# Qualitative rating labels (used in both module display and CLI prompt)
ARG_LABEL: dict[int, str] = {
    -2: "very bad",
    -1: "bad",
     0: "neutral",
     1: "good",
     2: "very good",
}