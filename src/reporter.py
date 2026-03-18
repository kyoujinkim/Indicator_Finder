"""
Reporter - generates the final analysis report.

Outputs:
  - Console summary (top N indicators)
  - CSV report saved to output/
  - Excel report with multiple sheets saved to output/
"""

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _impact_type(row: pd.Series) -> str:
    """Classify indicator dominant impact path: Continuous / Shock / Threshold / Mixed."""
    scores = {
        "Continuous": row.get("continuous_score", 0),
        "Shock":      row.get("shock_score", 0),
        "Threshold":  row.get("threshold_score", 0),
    }
    best = max(scores, key=scores.get)
    second = sorted(scores.values(), reverse=True)[1]
    # Only label as dominant if clearly ahead by 50%
    if scores[best] > second * 1.5:
        return best
    return "Mixed"


def _format_feature_name(col: str) -> tuple[str, str]:
    """Split 'sheet__indicator' into (sheet, indicator)."""
    if "__" in col:
        parts = col.split("__", 1)
        return parts[0], parts[1]
    return "Unknown", col


def generate_report(
    scores: pd.DataFrame,
    fs_results: dict,
    target_col: str = "price",
    top_n: int = 20,
    save: bool = True,
) -> None:
    """
    Print and save the MFMEA final report.

    Parameters
    ----------
    scores     : output of compute_relation_scores()
    fs_results : output of run_feature_selection()
    target_col : 'price' or 'eps'
    top_n      : number of top indicators to display
    save       : whether to save CSV/Excel output
    """
    out_dir = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Enrich with metadata
    scores = scores.copy()
    scores["impact_type"] = scores.apply(_impact_type, axis=1)
    scores["sheet"], scores["indicator"] = zip(
        *scores.index.map(_format_feature_name)
    )

    top = scores.head(top_n)

    # ------------------------------------------------------------------ #
    # Console Output
    # ------------------------------------------------------------------ #
    print("=" * 80)
    print(f"  MFMEA Report | Target: {target_col.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    print(f"\n{'Rank':<5} {'Score':>7} {'Type':<11} {'C':>6} {'S':>6} {'T':>6} {'Granger':>10} {'TrigScr':>8}  Indicator")
    print("-" * 90)
    for _, row in top.iterrows():
        gc_str = f"p={row['granger_pval']:.3f}" if not pd.isna(row["granger_pval"]) else "    n/a"
        trig_str = f"{row['trigger_score']:.3f}" if not pd.isna(row.get("trigger_score")) else "   n/a"
        c_str = f"{row['continuous_score']:.3f}"
        i_str = f"{row['irf_score']:.3f}" if not pd.isna(row.get("irf_score")) else "   n/a"
        s_str = f"{row['shock_score']:.3f}"
        t_str = f"{row['threshold_score']:.3f}"
        boruta_flag = " [B]" if row["boruta_confirmed"] else "    "
        print(
            f"{int(row['rank']):<5} {row['final_score']:>7.4f} {row['impact_type']:<11} "
            f"{c_str:>6} {i_str:>6} {s_str:>6} {t_str:>6} {gc_str:>10} {trig_str:>8}  "
            f"{row['sheet']} | {row['indicator'][:45]}{boruta_flag}"
        )

    print("\nColumns: C=Continuous, I=Irf, S=Shock, T=Threshold  |  [B]=Boruta-confirmed")

    # Threshold top triggers
    thresh_cols = [c for c in scores.columns if c == "trigger_score"]
    if thresh_cols:
        thresh_top = scores.nlargest(5, "threshold_score")
        print("\n--- Top Threshold / Indicator-Function Features ---")
        print(f"  {'Indicator':<55} {'Threshold':>10} {'Quantile':>9} {'TrigScr':>8} {'HitRate5':>9} {'Crossings':>10}")
        print("  " + "-" * 105)
        for _, row in thresh_top.iterrows():
            sheet, ind = _format_feature_name(row.name)
            thr = f"{row['threshold']:.3f}" if not pd.isna(row.get("threshold", np.nan)) else "  n/a"
            qtl = f"{row['threshold_quantile']:.2f}" if not pd.isna(row.get("threshold_quantile", np.nan)) else " n/a"
            ts  = f"{row['trigger_score']:.3f}"
            hr  = f"{row.get('hit_rate_5', np.nan):.3f}" if not pd.isna(row.get("hit_rate_5", np.nan)) else "  n/a"
            nc  = f"{int(row.get('n_crossings', 0))}"
            print(f"  {sheet[:20]}|{ind[:33]:<54} {thr:>10} {qtl:>9} {ts:>8} {hr:>9} {nc:>10}")

    # Regime importance
    regime_df = fs_results.get("regime_importance", pd.DataFrame())
    if not regime_df.empty:
        print("\n--- Regime-Aware Top Features ---")
        for regime_col in regime_df.columns:
            top_regime = regime_df[regime_col].nlargest(5)
            print(f"\n  {regime_col}:")
            for feat, val in top_regime.items():
                sheet, ind = _format_feature_name(feat)
                print(f"    {val:.4f}  {sheet} | {ind[:50]}")

    # Boruta summary
    boruta = fs_results.get("boruta", {})
    n_accepted = len(boruta.get("accepted", []))
    n_tentative = len(boruta.get("tentative", []))
    n_rejected = len(boruta.get("rejected", []))
    print(f"\n--- Boruta Selection ---")
    print(f"  Accepted:  {n_accepted}")
    print(f"  Tentative: {n_tentative}")
    print(f"  Rejected:  {n_rejected}")
    print("=" * 90)

    if not save:
        return

    # ------------------------------------------------------------------ #
    # Save CSV
    # ------------------------------------------------------------------ #
    csv_path = out_dir / f"mfmea_report_{target_col}_{timestamp}.csv"
    scores.to_csv(csv_path)
    print(f"\nCSV saved: {csv_path}")

    # ------------------------------------------------------------------ #
    # Save Excel (multi-sheet)
    # ------------------------------------------------------------------ #
    xlsx_path = out_dir / f"mfmea_report_{target_col}_{timestamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Full ranking
        scores.to_excel(writer, sheet_name="Full_Ranking")

        # Top N
        top.to_excel(writer, sheet_name=f"Top_{top_n}")

        # MI scores
        mi = fs_results.get("mi_scores", pd.Series(dtype=float))
        if not mi.empty:
            mi.reset_index().rename(columns={0: "mi_score"}).to_excel(
                writer, sheet_name="MI_Scores", index=False
            )

        # Regime importance
        if not regime_df.empty:
            regime_df.to_excel(writer, sheet_name="Regime_Importance")

        # Threshold analysis
        thresh_cols = [c for c in scores.columns if c in (
            "threshold", "threshold_quantile", "regime_diff", "threshold_pval",
            "trigger_score", "n_crossings", "hit_rate_1", "hit_rate_3",
            "hit_rate_5", "hit_rate_10", "asymmetry",
        )]
        if thresh_cols:
            scores[thresh_cols].to_excel(writer, sheet_name="Threshold_Analysis")

        # Boruta
        boruta_rows = (
            [{"feature": f, "status": "accepted"} for f in boruta.get("accepted", [])]
            + [{"feature": f, "status": "tentative"} for f in boruta.get("tentative", [])]
            + [{"feature": f, "status": "rejected"} for f in boruta.get("rejected", [])]
        )
        if boruta_rows:
            pd.DataFrame(boruta_rows).to_excel(writer, sheet_name="Boruta", index=False)

    print(f"Excel saved: {xlsx_path}")


# --------------------------------------------------------------------------- #
# Named-Event Impact Report
# --------------------------------------------------------------------------- #

def generate_event_impact_report(
    event_impact: dict,
    target_col: str = "price",
    top_n: int = 10,
    save: bool = True,
) -> None:
    """
    Print and save the named-event impact report.

    Parameters
    ----------
    event_impact : output of run_event_impact()
    target_col   : 'price' or 'eps'
    top_n        : number of top indicators to show per event
    save         : whether to save Excel output
    """
    eq_impact   = event_impact.get("equity_impact",      pd.DataFrame())
    eq_summary  = event_impact.get("equity_summary",     pd.DataFrame())
    ind_summary = event_impact.get("indicator_summary",  pd.DataFrame())
    feat_score  = event_impact.get("feature_score",      pd.Series(dtype=float))

    if eq_impact.empty:
        print("\n[Event Impact] No events matched in the data range.")
        return

    windows = [1, 2, 4, 8]
    ret_cols = [f"ret_+{w}w" for w in windows if f"ret_+{w}w" in eq_impact.columns]
    ar_cols  = [f"ar_+{w}w"  for w in windows if f"ar_+{w}w"  in eq_impact.columns]

    # ------------------------------------------------------------------ #
    # Console: Equity response per event
    # ------------------------------------------------------------------ #
    print()
    print("=" * 90)
    print(f"  Named-Event Impact Report | Target: {target_col.upper()}")
    print("=" * 90)

    # Header
    w_labels = "  ".join(f"+{w}w".rjust(7) for w in windows if f"ret_+{w}w" in eq_impact.columns)
    print(f"\n{'Event':<25} {'Date':<12} {'EvtVal':>10}  {w_labels}  (% ret from event)")
    print("-" * 90)

    for _, row in eq_impact.sort_values("event_date").iterrows():
        rets = "  ".join(
            f"{row[c]*100:>+6.1f}%" if not pd.isna(row.get(c)) else "   n/a "
            for c in ret_cols
        )
        print(f"{row['event_name']:<25} {str(row['matched_date'].date()):<12} "
              f"{row['event_value']:>10.1f}  {rets}")

    # Summary for recurring events
    recurring = eq_summary[eq_summary["n_occurrences"] > 1] if not eq_summary.empty else pd.DataFrame()
    if not recurring.empty:
        print(f"\n--- Recurring Event Averages ---")
        print(f"{'Event':<25} {'N':>3}  {w_labels}  (mean % ret)")
        print("-" * 70)
        for _, row in recurring.iterrows():
            rets = "  ".join(
                f"{row[c]*100:>+6.1f}%" if not pd.isna(row.get(c)) else "   n/a "
                for c in ret_cols
            )
            print(f"{row['event_name']:<25} {int(row['n_occurrences']):>3}  {rets}")

    # ------------------------------------------------------------------ #
    # Console: Top indicator reactions per event
    # ------------------------------------------------------------------ #
    if not ind_summary.empty:
        delta_cols = [f"delta_+{w}w" for w in windows if f"delta_+{w}w" in ind_summary.columns]
        print(f"\n--- Top {top_n} Indicators Reacting to Each Event (mean |Δz-score|) ---")

        for event_name, grp in ind_summary.groupby("event_name"):
            top = grp.nlargest(top_n, "mean_abs_delta")
            print(f"\n  {event_name}")
            d_hdr = "  ".join(f"+{w}w".rjust(7) for w in windows if f"delta_+{w}w" in delta_cols)
            print(f"  {'Indicator':<60} {d_hdr}  (Δz)")
            print("  " + "-" * 90)
            for _, r in top.iterrows():
                sheet, ind = _format_feature_name(r["feature"])
                deltas = "  ".join(
                    f"{r[c]:>+6.2f} " if not pd.isna(r.get(c)) else "   n/a "
                    for c in delta_cols
                )
                print(f"  {sheet[:15]}|{ind[:43]:<59} {deltas}")

    # Overall most event-reactive features
    if not feat_score.empty:
        print(f"\n--- Top {top_n} Most Event-Reactive Features (across all events) ---")
        for feat, score in feat_score.head(top_n).items():
            sheet, ind = _format_feature_name(feat)
            print(f"  {score:.3f}  {sheet} | {ind[:60]}")

    print("=" * 90)

    if not save:
        return

    # ------------------------------------------------------------------ #
    # Save Excel
    # ------------------------------------------------------------------ #
    out_dir   = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = out_dir / f"mfmea_event_impact_{target_col}_{timestamp}.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Per-occurrence equity response
        eq_impact.to_excel(writer, sheet_name="Equity_Impact", index=False)

        # Per-event-name equity summary
        if not eq_summary.empty:
            eq_summary.to_excel(writer, sheet_name="Equity_Summary", index=False)

        # Per-event × feature indicator response summary
        if not ind_summary.empty:
            ind_summary.to_excel(writer, sheet_name="Indicator_Response", index=False)

        # Overall feature event-reactiveness ranking
        if not feat_score.empty:
            feat_score.reset_index().rename(
                columns={"index": "feature", 0: "event_reactiveness"}
            ).to_excel(writer, sheet_name="Feature_Event_Scores", index=False)

    print(f"\nEvent Impact Excel saved: {xlsx_path}")


# --------------------------------------------------------------------------- #
# Temporal Comparison Report
# --------------------------------------------------------------------------- #

_STATUS_ARROW = {"risen": "↑", "fallen": "↓", "stable": "=", "new": "NEW", "dropped": "DEL"}
_SCORE_COLS = [
    "rank", "final_score",
    "continuous_score", "irf_score", "shock_score", "threshold_score",
]


def generate_comparison_report(
    comparison: dict,
    target_col: str = "price",
    top_n: int = 20,
    save: bool = True,
) -> None:
    """
    Print and save the temporal importance-change report.

    Parameters
    ----------
    comparison : output of compare_periods()
    target_col : 'price' or 'eps'
    top_n      : how many features to show in the main table
    save       : whether to save CSV/Excel output
    """
    out_dir = _ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    delta    = comparison["delta"].copy()
    w_cur    = comparison["window_current"]
    w_past   = comparison["window_past"]
    scores_c = comparison["current"]
    scores_p = comparison["past"]

    if delta.empty:
        print("No comparison data available.")
        return

    # Enrich with readable names
    delta["sheet"], delta["indicator"] = zip(
        *delta["feature"].map(_format_feature_name)
    )

    # ------------------------------------------------------------------ #
    # Console Output
    # ------------------------------------------------------------------ #
    print("=" * 95)
    print(f"  MFMEA Temporal Comparison | Target: {target_col.upper()}")
    print(f"  Current : {w_cur[0].date()} → {w_cur[1].date()}")
    print(f"  Past    : {w_past[0].date()} → {w_past[1].date()}")
    print("=" * 95)

    # --- Top movers (by |score_delta|) ---
    movers = delta.dropna(subset=["score_delta"]).copy()
    movers = movers[movers["status"] != "new"].sort_values("score_delta", key=abs, ascending=False)

    print(f"\n{'':5} {'Cur':>5} {'Past':>5} {'ΔScore':>8} {'ΔRank':>7}  {'Dom.Change':<12}  Indicator")
    print("-" * 95)
    shown = 0
    for _, row in movers.iterrows():
        if shown >= top_n:
            break
        arrow = _STATUS_ARROW.get(row["status"], "?")
        rank_c = f"#{int(row['rank_current'])}" if not pd.isna(row.get("rank_current")) else "  -"
        rank_p = f"#{int(row['rank_past'])}"    if not pd.isna(row.get("rank_past"))    else "  -"
        rd     = row["rank_delta"]
        rd_str = f"+{int(rd)}" if rd > 0 else str(int(rd))
        score_d = row["score_delta"]
        sd_str = f"{score_d:+.4f}"
        print(
            f"  {arrow:<3} {rank_c:>5} {rank_p:>5} {sd_str:>8} {rd_str:>7}  {row['dominant_change']:<12}  "
            f"{row['sheet']} | {row['indicator'][:45]}"
        )
        shown += 1

    # --- Rising stars ---
    risen = delta[delta["status"] == "risen"].nlargest(5, "score_delta")
    if not risen.empty:
        print(f"\n--- Rising Stars (score improved > +0.05) ---")
        for _, row in risen.iterrows():
            sheet, ind = _format_feature_name(row["feature"])
            print(f"  ↑  #{int(row['rank_current']):<4} (+{row['score_delta']:.4f})  [{row['dominant_change']}]  {sheet} | {ind[:50]}")

    # --- Falling stars ---
    fallen = delta[delta["status"] == "fallen"].nsmallest(5, "score_delta")
    if not fallen.empty:
        print(f"\n--- Falling Stars (score dropped > -0.05) ---")
        for _, row in fallen.iterrows():
            sheet, ind = _format_feature_name(row["feature"])
            print(f"  ↓  #{int(row['rank_current']):<4} ({row['score_delta']:.4f})  [{row['dominant_change']}]  {sheet} | {ind[:50]}")

    # --- New / Dropped ---
    new_feats = delta[delta["status"] == "new"]
    dropped   = delta[delta["status"] == "dropped"]
    if not new_feats.empty:
        print(f"\n--- New Features (appeared in current window) ---")
        for _, row in new_feats.iterrows():
            sheet, ind = _format_feature_name(row["feature"])
            print(f"  NEW  {sheet} | {ind[:60]}")
    if not dropped.empty:
        print(f"\n--- Dropped Features (absent in current window) ---")
        for _, row in dropped.iterrows():
            sheet, ind = _format_feature_name(row["feature"])
            print(f"  DEL  {sheet} | {ind[:60]}")

    # --- Per-path change summary ---
    print(f"\n--- Average Per-Path Score Change ---")
    for path in ("continuous", "irf", "shock", "threshold"):
        col = f"{path}_delta"
        if col in delta.columns:
            avg = delta[col].mean()
            print(f"  {path.capitalize():<12}: {avg:+.4f}")

    print("=" * 95)

    if not save:
        return

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    csv_path = out_dir / f"mfmea_comparison_{target_col}_{timestamp}.csv"
    delta.to_csv(csv_path, index=False)
    print(f"\nComparison CSV saved: {csv_path}")

    xlsx_path = out_dir / f"mfmea_comparison_{target_col}_{timestamp}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        delta.to_excel(writer, sheet_name="Delta", index=False)

        # Ranked current scores (key columns only)
        if not scores_c.empty:
            key_cols = [c for c in _SCORE_COLS if c in scores_c.columns]
            scores_c[key_cols].to_excel(writer, sheet_name="Current_Scores")

        # Ranked past scores
        if not scores_p.empty:
            key_cols = [c for c in _SCORE_COLS if c in scores_p.columns]
            scores_p[key_cols].to_excel(writer, sheet_name="Past_Scores")

        # Top movers sheet
        movers.head(top_n).to_excel(writer, sheet_name="Top_Movers", index=False)

    print(f"Comparison Excel saved: {xlsx_path}")
