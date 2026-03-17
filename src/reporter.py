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
