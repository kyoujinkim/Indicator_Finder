"""
Data loader for MFMEA.
Parses data/data.xlsx (economic indicators) and data/equity.xlsx (price/EPS).

data.xlsx structure:
  - Sheet name: [Category]_[Frequency]
  - Row 1  (index 0) : column headers ("Date", indicator names...)
  - Rows 2-15 (index 1-14): metadata / skip
  - Row 16+ (index 15+): timestamp + numeric data

equity.xlsx structure:
  - Sheet: "equity"
  - Row 15 (index 14): column labels ("D A T E", price col, EPS col)
  - Row 16+ (index 15+): datetime, price, EPS
"""

import pandas as pd
import re
from pathlib import Path

from .config import DATA_DIR, DATA_XLSX, EQUITY_XLSX, DATA_START_ROW

_DATA_START_ROW = DATA_START_ROW  # alias kept for internal use


def _parse_frequency(sheet_name: str) -> str:
    """Extract frequency tag from sheet name (e.g. '매크로_월별' -> 'Monthly')."""
    freq_map = {
        "일": "Daily",
        "주": "Weekly",
        "월": "Monthly",
        "분기": "Quarterly",
        "연": "Annual",
    }
    for kor, eng in freq_map.items():
        if kor in sheet_name:
            return eng
    return "Unknown"


def load_indicators() -> dict[str, pd.DataFrame]:
    """
    Returns a dict keyed by sheet name.
    Each value is a DataFrame indexed by date with numeric indicator columns.
    """
    xl = pd.ExcelFile(DATA_XLSX)
    result: dict[str, pd.DataFrame] = {}

    for sheet in xl.sheet_names:
        # Read everything without imposing header so we control parsing
        raw = pd.read_excel(DATA_XLSX, sheet_name=sheet, header=None)
        if raw.shape[0] < _DATA_START_ROW + 1:
            continue

        # Row 0 = column names
        col_names = raw.iloc[0].tolist()
        # Data starts at row 15 (0-indexed)
        data = raw.iloc[_DATA_START_ROW:].copy()
        data.columns = col_names

        # First column is always date
        date_col = col_names[0]
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.dropna(subset=[date_col])
        data = data.set_index(date_col).sort_index()

        # Keep only numeric columns
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna(how="all", axis=1)

        freq = _parse_frequency(sheet)
        data.attrs["sheet"] = sheet
        data.attrs["frequency"] = freq
        data.attrs["category"] = sheet.split("_")[0] if "_" in sheet else sheet

        result[sheet] = data

    return result


def load_equity(target, code) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['price', 'eps'] indexed by date.
    """
    raw = pd.read_excel(EQUITY_XLSX, sheet_name=target, header=None)

    # Row 14 (index 14) contains the column labels
    col_row = raw.iloc[0].tolist()
    data = raw.iloc[_DATA_START_ROW:].copy()
    data.columns = col_row

    date_col_name = col_row[0]
    date_col = pd.to_datetime(data[date_col_name], errors="coerce")

    data = data[[code]]

    data = data.set_index(date_col).sort_index()
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    # Rename to standard names
    data.columns = [target]

    return data
