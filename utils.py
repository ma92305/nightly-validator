# utils.py
import pandas as pd
import numpy as np

# Map severity emojis to numeric scale (customize as you want)
SEVERITY_MAP = {
    "⚪️": 0,  # none/very low
    "🟡": 1,
    "🟠": 2,
    "🔴": 3,
    "🟣": 4,  # example if you use other emojis for meds/notes
}

def to_datetime_if_possible(df, cols=None):
    """
    Try to convert columns with 'time' or 'date' in name to datetime.
    If cols provided, try those explicitly.
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if ("time" in c.lower() or "date" in c.lower())]
    for c in cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            pass
    return df

def severity_to_numeric(series):
    """Map emoji severities to numeric; unknown -> NaN."""
    return series.map(SEVERITY_MAP).astype('float')

def safe_daily_date_from_col(df, col_candidates=None):
    """
    Return a series of dates (datetime.date) choosing the best available column.
    If a 'date' column exists, use it; else use first datetime-like in columns.
    """
    if col_candidates is None:
        col_candidates = df.columns.tolist()
    # try exact 'date' column
    for c in col_candidates:
        if c.lower() == 'date':
            return pd.to_datetime(df[c], errors='coerce').dt.date
    # else find datetime columns
    for c in col_candidates:
        if 'time' in c.lower() or 'date' in c.lower():
            dt = pd.to_datetime(df[c], errors='coerce')
            if dt.notna().any():
                return dt.dt.date
    # fallback: index or NaN
    return pd.Series([pd.NaT] * len(df))

def maybe_numeric(series):
    """Try convert to numeric, else return NaN for non-numeric entries."""
    return pd.to_numeric(series, errors='coerce')

def one_hot_count(df, col, prefix=None):
    """Return series with counts per unique value in col (useful for nutrition item counts)."""
    if prefix is None:
        prefix = col
    dummies = pd.get_dummies(df[col].fillna(""), prefix=prefix)
    return dummies.sum(axis=0)
