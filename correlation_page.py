# correlation_page.py
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy.stats import t as tdist

from load_excel import load_excel_from_dropbox


# ----------------------------------
# Configurable scenarios
# ----------------------------------
CORRELATION_SCENARIOS = {
    "Symptoms & Sleep": {
        "groups": ("Symptoms", "Sleep"),
        "lags": [0, 1, 2, 3, 6, 12, 24],
    },
    "Symptoms & Weather": {
        "groups": ("Symptoms", "Weather"),
        "lags": [0, 6, 12, 24, 48],
    },
    "All Variables": {
        "groups": None,  # means everything
        "lags": [0, 6, 12, 24],
    },
}


# ----------------------------------
# Helpers
# ----------------------------------
def _shift_series(series, lag):
    """Shift a series by a number of hours (lag)."""
    return series.shift(lag)


def _calc_corr(x, y):
    """Pearson correlation with significance test."""
    corr = x.corr(y)
    if pd.isna(corr):
        return np.nan, 1.0
    n = (x.notna() & y.notna()).sum()
    if n < 3:
        return corr, 1.0
    t = corr * np.sqrt((n - 2) / (1 - corr**2))
    p = 2 * (1 - tdist.cdf(abs(t), df=n - 2))
    return corr, p


def _effect_size_label(r):
    if abs(r) < 0.2:
        return "weak"
    elif abs(r) < 0.4:
        return "moderate"
    elif abs(r) < 0.6:
        return "strong"
    else:
        return "very strong"


def _ensure_datetime_index(df):
    """Try to enforce datetime index if a 'date' or 'time' column exists."""
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df = df.copy()
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df = df.set_index(col)
                    df = df.sort_index()
                    return df
                except Exception:
                    continue
    return df


# ----------------------------------
# Main
# ----------------------------------
def correlation_page(dbx):
    st.header("Correlation Explorer")

    # 🔹 Load workbook from Dropbox
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.warning("No data available.")
        return

    # 🔹 Ensure datetime indices where possible
    for name, df in sheets.items():
        sheets[name] = _ensure_datetime_index(df)

    scenario = st.selectbox("Select scenario", list(CORRELATION_SCENARIOS.keys()))
    config = CORRELATION_SCENARIOS[scenario]
    groups = config["groups"]
    lags = config["lags"]

    # Build variable pairs
    if groups:
        left_vars = sheets.get(groups[0], pd.DataFrame()).columns.tolist()
        right_vars = sheets.get(groups[1], pd.DataFrame()).columns.tolist()
        var_pairs = [(lv, rv) for lv in left_vars for rv in right_vars]
    else:
        all_vars = [col for df in sheets.values() for col in df.columns]
        var_pairs = list(combinations(all_vars, 2))

    results = []
    for (var1, var2) in var_pairs:
        s1, s2 = None, None

        # 🔹 Find series in any sheet
        for df in sheets.values():
            if var1 in df.columns and s1 is None:
                s1 = df[var1]
            if var2 in df.columns and s2 is None:
                s2 = df[var2]

        if s1 is None or s2 is None:
            continue

        # 🔹 Align on datetime index if available
        s1 = s1.dropna()
        s2 = s2.dropna()
        if isinstance(s1.index, pd.DatetimeIndex) and isinstance(s2.index, pd.DatetimeIndex):
            combined = pd.concat([s1, s2], axis=1, join="inner").dropna()
            if combined.empty:
                continue
            s1, s2 = combined.iloc[:, 0], combined.iloc[:, 1]

        # 🔹 Calculate correlations with lags
        for lag in lags:
            corr, p = _calc_corr(s1, _shift_series(s2, lag))
            results.append({"var1": var1, "var2": var2, "lag": lag, "r": corr, "p": p})

    results_df = pd.DataFrame(results).dropna()

    if not results_df.empty:
        rej, p_corr, _, _ = multipletests(results_df["p"], method="fdr_bh")
        results_df["p_adj"] = p_corr
        results_df["significant"] = rej
        results_df["strength"] = results_df["r"].apply(_effect_size_label)

    st.subheader("Correlation Table")
    if results_df.empty:
        st.info("No correlations could be calculated.")
        return

    threshold = st.slider("Minimum |r| to show", 0.0, 1.0, 0.3, 0.05)
    filtered = results_df[results_df["r"].abs() >= threshold].copy()
    if filtered.empty:
        st.info("No correlations pass the threshold.")
    else:
        st.dataframe(filtered.sort_values("r", ascending=False))

        st.subheader("Lag Profiles")
        sel = st.selectbox(
            "Select variable pair",
            filtered[["var1", "var2"]].drop_duplicates().apply(lambda r: f"{r.var1} vs {r.var2}", axis=1),
        )
        v1, v2 = sel.split(" vs ")
        subset = results_df[(results_df["var1"] == v1) & (results_df["var2"] == v2)]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(subset["lag"], subset["r"], marker="o")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Correlation (r)")
        ax.set_title(f"Lag Profile: {v1} vs {v2}")
        st.pyplot(fig)

    st.caption("Note: adjusted p-values use Benjamini–Hochberg correction.")
