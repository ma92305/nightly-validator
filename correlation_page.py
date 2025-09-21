# correlation_page.py
import streamlit as st
import pandas as pd
import numpy as np
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

def correlation_page(dbx):
    st.title("📊 Time-lagged Correlation Explorer")

    # -------------------------
    # Load Excel from Dropbox
    # -------------------------
    st.info("Loading Excel data from Dropbox...")
    sheets_dict = load_excel_from_dropbox(dbx)
    if not sheets_dict:
        st.error("Failed to load sheets.")
        return

    # -------------------------
    # Variable mapping
    # -------------------------
    def convert_to_numeric(series, col_name=None):
        """Convert non-numeric entries (emojis, text) to numbers"""
        if series.dtype.kind in "biufc":
            return series
        mapping = {
            "⚪️": 0, "🟡": 1, "🟠": 2, "🔴": 3,
            "A little": 1, "Some": 2, "Lots": 3,
            "taken": 1, "skipped": 0
        }
        return series.map(mapping).fillna(0)

    # Build dropdown options for all numeric-like columns across sheets
    VAR_MAPPING = {}
    for sheet_name, df in sheets_dict.items():
        time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        value_cols = [c for c in df.columns if c not in time_cols and c not in ["file", "emoji", "status"]]
        if not time_cols or not value_cols:
            continue
        for val_col in value_cols:
            key = f"{sheet_name} - {val_col}"
            VAR_MAPPING[key] = (sheet_name, val_col, time_cols[0])

    # -------------------------
    # Variable selectors
    # -------------------------
    varA_key = st.selectbox("Select Variable A (cause)", options=list(VAR_MAPPING.keys()))
    varB_key = st.selectbox("Select Variable B (effect)", options=list(VAR_MAPPING.keys()))
    sheetA, valA_col, timeA_col = VAR_MAPPING[varA_key]
    sheetB, valB_col, timeB_col = VAR_MAPPING[varB_key]

    dfA = sheets_dict[sheetA]
    dfB = sheets_dict[sheetB]
    seriesA = convert_to_numeric(dfA[valA_col], valA_col)
    seriesB = convert_to_numeric(dfB[valB_col], valB_col)

    # -------------------------
    # Correlation parameters
    # -------------------------
    st.sidebar.header("Correlation Parameters")
    lags_hours = st.sidebar.text_input("Lags (hours, comma-separated)", "0,1,2,6,12,24,48")
    try:
        lags_hours = [float(x.strip()) for x in lags_hours.split(",")]
    except Exception:
        st.error("Invalid lag input! Must be comma-separated numbers.")
        return

    match_window_hours = st.sidebar.number_input("Match window (hours)", value=4.0, min_value=0.1)
    min_pairs = st.sidebar.number_input("Minimum paired observations", value=10, min_value=2)
    effect_size_thresh = st.sidebar.slider("Minimum |r| for effect size", 0.0, 1.0, 0.25)
    alpha = st.sidebar.slider("Significance alpha", 0.001, 0.2, 0.05)

    # -------------------------
    # Run correlation
    # -------------------------
    if st.button("Run Correlation"):
        with st.spinner("Computing correlations..."):
            results_df, significant_df = find_correlations(
                seriesA, seriesB,
                lags_hours=lags_hours,
                match_window_hours=match_window_hours,
                min_pairs=min_pairs,
                effect_size_thresh=effect_size_thresh,
                alpha=alpha
            )

        st.subheader("Top Results (sorted by |Pearson r|)")
        st.dataframe(results_df.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False))

        st.subheader("Significant Correlations")
        if significant_df.empty:
            st.info("No significant correlations found under current parameters.")
        else:
            st.dataframe(significant_df)
