# correlation_page.py

import streamlit as st
import pandas as pd
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

def correlation_page(dbx):
    st.header("Time-Lagged Correlations")

    # --- Load Excel data from Dropbox ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data provided. Please load Excel data first.")
        return

    # Example: using Heart Rate and Sleep Stats
    hr_df = sheets.get("HR Stats", pd.DataFrame())
    sleep_df = sheets.get("Sleep Stats", pd.DataFrame())

    if hr_df.empty or sleep_df.empty:
        st.warning("Heart Rate or Sleep Stats sheet is empty.")
        return

    # Convert date/time columns
    for col in ["date", "time", "bedtime", "waketime"]:
        if col in hr_df.columns:
            hr_df[col] = pd.to_datetime(hr_df[col], errors="coerce")
        if col in sleep_df.columns:
            sleep_df[col] = pd.to_datetime(sleep_df[col], errors="coerce")

    # Select variables for correlation
    st.subheader("Select variables to correlate")
    hr_options = [c for c in hr_df.columns if hr_df[c].dtype in ["int64", "float64"]]
    sleep_options = [c for c in sleep_df.columns if sleep_df[c].dtype in ["int64", "float64"]]

    var_A = st.selectbox("Variable A (cause)", hr_options)
    var_B = st.selectbox("Variable B (effect)", sleep_options)

    if st.button("Run Correlation Scan"):
        # Align by date/time
        A_series = hr_df.set_index("date")[var_A]
        B_series = sleep_df.set_index("date")[var_B]

        # Run correlation scan
        res_df, sig_df = find_correlations(
            A_series, B_series,
            lags_hours=[0, 1, 2, 6, 12, 24, 48],
            match_window_hours=4.0,
            min_pairs=3,
            permutation_n=500,
            bootstrap_n=500,
            effect_size_thresh=0.2,
            alpha=0.05,
            random_state=42
        )

        st.subheader("Correlation Results")
        st.dataframe(res_df)

        st.subheader("Significant Correlations")
        if not sig_df.empty:
            st.dataframe(sig_df)
        else:
            st.info("No significant correlations found.")

