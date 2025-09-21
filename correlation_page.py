# correlation_page.py

import streamlit as st
import dropbox
import pandas as pd
from load_excel import load_excel_from_dropbox
from correlation_engine import find_correlations
import numpy as np
import matplotlib.pyplot as plt

def correlation_page():
    st.header("Time-Lagged Correlations")

    # --- Initialize Dropbox client ---
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=st.secrets["dropbox_refresh_token"],
        app_key=st.secrets["dropbox_app_key"],
        app_secret=st.secrets["dropbox_app_secret"]
    )

    # --- Load Excel data from Dropbox ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data provided. Please load Excel data first.")
        return

    # --- Sheet selection ---
    sheet_names = list(sheets.keys())
    sheet_A_name = st.selectbox("Select Sheet for Variable A (cause)", sheet_names)
    sheet_B_name = st.selectbox("Select Sheet for Variable B (effect)", sheet_names)

    df_A = sheets.get(sheet_A_name, pd.DataFrame())
    df_B = sheets.get(sheet_B_name, pd.DataFrame())

    if df_A.empty or df_B.empty:
        st.warning("Selected sheet(s) are empty.")
        return

    # --- Column selection ---
    numeric_cols_A = df_A.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_B = df_B.select_dtypes(include=[np.number]).columns.tolist()
    time_cols_A = [c for c in df_A.columns if "time" in c.lower() or "date" in c.lower()]
    time_cols_B = [c for c in df_B.columns if "time" in c.lower() or "date" in c.lower()]

    if not numeric_cols_A or not numeric_cols_B or not time_cols_A or not time_cols_B:
        st.warning("Sheets must have at least one numeric column and one datetime column each.")
        return

    value_col_A = st.selectbox("Select Variable A (numeric)", numeric_cols_A)
    value_col_B = st.selectbox("Select Variable B (numeric)", numeric_cols_B)
    time_col_A = st.selectbox("Select Time Column for A", time_cols_A)
    time_col_B = st.selectbox("Select Time Column for B", time_cols_B)

    # --- Lag options ---
    lags_input = st.text_input(
        "Enter lags in hours (comma-separated)", 
        value="0,1,2,6,12,24,48"
    )
    try:
        lags_hours = [float(x.strip()) for x in lags_input.split(",")]
    except Exception:
        st.warning("Invalid lag input. Using default: [0,1,2,6,12,24,48]")
        lags_hours = [0,1,2,6,12,24,48]

    match_window = st.number_input("Matching window (hours)", min_value=0.1, value=4.0, step=0.1)

    run_btn = st.button("Run Correlation")
    if run_btn:
        with st.spinner("Computing correlations..."):
            results_df, significant_df = find_correlations(
                df_A, df_B,
                A_time_col=time_col_A,
                B_time_col=time_col_B,
                A_value_col=value_col_A,
                B_value_col=value_col_B,
                lags_hours=lags_hours,
                match_window_hours=match_window,
                min_pairs=5,
                effect_size_thresh=0.2,
                alpha=0.05,
            )

        st.subheader("All Results")
        st.dataframe(results_df)

        st.subheader("Significant Results")
        st.dataframe(significant_df)

        # --- Simple plot ---
        if not results_df.empty:
            fig, ax = plt.subplots()
            ax.plot(results_df["lag_hours"], results_df["pearson_r"], marker="o")
            ax.set_xlabel("Lag (hours)")
            ax.set_ylabel("Pearson r")
            ax.set_title(f"{value_col_A} -> {value_col_B}")
            st.pyplot(fig)
