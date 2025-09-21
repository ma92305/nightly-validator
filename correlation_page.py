# correlation_page.py

import streamlit as st
import pandas as pd
from datetime import datetime
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

def correlation_page(dbx):
    st.header("Activity vs Heart Rate Correlations")

    # --- Load Excel data from Dropbox ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data provided. Please load Excel data first.")
        return

    # --- Define simplified variable structure ---
    variable_A_map = {
        "Stairs": ("Stairs", ["Time", "Quantity"]),
        "Standing": ("Standing", ["Start_time", "Duration"]),
        "Walking": ("Walking", ["Start_time", "Steps", "Steps/min", "Item"]),
    }

    variable_B_map = {
        "Tachycardia": (["Tachy Events", "HR Stats"], ["event_start", "duration_seconds", "max_bpm", "tachy_percent"]),
        "Daily HR Stats": (["HR Stats"], ["date", "HR_max", "HR_avg", "HR_min", "HRV"]),
    }

    # --- Variable A/B dropdowns ---
    st.subheader("Select Variables to Correlate")
    var_A_col = st.selectbox("Variable A", list(variable_A_map.keys()))
    var_B_col = st.selectbox("Variable B", list(variable_B_map.keys()))

    # --- Load DataFrames based on selections ---
    sheet_A_name, sheet_A_cols = variable_A_map[var_A_col]
    df_A = sheets.get(sheet_A_name, pd.DataFrame())
    if df_A.empty:
        st.warning(f"Activity sheet '{sheet_A_name}' is empty.")
        return

    sheet_B_names, sheet_B_cols = variable_B_map[var_B_col]
    # If multiple sheets (Tachycardia), concatenate
    df_B_list = [sheets.get(s, pd.DataFrame()) for s in sheet_B_names]
    df_B = pd.concat(df_B_list, ignore_index=True)
    if df_B.empty:
        st.warning(f"Heart Rate sheets '{sheet_B_names}' are empty.")
        return

    # --- Convert date/time columns ---
    for df in [df_A, df_B]:
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower() or "start" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Coerce numeric columns ---
    numeric_cols_A = ["Quantity", "Duration", "Steps", "Steps/min"]
    numeric_cols_B = ["tachy_percent", "HR_max", "HR_avg", "HR_min", "HRV", "duration_seconds", "max_bpm"]
    for col in numeric_cols_A:
        if col in df_A.columns:
            df_A[col] = pd.to_numeric(df_A[col], errors="coerce")
    for col in numeric_cols_B:
        if col in df_B.columns:
            df_B[col] = pd.to_numeric(df_B[col], errors="coerce")

    # --- Handle Walking duration if needed ---
    if var_A_col == "Walking" and "Start_time" in df_A.columns and "End_time" in df_A.columns:
        df_A["Duration"] = (df_A["End_time"] - df_A["Start_time"]).dt.total_seconds() / 60.0

    # --- Determine time columns for alignment ---
    time_A = next((c for c in df_A.columns if "time" in c.lower() or "date" in c.lower() or "start" in c.lower()), None)
    time_B = next((c for c in df_B.columns if "time" in c.lower() or "date" in c.lower() or "start" in c.lower()), None)
    if not time_A or not time_B:
        st.error("Cannot detect date/time columns for alignment.")
        return

    # --- Select column to correlate ---
    if var_A_col == "Stairs":
        col_A = "Quantity"
    elif var_A_col == "Standing":
        col_A = "Duration"
    else:  # Walking
        col_A = "Steps"

    if var_B_col == "Tachycardia":
        # Use tachy_percent from HR Stats for continuous correlation
        col_B = "tachy_percent"
    else:  # Daily HR Stats
        col_B = "HR_avg"

    # --- Run correlation ---
    if st.button("Run Correlation Scan"):
        series_A = df_A.set_index(time_A)[col_A]
        series_B = df_B.set_index(time_B)[col_B]

        res_df, sig_df = find_correlations(
            series_A,
            series_B,
            lags_hours=[0, 1, 2, 6, 12, 24, 48],
            match_window_hours=4.0,
            min_pairs=3,
            permutation_n=500,
            bootstrap_n=500,
            effect_size_thresh=0.2,
            alpha=0.05,
            random_state=42,
        )

        st.subheader("Correlation Results")
        st.dataframe(res_df)

        st.subheader("Significant Correlations")
        if not sig_df.empty:
            st.dataframe(sig_df)
        else:
            st.info("No significant correlations found.")
