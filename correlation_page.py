# correlation_page.py

import streamlit as st
import pandas as pd
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox
from datetime import timedelta

# --- Helper to normalize time columns ---
def normalize_times(df, time_col):
    """
    Convert a column to pandas datetime, drop NaNs, sort, and set as index.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)
    return df

# --- Helper for stair-triggered tachy metrics ---
def stair_triggered_tachy(series_A, df_tachy, lag_hours_list=[0,1,2,6,12,24,48]):
    """
    For each stair event, calculate tachy metrics within lag windows.
    Returns a dict of pandas Series for correlation.
    """
    results = {}

    # Initialize empty series per metric
    occurrence_series = pd.Series(0, index=series_A.index, dtype=int)
    max_bpm_series = pd.Series(index=series_A.index, dtype=float)
    duration_series = pd.Series(index=series_A.index, dtype=float)

    # Loop over stair events
    for idx, stairs_time in enumerate(series_A.index):
        for lag in lag_hours_list:
            window_end = stairs_time + timedelta(hours=lag)
            events = df_tachy[(df_tachy.index >= stairs_time) & (df_tachy.index <= window_end)]

            # Tachy occurrence
            occurrence_series.iloc[idx] = int(len(events) > 0)

            # Max BPM
            max_bpm_series.iloc[idx] = events["max_bpm"].max() if not events.empty else None

            # Duration
            duration_series.iloc[idx] = events["duration_seconds"].max() if not events.empty else None

    results["Tachy Event Occurrence"] = occurrence_series
    results["Tachy Event Max BPM"] = max_bpm_series
    results["Tachy Event Duration (s)"] = duration_series

    return results

def correlation_page(dbx):
    st.header("Activity ↔ Heart Rate Correlations")

    # --- Load Excel data ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data loaded. Please upload Excel data first.")
        return

    # --- Variable A / Activity options ---
    var_A_options = ["Stairs", "Standing", "Walking"]
    var_A_col = st.selectbox("Select Activity", var_A_options)

    # --- Variable B / Heart Rate options ---
    var_B_options = ["Tachycardia", "Daily HR Stats"]
    var_B_col = st.selectbox("Select Heart Rate Metric", var_B_options)

    # --- Extract Activity Data ---
    if var_A_col == "Stairs":
        df_A = sheets.get("Stairs", pd.DataFrame())
        if df_A.empty:
            st.warning("Stairs sheet is empty.")
            return
        df_A = normalize_times(df_A, "Time")
        series_A = df_A["Quantity"]

    elif var_A_col == "Standing":
        df_A = sheets.get("Standing", pd.DataFrame())
        if df_A.empty:
            st.warning("Standing sheet is empty.")
            return
        df_A = normalize_times(df_A, "Start_time")
        series_A = df_A["Duration"]

    elif var_A_col == "Walking":
        df_A = sheets.get("Walking", pd.DataFrame())
        if df_A.empty:
            st.warning("Walking sheet is empty.")
            return
        df_A = normalize_times(df_A, "Start_time")
        series_A = df_A["Steps"]

    # --- Extract Heart Rate Data ---
    if var_B_col == "Daily HR Stats":
        df_B = sheets.get("HR Stats", pd.DataFrame())
        if df_B.empty:
            st.warning("HR Stats sheet is empty.")
            return
        df_B = normalize_times(df_B, "date")
        series_B = df_B["HR_avg"]

        hr_metrics_dict = {"HR Avg": series_B}

    elif var_B_col == "Tachycardia":
        df_hr = sheets.get("HR Stats", pd.DataFrame())
        df_tachy = sheets.get("Tachy Events", pd.DataFrame())
        if df_hr.empty or df_tachy.empty:
            st.warning("HR Stats or Tachy Events sheet is empty.")
            return

        df_hr = normalize_times(df_hr, "date")
        df_tachy = normalize_times(df_tachy, "event_start")

        # Overall tachy percent series (optional)
        tachy_percent_series = df_hr["tachy_percent"]

        # Calculate stair-triggered tachy metrics
        stair_metrics = stair_triggered_tachy(series_A, df_tachy)

        # Include tachy_percent as well if desired
        stair_metrics["Tachy % of Day"] = pd.Series(tachy_percent_series.values, index=series_A.index)

        hr_metrics_dict = stair_metrics

    # --- Run correlations ---
    results = []
    sig_results = []

    if st.button("Run Correlation Scan"):
        for metric_name, series_B in hr_metrics_dict.items():
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

            res_df["Var_A"] = f"Activity - {var_A_col}"
            res_df["Var_B"] = f"Heart Rate - {metric_name}"
            sig_df["Var_A"] = f"Activity - {var_A_col}"
            sig_df["Var_B"] = f"Heart Rate - {metric_name}"

            results.append(res_df)
            if not sig_df.empty:
                sig_results.append(sig_df)

        full_res = pd.concat(results, ignore_index=True)
        full_sig = pd.concat(sig_results, ignore_index=True) if sig_results else pd.DataFrame()

        st.subheader("Correlation Results")
        st.dataframe(full_res)

        st.subheader("Significant Correlations")
        if not full_sig.empty:
            st.dataframe(full_sig)
        else:
            st.info("No significant correlations found.")
