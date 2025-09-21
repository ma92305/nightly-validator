# correlation_page.py

import streamlit as st
import pandas as pd
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

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
        # For simplicity, use Steps or Steps/min
        series_A = df_A["Steps"]

    # --- Extract Heart Rate Data ---
    if var_B_col == "Daily HR Stats":
        df_B = sheets.get("HR Stats", pd.DataFrame())
        if df_B.empty:
            st.warning("HR Stats sheet is empty.")
            return
        df_B = normalize_times(df_B, "date")  # lowercase date
        # Select one column; you can extend to allow column selection
        series_B = df_B["HR_avg"]

    elif var_B_col == "Tachycardia":
        df_hr = sheets.get("HR Stats", pd.DataFrame())
        df_tachy = sheets.get("Tachy Events", pd.DataFrame())
        if df_hr.empty or df_tachy.empty:
            st.warning("HR Stats or Tachy Events sheet is empty.")
            return
        df_hr = normalize_times(df_hr, "date")
        df_tachy = normalize_times(df_tachy, "event_start")

        # Combine tachy_percent from HR Stats with event info
        series_hr_percent = df_hr["tachy_percent"]
        # Here, for simplicity, we can just use max_bpm from events
        series_events = df_tachy["max_bpm"]
        # Concatenate and sort by index
        series_B = pd.concat([series_hr_percent, series_events]).sort_index()

    # --- Run Correlation ---
    if st.button("Run Correlation Scan"):
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
    
        # Label variables clearly
        var_a_name = f"Activity - {var_A_col}"
        var_b_name = f"Heart Rate - {var_B_col}"
    
        res_df["Var_A"] = var_a_name
        res_df["Var_B"] = var_b_name
        sig_df["Var_A"] = var_a_name
        sig_df["Var_B"] = var_b_name
    
        # Reorder so labels come first
        cols = ["Var_A", "Var_B"] + [c for c in res_df.columns if c not in ["Var_A", "Var_B"]]
        res_df = res_df[cols]
        sig_df = sig_df[cols]
    
        st.subheader("Correlation Results")
        st.dataframe(res_df)
    
        st.subheader("Significant Correlations")
        if not sig_df.empty:
            st.dataframe(sig_df)
        else:
            st.info("No significant correlations found.")
