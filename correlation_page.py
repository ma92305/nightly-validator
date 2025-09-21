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

# --- Helper to find the correct time column ---
def get_time_col(df):
    for col in ["Time", "Start_time", "DateTime", "date"]:
        if col in df.columns:
            return col
    raise ValueError(f"No recognized time column in dataframe. Columns: {df.columns.tolist()}")

# --- Aggregate daily stairs ---
def aggregate_daily_stairs(df, quantity_col):
    time_col = get_time_col(df)
    df = normalize_times(df, time_col)
    daily_total = df[quantity_col].resample("D").sum()
    return daily_total

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
        time_col = get_time_col(df_A)
        df_A = normalize_times(df_A, time_col)
        event_series = df_A["Quantity"]  # per-event number of stairs
        daily_series = aggregate_daily_stairs(df_A, "Quantity")  # per-day total stairs

    elif var_A_col == "Standing":
        df_A = sheets.get("Standing", pd.DataFrame())
        if df_A.empty:
            st.warning("Standing sheet is empty.")
            return
        time_col = get_time_col(df_A)
        df_A = normalize_times(df_A, time_col)
        event_series = df_A["Duration"]
        daily_series = df_A["Duration"].resample("D").sum()

    elif var_A_col == "Walking":
        df_A = sheets.get("Walking", pd.DataFrame())
        if df_A.empty:
            st.warning("Walking sheet is empty.")
            return
        time_col = get_time_col(df_A)
        df_A = normalize_times(df_A, time_col)
        event_series = df_A["Steps"]
        daily_series = df_A["Steps"].resample("D").sum()

    # --- Extract Heart Rate Data ---
    df_hr = sheets.get("HR Stats", pd.DataFrame())
    df_tachy = sheets.get("Tachy Events", pd.DataFrame())
    if df_hr.empty or df_tachy.empty:
        st.warning("HR Stats or Tachy Events sheet is empty.")
        return
    df_hr = normalize_times(df_hr, get_time_col(df_hr))
    df_tachy = normalize_times(df_tachy, get_time_col(df_tachy))

    # --- Prepare Tachycardia series ---
    tachy_series_dict = {}

    # 1️⃣ Daily total tachy %
    daily_tachy_percent = df_hr["tachy_percent"].resample("D").mean()
    tachy_series_dict["Daily Tachy %"] = daily_tachy_percent

    # 2️⃣ Event-level tachy occurrence, max BPM, duration (aligned to stairs events)
    binary_occurrence = pd.Series(0, index=event_series.index)
    max_bpm_series = pd.Series(index=event_series.index, dtype=float)
    duration_series = pd.Series(index=event_series.index, dtype=float)

    for idx, event_time in enumerate(event_series.index):
        window_start = event_time
        window_end = event_time + pd.Timedelta(hours=4)
        events_in_window = df_tachy[(df_tachy.index >= window_start) & (df_tachy.index <= window_end)]
        if not events_in_window.empty:
            binary_occurrence.iloc[idx] = 1
            max_bpm_series.iloc[idx] = events_in_window["max_bpm"].max()
            duration_series.iloc[idx] = events_in_window["duration_seconds"].max()
        else:
            max_bpm_series.iloc[idx] = None
            duration_series.iloc[idx] = None

    tachy_series_dict["Tachy Event Occurrence"] = binary_occurrence
    tachy_series_dict["Tachy Event Max BPM"] = max_bpm_series
    tachy_series_dict["Tachy Event Duration (s)"] = duration_series

    # --- Run correlations ---
    if st.button("Run Correlation Scan"):
        all_results = []
        all_sig = []

        # Event-level correlations
        for metric_name, series_B in tachy_series_dict.items():
            res_df, sig_df = find_correlations(
                event_series,
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
            res_df["Var_A"] = f"{var_A_col} (event-level)"
            res_df["Var_B"] = f"Heart Rate - {metric_name}"
            sig_df["Var_A"] = f"{var_A_col} (event-level)"
            sig_df["Var_B"] = f"Heart Rate - {metric_name}"
            all_results.append(res_df)
            if not sig_df.empty:
                all_sig.append(sig_df)

        # Daily-level correlation: total stairs per day ↔ daily tachy %
        if "Daily Tachy %" in tachy_series_dict and var_A_col == "Stairs":
            daily_series_aligned, daily_tachy_aligned = daily_series.align(daily_tachy_percent, join="inner")
            res_df, sig_df = find_correlations(
                daily_series_aligned,
                daily_tachy_aligned,
                lags_hours=[0, 1, 2, 6, 12, 24],
                match_window_hours=24,
                min_pairs=3,
                permutation_n=500,
                bootstrap_n=500,
                effect_size_thresh=0.2,
                alpha=0.05,
                random_state=42,
            )
            res_df["Var_A"] = f"{var_A_col} (daily total)"
            res_df["Var_B"] = "Heart Rate - Daily Tachy %"
            sig_df["Var_A"] = f"{var_A_col} (daily total)"
            sig_df["Var_B"] = "Heart Rate - Daily Tachy %"
            all_results.append(res_df)
            if not sig_df.empty:
                all_sig.append(sig_df)

        # Combine results
        full_res = pd.concat(all_results, ignore_index=True)
        full_sig = pd.concat(all_sig, ignore_index=True) if all_sig else pd.DataFrame()

        st.subheader("Correlation Results")
        st.dataframe(full_res)

        st.subheader("Significant Correlations")
        if not full_sig.empty:
            st.dataframe(full_sig)
        else:
            st.info("No significant correlations found.")
