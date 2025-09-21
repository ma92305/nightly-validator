# correlation_page.py

import streamlit as st
import pandas as pd
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

# --- Helper to normalize time columns ---
def normalize_times(df, time_col):
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
        series_A = df_A["Quantity"]  # flights per event
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
    df_hr = sheets.get("HR Stats", pd.DataFrame())
    df_tachy = sheets.get("Tachy Events", pd.DataFrame())

    if df_hr.empty:
        st.warning("HR Stats sheet is empty.")
        return

    df_hr = normalize_times(df_hr, "date")

    if var_B_col == "Daily HR Stats":
        # --- Event-level correlations not relevant here ---
        st.info("Daily HR Stats selected. Use aggregated stair data per day for correlations.")
        series_A_daily = series_A.groupby(series_A.index.date).sum()
        series_B_daily = df_hr["tachy_percent"]
        # Align by day
        df_daily = pd.DataFrame({
            "Stairs_Flights": series_A_daily,
            "Tachy_Percent": series_B_daily
        }).dropna()

        if st.button("Run Daily Totals Correlation"):
            res_df, sig_df = find_correlations(
                df_daily["Stairs_Flights"],
                df_daily["Tachy_Percent"],
                lags_hours=[0],  # daily totals, lag not meaningful
                min_pairs=3,
                permutation_n=500,
                bootstrap_n=500,
                effect_size_thresh=0.2,
                alpha=0.05,
                random_state=42,
            )
            res_df["Var_A"] = f"Activity - {var_A_col} (Daily Total)"
            res_df["Var_B"] = "Heart Rate - Tachy % of Day"
            sig_df["Var_A"] = f"Activity - {var_A_col} (Daily Total)"
            sig_df["Var_B"] = "Heart Rate - Tachy % of Day"

            st.subheader("Daily Totals Correlation Results")
            st.dataframe(res_df)
            st.subheader("Significant Correlations")
            if not sig_df.empty:
                st.dataframe(sig_df)
            else:
                st.info("No significant daily correlations found.")

    elif var_B_col == "Tachycardia":
        if df_tachy.empty:
            st.warning("Tachy Events sheet is empty.")
            return
        df_tachy = normalize_times(df_tachy, "event_start")

        # --- Event-level series ---
        binary_events = pd.Series(0, index=series_A.index)
        max_bpm_series = pd.Series(index=series_A.index, dtype=float)
        duration_series = pd.Series(index=series_A.index, dtype=float)

        for idx, act_time in enumerate(series_A.index):
            window_start = act_time
            window_end = act_time + pd.Timedelta(hours=4)
            events_in_window = df_tachy[(df_tachy.index >= window_start) & (df_tachy.index <= window_end)]
            binary_events.iloc[idx] = 1 if not events_in_window.empty else 0
            max_bpm_series.iloc[idx] = events_in_window["max_bpm"].max() if not events_in_window.empty else None
            duration_series.iloc[idx] = events_in_window["duration_seconds"].max() if not events_in_window.empty else None

        tachy_series_dict = {
            "Tachy Event Occurrence": binary_events,
            "Tachy Event Max BPM": max_bpm_series,
            "Tachy Event Duration (s)": duration_series
        }

        results = []
        sig_results = []

        if st.button("Run Event-Level Correlation Scan"):
            for metric_name, series_B in tachy_series_dict.items():
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

            st.subheader("Event-Level Correlation Results")
            st.dataframe(full_res)
            st.subheader("Significant Correlations")
            if not full_sig.empty:
                st.dataframe(full_sig)
            else:
                st.info("No significant event-level correlations found.")
