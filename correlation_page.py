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

# --- Aggregate stairs by day ---
def aggregate_daily_stairs(df, time_col, quantity_col):
    df = normalize_times(df, time_col)
    daily_total = df[quantity_col].resample("D").sum()
    return daily_total

# --- Aggregate tachy % by day ---
def aggregate_daily_tachy_percent(df):
    df = normalize_times(df, "date")
    daily_percent = df["tachy_percent"].resample("D").mean()
    return daily_percent

# --- Event-level tachy metrics per stairs session ---
def compute_event_level_tachy_metrics(stairs_df, tachy_df, window_hours=4):
    stairs_df = stairs_df.copy()
    tachy_df = normalize_times(tachy_df, "event_start")
    event_metrics = pd.DataFrame(index=stairs_df.index)
    
    # Rolling 4hr window after each stairs session
    occurrences = []
    max_bpms = []
    durations = []
    
    for t in stairs_df.index:
        window_start = t
        window_end = t + pd.Timedelta(hours=window_hours)
        events = tachy_df[(tachy_df.index >= window_start) & (tachy_df.index <= window_end)]
        
        occurrences.append(1 if not events.empty else 0)
        max_bpms.append(events["max_bpm"].max() if not events.empty else None)
        durations.append(events["duration_seconds"].max() if not events.empty else None)
    
    event_metrics["Tachy Occurrence"] = pd.Series(occurrences, index=stairs_df.index)
    event_metrics["Tachy Max BPM"] = pd.Series(max_bpms, index=stairs_df.index)
    event_metrics["Tachy Duration"] = pd.Series(durations, index=stairs_df.index)
    
    return event_metrics

def correlation_page(dbx):
    st.header("Stairs ↔ Tachycardia Correlations (POTS Analysis)")

    # --- Load Excel data ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data loaded. Please upload Excel data first.")
        return

    # --- Activity selection ---
    activity_options = ["Stairs", "Standing", "Walking"]
    activity_sel = st.selectbox("Select Activity", activity_options)

    # --- Heart Rate metric selection ---
    hr_options = ["Daily Tachy %", "Event-level Tachy Metrics"]
    hr_sel = st.selectbox("Select Heart Rate Metric", hr_options)

    # --- Load stairs data ---
    if activity_sel != "Stairs":
        st.warning("Currently only 'Stairs' activity is supported for correlation.")
        return
    
    df_stairs = sheets.get("Stairs", pd.DataFrame())
    if df_stairs.empty:
        st.warning("Stairs sheet is empty.")
        return
    df_stairs = normalize_times(df_stairs, "Time")
    stairs_series = df_stairs["Quantity"]

    # --- Load tachy data ---
    df_hr = sheets.get("HR Stats", pd.DataFrame())
    df_tachy = sheets.get("Tachy Events", pd.DataFrame())
    if df_hr.empty or df_tachy.empty:
        st.warning("HR Stats or Tachy Events sheet is empty.")
        return

    results = []
    sig_results = []

    if st.button("Run Correlation Scan"):
        if hr_sel == "Daily Tachy %":
            # --- Aggregate daily-level metrics ---
            daily_stairs = aggregate_daily_stairs(df_stairs, "Time", "Quantity")
            daily_tachy = aggregate_daily_tachy_percent(df_hr)

            # Align days
            combined = pd.concat([daily_stairs, daily_tachy], axis=1).dropna()
            series_A = combined.iloc[:, 0]
            series_B = combined.iloc[:, 1]

            res_df, sig_df = find_correlations(
                series_A,
                series_B,
                lags_hours=[0, 1, 2, 6, 12, 24, 48],
                match_window_hours=24.0,
                min_pairs=3,
                permutation_n=500,
                bootstrap_n=500,
                effect_size_thresh=0.2,
                alpha=0.05,
                random_state=42,
            )
            res_df["Var_A"] = "Total Stairs per Day"
            res_df["Var_B"] = "Daily Tachy %"
            sig_df["Var_A"] = "Total Stairs per Day"
            sig_df["Var_B"] = "Daily Tachy %"
            results.append(res_df)
            if not sig_df.empty:
                sig_results.append(sig_df)

        elif hr_sel == "Event-level Tachy Metrics":
            # --- Compute event-level tachy metrics ---
            event_metrics = compute_event_level_tachy_metrics(df_stairs, df_tachy, window_hours=4)

            for metric in event_metrics.columns:
                series_B = event_metrics[metric]
                series_A = stairs_series.loc[series_B.index]

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
                res_df["Var_A"] = "Stairs (event-level)"
                res_df["Var_B"] = f"Tachy - {metric}"
                sig_df["Var_A"] = "Stairs (event-level)"
                sig_df["Var_B"] = f"Tachy - {metric}"
                results.append(res_df)
                if not sig_df.empty:
                    sig_results.append(sig_df)

        # --- Combine results ---
        full_res = pd.concat(results, ignore_index=True)
        full_sig = pd.concat(sig_results, ignore_index=True) if sig_results else pd.DataFrame()

        # --- Display results ---
        st.subheader("Correlation Results")
        st.dataframe(full_res)

        st.subheader("Significant Correlations")
        if not full_sig.empty:
            st.dataframe(full_sig)
        else:
            st.info("No significant correlations found.")

