# correlation_page.py

import streamlit as st
import pandas as pd
from datetime import datetime
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

# --- Helper function to normalize times ---
def normalize_times(df, time_col, round_to="min"):
    """
    Convert a time column to a DatetimeIndex and round to the desired resolution.
    Returns a pd.DataFrame indexed by normalized datetime.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")
    
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    
    if round_to == "min":
        df[time_col] = df[time_col].dt.floor("T")
    elif round_to == "s":
        df[time_col] = df[time_col].dt.floor("S")
    elif round_to == "H":
        df[time_col] = df[time_col].dt.floor("H")
    
    return df.set_index(time_col)

def correlation_page(dbx):
    st.header("Activity ↔ Heart Rate Correlations")

    # --- Load Excel data from Dropbox ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data provided. Please load Excel data first.")
        return

    # --- Define simplified variable structure ---
    variable_map_A = {
        "Stairs": ("Stairs", "Quantity"),
        "Standing": ("Standing", "Duration"),
        "Walking": ("Walking", "Steps"),  # or "Steps/min" if preferred
    }

    variable_map_B = {
        "Tachycardia": ("Tachy Events", ["event_start", "duration_seconds", "max_bpm"], "HR Stats", "tachy_percent"),
        "Daily HR Stats": ("HR Stats", ["HR_max", "HR_avg", "HR_min", "HRV"], None, None),
    }

    # --- Variable A dropdowns ---
    st.subheader("Select Activity Variable")
    var_A_col = st.selectbox("Activity Type", list(variable_map_A.keys()))

    # --- Variable B dropdowns ---
    st.subheader("Select Heart Rate Variable")
    var_B_col = st.selectbox("Heart Rate Type", list(variable_map_B.keys()))

    # --- Load selected sheets ---
    sheet_A_name, col_A = variable_map_A[var_A_col]
    df_A = sheets.get(sheet_A_name, pd.DataFrame())
    if df_A.empty:
        st.warning(f"Activity sheet '{sheet_A_name}' is empty.")
        return

    # --- Variable B setup ---
    if var_B_col == "Tachycardia":
        tachy_sheet, tachy_cols, hr_sheet, hr_col = variable_map_B[var_B_col]
        df_tachy = sheets.get(tachy_sheet, pd.DataFrame())
        df_hr = sheets.get(hr_sheet, pd.DataFrame())
        if df_tachy.empty and df_hr.empty:
            st.warning("No Tachycardia or HR percent data found.")
            return
        # For now just combine tachy events into single series (max_bpm per event_start)
        df_tachy["event_start"] = pd.to_datetime(df_tachy["event_start"], errors="coerce")
        df_tachy = df_tachy.dropna(subset=["event_start"])
        series_B = df_tachy.set_index("event_start")["max_bpm"]
        # append tachy_percent from HR Stats as another series if needed
        df_hr["Date"] = pd.to_datetime(df_hr["Date"], errors="coerce")
        df_hr = df_hr.dropna(subset=["Date"])
        series_hr_percent = df_hr.set_index("Date")["tachy_percent"]
        # Combine by reindexing both onto a common time axis later in find_correlations
    else:
        hr_sheet, hr_cols, _, _ = variable_map_B[var_B_col]
        df_hr = sheets.get(hr_sheet, pd.DataFrame())
        if df_hr.empty:
            st.warning(f"Heart Rate sheet '{hr_sheet}' is empty.")
            return
        df_hr["Date"] = pd.to_datetime(df_hr["Date"], errors="coerce")
        df_hr = df_hr.dropna(subset=["Date"])
        series_B = df_hr.set_index("Date")[hr_cols[0]]  # default to HR_max for simplicity

    # --- Normalize times ---
    time_col_A = "Time" if var_A_col == "Stairs" else "Start_time"
    df_A = normalize_times(df_A, time_col_A)
    series_A = df_A[col_A]

    series_B = series_B.sort_index()
    series_A = series_A.sort_index()

    # --- Run correlation ---
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

        st.subheader("Correlation Results")
        st.dataframe(res_df)

        st.subheader("Significant Correlations")
        if not sig_df.empty:
            st.dataframe(sig_df)
        else:
            st.info("No significant correlations found.")
