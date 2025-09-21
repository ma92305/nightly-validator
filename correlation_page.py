# correlation_page.py

import streamlit as st
import pandas as pd
from datetime import datetime
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

def correlation_page(dbx):
    st.header("Time-Lagged Correlations")

    # --- Load Excel data from Dropbox ---
    sheets = load_excel_from_dropbox(dbx)
    if not sheets:
        st.error("No data provided. Please load Excel data first.")
        return

    # --- Define variable structure ---
    variable_map = {
        "Conditions": {"Condition": ("Conditions", "item")},
        "Activity": {
            "Stairs": ("Activity", "Stairs_Quantity"),
            "Walking Long": ("Activity", "Walking_Long"),
            "Walking Brisk": ("Activity", "Walking_Brisk"),
            "Standing": ("Activity", "Standing_Duration"),
        },
        "Tachycardia": {"Tachy %": ("HR Stats", "tachy_percent")},
        "Daily HR Stats": {
            "Max HR": ("HR Stats", "HR_max"),
            "Avg HR": ("HR Stats", "HR_avg"),
            "Min HR": ("HR Stats", "HR_min"),
            "HRV": ("HR Stats", "HRV"),
        },
        "Medications": {"Name": ("Meds", "name"), "Status": ("Meds", "status"), "Dose": ("Meds", "dose")},
        "Nutrition": {
            "Ingredients": ("Nutrition - General", "Item"),
            "Liquids": ("Nutrition - Liquids", "amount"),
            "Meals": ("Nutrition - Meals", "amount"),
        },
        "Weather": {
            "Temp High": ("Weather Stats", "temp_high"),
            "Temp Low": ("Weather Stats", "temp_low"),
            "Temp Avg": ("Weather Stats", "temp_avg"),
            "Humidity Avg": ("Weather Stats", "humidity_avg"),
            "Pressure Avg": ("Weather Stats", "pressure_avg"),
            "Precipitation Hours": ("Weather Stats", "precipitation_hours"),
        },
        "Sleep Stats": {
            "Duration": ("Sleep Stats", "duration"),
            "Score": ("Sleep Stats", "score"),
            "Bedtime": ("Sleep Stats", "bedtime"),
            "Waketime": ("Sleep Stats", "waketime"),
            "REM": ("Sleep Stats", "rem_time"),
            "Core": ("Sleep Stats", "core_time"),
            "Deep": ("Sleep Stats", "deep_time"),
            "Awake": ("Sleep Stats", "awake_time"),
        },
    }

    # --- Variable A/B dropdowns ---
    st.subheader("Select Variables to Correlate")
    var_A_cat = st.selectbox("Variable A Category", list(variable_map.keys()), key="var_A_cat")
    var_B_cat = st.selectbox("Variable B Category", list(variable_map.keys()), key="var_B_cat")

    var_A_col = st.selectbox(f"Variable A Column ({var_A_cat})", list(variable_map[var_A_cat].keys()), key="var_A_col")
    var_B_col = st.selectbox(f"Variable B Column ({var_B_cat})", list(variable_map[var_B_cat].keys()), key="var_B_col")

    # --- Extract sheet & column ---
    sheet_A, col_A = variable_map[var_A_cat][var_A_col]
    sheet_B, col_B = variable_map[var_B_cat][var_B_col]

    df_A = sheets.get(sheet_A, pd.DataFrame())
    df_B = sheets.get(sheet_B, pd.DataFrame())

    if df_A.empty or df_B.empty:
        st.warning(f"Selected sheets {sheet_A} or {sheet_B} are empty.")
        return

    # --- Convert time/date columns ---
    for df in [df_A, df_B]:
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # --- Coerce numeric HR/Tachy columns ---
    numeric_cols = ["HR_max", "HR_avg", "HR_min", "HRV", "tachy_percent", "Stairs_Quantity", "Duration", "Steps", "Steps/min"]
    for col in numeric_cols:
        if col in df_A.columns:
            before_na = df_A[col].isna().sum()
            df_A[col] = pd.to_numeric(df_A[col], errors="coerce")
            after_na = df_A[col].isna().sum()
            if after_na > before_na:
                st.warning(f"Column '{col}' in Variable A had {after_na-before_na} non-numeric values coerced to NaN")
        if col in df_B.columns:
            before_na = df_B[col].isna().sum()
            df_B[col] = pd.to_numeric(df_B[col], errors="coerce")
            after_na = df_B[col].isna().sum()
            if after_na > before_na:
                st.warning(f"Column '{col}' in Variable B had {after_na-before_na} non-numeric values coerced to NaN")

    # --- Determine time columns ---
    time_A = next((c for c in df_A.columns if "time" in c.lower() or "date" in c.lower()), None)
    time_B = next((c for c in df_B.columns if "time" in c.lower() or "date" in c.lower()), None)
    if not time_A or not time_B:
        st.error("Cannot detect date/time columns for alignment.")
        return

    # --- Handle Walking duration ---
    if "Walking" in var_A_col and "Start_time" in df_A.columns and "End_time" in df_A.columns:
        df_A["Duration"] = (df_A["End_time"] - df_A["Start_time"]).dt.total_seconds() / 60.0
        col_A = "Duration"
    if "Walking" in var_B_col and "Start_time" in df_B.columns and "End_time" in df_B.columns:
        df_B["Duration"] = (df_B["End_time"] - df_B["Start_time"]).dt.total_seconds() / 60.0
        col_B = "Duration"

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
