# correlation_page.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from correlation_engine import find_correlations
from load_excel import load_excel_from_dropbox

st.set_page_config(page_title="Time-lagged Correlations", layout="wide")

st.title("📊 Time-lagged Correlation Explorer")

# -------------------------
# Load data from Dropbox
# -------------------------
st.info("Loading Excel data from Dropbox...")
if "sheets_dict" not in st.session_state:
    # _dbx is your Dropbox client object
    sheets_dict = load_excel_from_dropbox(st.session_state._dbx)
    st.session_state.sheets_dict = sheets_dict
else:
    sheets_dict = st.session_state.sheets_dict

if not sheets_dict:
    st.stop()

# -------------------------
# Variable mapping
# -------------------------
def convert_to_numeric(series, col_name=None):
    """Convert categorical / emoji / string columns to numeric if needed."""
    if series.dtype.kind in "biufc":
        return series
    # map emojis / severity
    mapping = {
        "⚪️": 0, "🟡": 1, "🟠": 2, "🔴": 3,
        "A little": 1, "Some": 2, "Lots": 3,
        "taken": 1, "skipped": 0
    }
    return series.map(mapping).fillna(0)

# Build variable key mapping: key -> (sheet_name, value_col, time_col)
VAR_MAPPING = {}

for sheet_name, df in sheets_dict.items():
    # Auto-detect columns
    cols_lower = [c.lower() for c in df.columns]
    time_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    value_cols = [c for c in df.columns if c not in time_cols and c not in ["file", "emoji", "status"]]

    if not time_cols or not value_cols:
        continue

    for val_col in value_cols:
        key = f"{sheet_name} - {val_col}"
        VAR_MAPPING[key] = (sheet_name, val_col, time_cols[0])

# -------------------------
# Dropdown selectors
# -------------------------
varA_key = st.selectbox("Select Variable A (cause)", options=list(VAR_MAPPING.keys()))
varB_key = st.selectbox("Select Variable B (effect)", options=list(VAR_MAPPING.keys()))

sheetA, valA_col, timeA_col = VAR_MAPPING[varA_key]
sheetB, valB_col, timeB_col = VAR_MAPPING[varB_key]

dfA = sheets_dict[sheetA]
dfB = sheets_dict[sheetB]

seriesA = convert_to_numeric(dfA[valA_col], valA_col)
seriesB = convert_to_numeric(dfB[valB_col], valB_col)

# -------------------------
# Correlation parameters
# -------------------------
st.sidebar.header("Correlation Parameters")
lags_hours = st.sidebar.text_input("Lags (hours, comma-separated)", "0,1,2,6,12,24,48")
lags_hours = [float(x.strip()) for x in lags_hours.split(",")]

match_window_hours = st.sidebar.number_input("Match window (hours)", value=4.0, min_value=0.1)
min_pairs = st.sidebar.number_input("Minimum paired observations", value=10, min_value=2)
effect_size_thresh = st.sidebar.slider("Minimum |r| for effect size", 0.0, 1.0, 0.25)
alpha = st.sidebar.slider("Significance alpha", 0.001, 0.2, 0.05)

# -------------------------
# Run correlation
# -------------------------
if st.button("Run Correlation"):
    with st.spinner("Computing correlations..."):
        results_df, significant_df = find_correlations(
            seriesA, seriesB,
            lags_hours=lags_hours,
            match_window_hours=match_window_hours,
            min_pairs=min_pairs,
            effect_size_thresh=effect_size_thresh,
            alpha=alpha
        )

    st.subheader("Top Results (sorted by |Pearson r|)")
    st.dataframe(results_df.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False))

    st.subheader("Significant Correlations")
    if significant_df.empty:
        st.info("No significant correlations found under current parameters.")
    else:
        st.dataframe(significant_df)
