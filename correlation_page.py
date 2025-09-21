import streamlit as st
import pandas as pd
from correlation_engine import find_correlations

def correlation_page(dbx=None, sheets_dict=None):
    """
    Streamlit page for running time-lagged correlations between two variables.
    
    Inputs:
      - dbx: optional Dropbox object (if needed to reload files)
      - sheets_dict: optional preloaded dict of dataframes (e.g., from Excel)
    """
    st.header("Correlation Analysis")

    # --- Load data ---
    if sheets_dict is None:
        st.warning("No data provided. Please load Excel data first.")
        return

    all_sheets = list(sheets_dict.keys())
    sheet_to_use = st.selectbox("Select sheet to analyze", all_sheets)

    if sheet_to_use not in sheets_dict:
        st.warning("Selected sheet not found in data.")
        return

    df = sheets_dict[sheet_to_use].copy()
    st.write(f"Loaded sheet `{sheet_to_use}` with {len(df)} rows.")

    # --- Select columns for correlation ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to run correlation.")
        return

    col_A = st.selectbox("Variable A (cause)", numeric_cols)
    col_B = st.selectbox("Variable B (effect)", [c for c in numeric_cols if c != col_A])
    
    time_col = st.selectbox("Time column", df.columns)

    # --- Correlation settings ---
    lags_input = st.text_input("Lags in hours (comma-separated)", "0,1,2,6,12,24,48")
    try:
        lags_hours = [float(x.strip()) for x in lags_input.split(",")]
    except Exception:
        st.error("Invalid lag input")
        return

    match_window = st.number_input("Match window in hours", min_value=0.5, max_value=24.0, value=4.0, step=0.5)
    min_pairs = st.number_input("Minimum pairs per lag", min_value=3, value=10, step=1)
    effect_thresh = st.number_input("Effect size threshold |r|", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    if st.button("Run correlations"):
        with st.spinner("Calculating correlations..."):
            try:
                results_df, sig_df = find_correlations(
                    df, df,
                    A_time_col=time_col, B_time_col=time_col,
                    A_value_col=col_A, B_value_col=col_B,
                    lags_hours=lags_hours,
                    match_window_hours=match_window,
                    min_pairs=min_pairs,
                    effect_size_thresh=effect_thresh
                )
            except Exception as e:
                st.error(f"Error running correlations: {e}")
                return

        st.subheader("All lag results")
        st.dataframe(results_df)

        if sig_df.empty:
            st.info("No significant correlations found with current settings.")
        else:
            st.subheader("Significant correlations")
            st.dataframe(sig_df)

        # --- Optional plotting ---
        st.subheader("Lag vs Pearson r plot")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(results_df["lag_hours"], results_df["pearson_r"], marker="o", label="Pearson r")
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_xlabel("Lag (hours)")
            ax.set_ylabel("Pearson r")
            ax.set_title(f"{col_A} → {col_B} correlations")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate plot: {e}")
