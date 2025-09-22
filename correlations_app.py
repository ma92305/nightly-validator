# correlations_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix, compute_lagged_correlations

def show_correlation_page(sheets):
    st.title("Correlation Explorer — Health Logs")
    st.markdown("This page aggregates your sheets into daily features and computes correlations. Use controls below to explore.")

    # 1) Build daily aggregates
    with st.spinner("Aggregating daily features..."):
        daily = compute_daily_aggregates(sheets)

    if daily.empty:
        st.warning("No daily features produced. Check sheet names and sample data.")
        st.write("Available sheets:", list(sheets.keys()))
        return

    st.subheader("Preview daily features")
    st.dataframe(daily.tail(50))

    # 2) Column selection
    all_cols = daily.columns.tolist()
    selected_cols = st.multiselect("Select features to include in correlation matrix", all_cols, default=all_cols[:8])

    if len(selected_cols) >= 2:
        method = st.radio("Correlation method", ["pearson", "spearman"])
        corr_df, p_df = compute_pairwise_correlations(daily, columns=selected_cols, method=method)

        st.subheader("Correlation matrix")
        st.dataframe(corr_df.style.format("{:.2f}"))
        st.caption("P-values shown separately below.")

        fig, ax = plt.subplots(figsize=(max(6, len(selected_cols)), max(4, len(selected_cols)/2)))
        sns.heatmap(corr_df.astype(float), annot=True, vmin=-1, vmax=1, cmap="vlag", ax=ax, fmt=".2f")
        st.pyplot(fig)

        st.subheader("P-values")
        st.dataframe(p_df.style.format("{:.3f}"))

        # allow download of matrix
        if st.button("Download correlation matrix CSV"):
            csv = corr_df.to_csv().encode('utf-8')
            st.download_button("Download", csv, file_name="correlation_matrix.csv", mime="text/csv")

    # 3) Lagged correlation explorer for two features
    st.markdown("---")
    st.subheader("Lagged correlation (single pair)")
    col_x = st.selectbox("X (predictor)", all_cols, index=0)
    col_y = st.selectbox("Y (response)", all_cols, index=min(1, len(all_cols)-1))
    max_lag = st.slider("Max lag (in days)", 0, 30, 7)
    freq = st.radio("Frequency for lagging", ["D", "H"], help="D = days, H = hours. Use H only if both series are hourly-indexed.")
    method2 = st.radio("Method for lagged correlation", ["pearson", "spearman"], index=0, key="lag_method")

    if st.button("Compute lagged correlations"):
        s_x = daily[col_x].dropna()
        s_y = daily[col_y].dropna()
        # align on union index and pass
        merged_index = s_x.index.union(s_y.index)
        s_x = s_x.reindex(merged_index)
        s_y = s_y.reindex(merged_index)
        lagged = compute_lagged_correlations(s_x, s_y, max_lag=max_lag, freq=freq, method=method2)
        st.line_chart(lagged.set_index('lag')['corr'])
        st.dataframe(lagged)

        # show the lag with highest abs(corr)
        best = lagged.loc[lagged['corr'].abs().idxmax()]
        st.write(f"Highest |corr| at lag {int(best['lag'])}: corr={best['corr']:.3f}, p={best['pval']} (n={int(best['n'])})")

    # 4) Time-series overlay viewer for two columns
    st.markdown("---")
    st.subheader("Time series overlay")
    ts_x = st.selectbox("Time series X", all_cols, index=0, key="ts_x")
    ts_y = st.selectbox("Time series Y", all_cols, index=min(1, len(all_cols)-1), key="ts_y")
    if st.button("Plot time series overlay"):
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(daily.index, daily[ts_x], label=ts_x)
        ax.plot(daily.index, daily[ts_y], label=ts_y)
        ax.legend()
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        st.pyplot(fig)
