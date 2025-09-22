# correlations_app.py
import streamlit as st
import pandas as pd
import numpy as np
from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix, compute_lagged_correlations

def show_correlation_page(sheets):
    st.title("Correlation Explorer — Health Logs")
    st.markdown(
        "This page aggregates your sheets into daily features and computes correlations. "
        "The top correlations are automatically summarized in human-readable language."
    )

    # 1) Build daily aggregates
    with st.spinner("Aggregating daily features..."):
        daily = compute_daily_aggregates(sheets)

    if daily.empty:
        st.warning("No daily features produced. Check sheet names and sample data.")
        st.write("Available sheets:", list(sheets.keys()))
        return

    st.subheader("Preview daily features")
    st.dataframe(daily.tail(50))

    all_cols = daily.columns.tolist()
    if len(all_cols) < 2:
        st.warning("Not enough features to compute correlations.")
        return

    # 2) Compute correlations across all columns
    method = st.radio("Correlation method", ["pearson", "spearman"], index=0)
    corr_df, p_df = compute_pairwise_cross_group_matrix(daily, method=method, min_periods=3)

    # --- Top correlations summary (human-readable) ---
    st.markdown("---")
    st.subheader("Top correlations summary")

    # Flatten correlation matrix (upper triangle only)
    corr_flat = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
    p_flat = p_df.where(np.triu(np.ones(p_df.shape), k=1).astype(bool))

    summary_list = []
    for col1 in corr_flat.columns:
        for col2 in corr_flat.index:
            r = corr_flat.loc[col2, col1]
            p = p_flat.loc[col2, col1]
            if pd.notna(r):
                summary_list.append({
                    'Feature 1': col1,
                    'Feature 2': col2,
                    'r': r,
                    'p': p
                })

    if summary_list:
        summary_df = pd.DataFrame(summary_list)
        top_corrs = summary_df.reindex(summary_df['r'].abs().sort_values(ascending=False).index).head(20)

        # Human-readable description
        def describe_corr(row):
            abs_r = abs(row['r'])
            if abs_r >= 0.8:
                strength = "very strong"
            elif abs_r >= 0.6:
                strength = "strong"
            elif abs_r >= 0.4:
                strength = "moderate"
            elif abs_r >= 0.2:
                strength = "weak"
            else:
                strength = "very weak"
            direction = "positively" if row['r'] > 0 else "negatively"
            return f"{row['Feature 1']} is {direction} correlated with {row['Feature 2']} ({strength}, r={row['r']:.2f}, p={row['p']:.3f})"

        top_corrs['Description'] = top_corrs.apply(describe_corr, axis=1)

        # Display table
        st.table(top_corrs[['Feature 1', 'Feature 2', 'r', 'p', 'Description']])

        # Human-readable summary
        st.markdown("**Human-readable top correlations:**")
        for desc in top_corrs['Description']:
            st.write(f"- {desc}")
    else:
        st.write("No correlations found.")

    # 3) Lagged correlation explorer
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
        merged_index = s_x.index.union(s_y.index)
        s_x = s_x.reindex(merged_index)
        s_y = s_y.reindex(merged_index)
        lagged = compute_lagged_correlations(s_x, s_y, max_lag=max_lag, freq=freq, method=method2)
        st.line_chart(lagged.set_index('lag')['corr'])
        st.dataframe(lagged)
        best = lagged.loc[lagged['corr'].abs().idxmax()]
        st.write(f"Highest |corr| at lag {int(best['lag'])}: corr={best['corr']:.3f}, p={best['pval']} (n={int(best['n'])})")

    # 4) Time-series overlay viewer
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
