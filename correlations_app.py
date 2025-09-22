# correlations_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix, compute_lagged_correlations

def show_correlation_page(sheets):
    st.title("Correlation Explorer — Health Logs")
    st.markdown(
        "This page aggregates your sheets into daily features and computes correlations. "
        "The strongest correlations are summarized in easy-to-read plain English."
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

    # --- Diary-style description function with certainty ---
    def diary_style_desc(row):
        f1 = row['Feature 1']
        f2 = row['Feature 2']

        # Detect lag
        lag_note = " roughly 7 days ago" if "_7d_avg" in f1 else ""

        # Clean feature names
        f1_clean = f1.replace("nutrition_", "").replace("_7d_avg", "").replace("_sum", "").replace("_minutes", " minutes").replace("_count", " count")
        f2_clean = f2.replace("HR_avg", "average heart rate")\
                     .replace("HR_max", "maximum heart rate")\
                     .replace("HRV", "HRV")\
                     .replace("sleep_duration", "sleep duration")\
                     .replace("sleep_score", "sleep score")

        # Determine direction
        verb = "higher" if row['r'] > 0 and any(x in f2_clean for x in ["HR", "HRV", "sleep"]) else \
               "lower" if row['r'] < 0 and any(x in f2_clean for x in ["HR", "HRV", "sleep"]) else \
               "more" if row['r'] > 0 else "less"

        # Compute certainty
        certainty = max(0, min(100, int(abs(row['r']) * max(0, 1 - row['p'] / 0.05) * 100)))

        # Add human-readable thresholds/context
        if "meals" in f1_clean:
            sentence = f"Eating multiple meals totaling a lot in one day{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "stairs" in f1_clean:
            sentence = f"Climbing a high number of stairs in one day{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "standing" in f1_clean:
            sentence = f"Spending a long time standing in one day{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Chocolate" in f1_clean:
            sentence = f"Eating chocolate in a day{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Caffeine" in f1_clean:
            sentence = f"Drinking caffeine{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Ginger" in f1_clean:
            sentence = f"Consuming ginger{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Cheese" in f1_clean:
            sentence = f"Consuming cheese{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Dairy" in f1_clean:
            sentence = f"Consuming dairy{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Gluten" in f1_clean:
            sentence = f"Consuming gluten{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Spice" in f1_clean:
            sentence = f"Consuming spicy food{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Oil" in f1_clean:
            sentence = f"Consuming oil{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        else:
            sentence = f"{f1_clean}{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"

        return sentence, certainty

    # --- Top correlations summary ---
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

        # Compute certainty
        summary_df['certainty'] = summary_df.apply(lambda row: max(0, int(abs(row['r']) * max(0, 1 - row['p'] / 0.05) * 100)), axis=1)

        # Separate by certainty and p thresholds
        likely_real = summary_df[(summary_df['certainty'] >= 50)]  # can adjust threshold
        possible = summary_df[(summary_df['certainty'] < 50) & (summary_df['certainty'] > 0)]

        # Sort by certainty descending
        likely_real = likely_real.sort_values(by='certainty', ascending=False)
        possible = possible.sort_values(by='certainty', ascending=False)

        st.subheader("Likely real correlations")
        if not likely_real.empty:
            for idx, row in likely_real.iterrows():
                desc, _ = diary_style_desc(row)
                st.write(f"- {desc}")
        else:
            st.write("No strong correlations found.")

        st.subheader("Possible correlations (borderline)")
        if not possible.empty:
            for idx, row in possible.iterrows():
                desc, _ = diary_style_desc(row)
                st.write(f"- {desc}")
        else:
            st.write("No borderline correlations found.")
    else:
        st.write("No correlations found.")

    # --- Lagged correlation explorer ---
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

    # --- Time-series overlay viewer ---
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
