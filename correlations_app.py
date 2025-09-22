# correlations_app.py
import streamlit as st
import pandas as pd
import numpy as np
from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix

def show_correlation_page(sheets):
    st.title("Correlation Explorer — Health Logs")
    st.markdown(
        "This page aggregates your sheets into daily features and computes correlations. "
        "The strongest correlations are summarized in easy-to-read plain English with a certainty meter."
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

    # --- Diary-style description with combined certainty metric ---
    def diary_style_desc(row):
        f1 = row['Feature 1']
        f2 = row['Feature 2']

        # Detect lag
        lag_note = " roughly 7 days ago" if "_7d_avg" in f1 else ""

        # Clean feature names
        f1_clean = f1.replace("nutrition_", "").replace("_7d_avg", "").replace("_sum", "") \
                     .replace("_minutes", " minutes").replace("_count", " count")
        f2_clean = f2.replace("HR_avg", "average heart rate")\
                     .replace("HR_max", "maximum heart rate")\
                     .replace("HRV", "HRV")\
                     .replace("sleep_duration", "sleep duration")\
                     .replace("sleep_score", "sleep score")

        # Determine direction
        if row['r'] > 0:
            verb = "higher" if "HR" in f2_clean or "HRV" in f2_clean or "sleep" in f2_clean else "greater"
        else:
            verb = "lower" if "HR" in f2_clean or "HRV" in f2_clean or "sleep" in f2_clean else "lesser"

        # --- Combined certainty metric ---
        # Base: effect size |r| scaled 0-100
        effect_score = abs(row['r']) * 100
        # P-value boost if p <= 0.05, scaled to max +50 points
        p_boost = max(0, min(50, ((0.05 - row['p']) / 0.05) * 50))
        certainty = int(min(100, effect_score + p_boost))

        # Human-readable thresholds/context
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
            sentence = f"Consuming dairy{lag_note} is correlated with {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Gluten" in f1_clean:
            sentence = f"Consuming gluten{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Spice" in f1_clean:
            sentence = f"Consuming spicy food{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        elif "Oil" in f1_clean:
            sentence = f"Consuming oily foods{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"
        else:
            sentence = f"{f1_clean}{lag_note} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"

        return sentence

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
        top_corrs = summary_df.reindex(summary_df['r'].abs().sort_values(ascending=False).index)

        # --- Apply categories ---
        def compute_certainty(p, r):
            effect_score = abs(r) * 100
            p_boost = max(0, min(50, ((0.05 - p) / 0.05) * 50))
            return int(min(100, effect_score + p_boost))

        top_corrs['certainty'] = top_corrs.apply(lambda x: compute_certainty(x['p'], x['r']), axis=1)

        likely_real = top_corrs[top_corrs['certainty'] >= 50].sort_values(by='certainty', ascending=False)
        possible = top_corrs[(top_corrs['certainty'] >= 20) & (top_corrs['certainty'] < 50)].sort_values(by='certainty', ascending=False)

        st.subheader("Likely real correlations")
        if not likely_real.empty:
            likely_real['Description'] = likely_real.apply(diary_style_desc, axis=1)
            for desc in likely_real['Description']:
                st.write(f"- {desc}")
        else:
            st.write("No strong correlations found.")

        st.subheader("Possible correlations (borderline)")
        if not possible.empty:
            possible['Description'] = possible.apply(diary_style_desc, axis=1)
            for desc in possible['Description']:
                st.write(f"- {desc}")
        else:
            st.write("No borderline correlations found.")
    else:
        st.write("No correlations found.")
