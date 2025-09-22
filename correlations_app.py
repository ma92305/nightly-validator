# correlations_app.py
import streamlit as st
import pandas as pd
import numpy as np
from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix

# --- Helper functions for human-readable feature names ---
def human_readable_feature(name):
    """
    Convert raw variable names into readable English for correlations output.
    Adds emojis and lag formatting if needed.
    """
    # Detect lag
    lag_note = " roughly 7 days ago" if "_7d_avg" in name else ""

    # Base name without lag suffix
    base_name = name.replace("_7d_avg", "").replace("_sum", "").replace("_count", "").replace("_minutes", " minutes")

    # Map known nutrition features to emojis
    emoji_map = {
        "sugar": "🍬 Sugar",
        "protein": "🥩 Protein",
        "caffeine": "☕️ Caffeine"
    }

    nutrition_map = {
        "meals": "Eating multiple meals totaling a lot in one day",
        "chocolate": "Eating chocolate in a day",
        "ginger": "Consuming ginger",
        "cheese": "Consuming cheese",
        "dairy": "Consuming dairy",
        "gluten": "Consuming gluten",
        "spice": "Consuming spicy food",
        "oil": "Consuming oily foods"
    }

    # Apply emoji mapping first
    for key, val in emoji_map.items():
        if key in base_name.lower():
            return f"{val}{lag_note}"

    # Apply nutrition mapping
    for key, val in nutrition_map.items():
        if key in base_name.lower():
            return f"{val}{lag_note}"

    # Special cases
    if "liquids" in base_name.lower():
        return f"Amount of liquids consumed{lag_note}"
    if "stairs" in base_name.lower():
        return f"Climbing a high number of stairs in one day{lag_note}"
    if "standing" in base_name.lower():
        return f"Spending a long time standing in one day{lag_note}"

    # Weather features: replace underscores with spaces, capitalize words
    if "weather" in base_name.lower():
        return base_name.replace("_", " ").capitalize() + lag_note

    # Default: just replace underscores with spaces and capitalize
    return base_name.replace("_", " ").capitalize() + lag_note

def human_readable_target(name):
    """
    Convert target variables (Y) into readable English.
    """
    mapping = {
        "HR_avg": "average heart rate",
        "HR_max": "maximum heart rate",
        "HR_min": "minimum heart rate",
        "HRV": "HRV",
        "sleep_duration": "sleep duration",
        "sleep_score": "sleep score"
    }
    return mapping.get(name, name.replace("_", " ").capitalize())

# --- Main diary description ---
def diary_style_desc(row):
    f1_clean = human_readable_feature(row['Feature 1'])
    f2_clean = human_readable_target(row['Feature 2'])

    # Determine direction
    if row['r'] > 0:
        verb = "higher" if any(x in f2_clean.lower() for x in ["hr", "sleep", "hrv"]) else "greater"
    else:
        verb = "lower" if any(x in f2_clean.lower() for x in ["hr", "sleep", "hrv"]) else "lesser"

    # Combined certainty metric
    effect_score = abs(row['r']) * 100
    p_boost = max(0, min(50, ((0.05 - row['p']) / 0.05) * 50))
    certainty = int(min(100, effect_score + p_boost))

    return f"{f1_clean} is linked to {verb} {f2_clean} — Certainty: {certainty}/100"

# --- Main Streamlit app ---
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

        # Combined certainty metric
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
