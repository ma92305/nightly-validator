
# correlations_app.py
import streamlit as st
import pandas as pd
import numpy as np
from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix
import matplotlib.pyplot as plt

def detect_migraine_days(daily_df, migraine_col="migraine"):
    """
    Returns indices of migraine days based on daily_df.
    """
    if migraine_col not in daily_df.columns:
        return []

    migraine_days = daily_df.index[daily_df[migraine_col] == 1].tolist()
    return migraine_days


def identify_prodrome_signs(daily_df, migraine_col="migraine", lookback=1, min_count=3):
    """
    Identify symptoms/conditions that appear in prodrome (1 day before migraine) more often than baseline.
    """
    if migraine_col not in daily_df.columns:
        return pd.DataFrame()

    migraine_days = detect_migraine_days(daily_df, migraine_col)
    if not migraine_days:
        return pd.DataFrame()

    # Collect prodrome days (lookback days before migraine)
    prodrome_days = []
    for d in migraine_days:
        if d - lookback in daily_df.index:
            prodrome_days.append(d - lookback)

    if not prodrome_days:
        return pd.DataFrame()

    # Candidate columns = all binary/symptom-like columns except migraine flag
    symptom_cols = [c for c in daily_df.columns if c != migraine_col]

    records = []
    for col in symptom_cols:
        prod_freq = daily_df.loc[prodrome_days, col].mean()
        base_freq = daily_df.loc[daily_df.index.difference(migraine_days + prodrome_days), col].mean()

        if prod_freq > base_freq and daily_df.loc[prodrome_days, col].sum() >= min_count:
            records.append({
                "Symptom": col,
                "Prodrome Frequency": prod_freq,
                "Baseline Frequency": base_freq,
                "Lift": prod_freq - base_freq
            })

    results = pd.DataFrame(records).sort_values(by="Lift", ascending=False)
    return results

def get_threshold_description(feature_name, daily_data):
    """
    Returns a human-readable threshold description for a given feature.
    Handles continuous (standing, stairs, liquids) and binary/rare nutrition items differently.
    """
    base_name = (
        feature_name.replace("_7d_avg", "")
        .replace("_sum", "")
        .replace("_count", "")
        .replace("_minutes", "")
    )

    # Thresholds for continuous/numeric features (standing, stairs, liquids)
    if base_name in ["standing_minutes", "stairs_count", "liquids_amount"]:
        threshold = daily_data[feature_name].quantile(0.75)
        if base_name == "standing_minutes":
            return f"Spending more than {int(threshold)} minutes standing in one day"
        elif base_name == "stairs_count":
            return f"Climbing more than {int(threshold)} stairs in one day"
        elif base_name == "liquids_amount":
            return f"Consuming more than {int(threshold)} ml of liquids in a day"

    # For count-based nutrition items and meals
    nutrition_items = [
        "ginger", "cheese", "dairy", "sugar", "protein",
        "caffeine", "chocolate", "meals"
    ]
    for item in nutrition_items:
        if item in base_name.lower():
            threshold = daily_data[feature_name].quantile(0.75)

            # ✅ Special-case: 0 means binary or rare → "at least once"
            if threshold <= 0.5:
                return f"Days on which you consumed {item} at least once"
            else:
                return f"Days on which you consumed {item} {int(round(threshold))} or more times"

    # Default fallback
    return human_readable_feature(feature_name)

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
def diary_style_desc_with_threshold(row, daily):
    f1_name = row['Feature 1']
    f2_clean = human_readable_target(row['Feature 2'])
    threshold_desc = get_threshold_description(f1_name, daily)

    # Determine direction and effect
    if row['r'] > 0:
        verb = "increase in" if any(x in f2_clean.lower() for x in ["hr", "sleep", "hrv"]) else "greater"
    else:
        verb = "decrease in" if any(x in f2_clean.lower() for x in ["hr", "sleep", "hrv"]) else "lesser"

    # Combined certainty metric
    effect_score = abs(row['r']) * 100
    p_boost = max(0, min(50, ((0.05 - row['p']) / 0.05) * 50))
    certainty = int(min(100, effect_score + p_boost))

    # Optional: include % change approximation
    percent_effect = f"{int(effect_score)}% "

    return f"{threshold_desc} is linked to a {percent_effect}{verb} {f2_clean} — Certainty: {certainty}/100"

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

        def compute_certainty(p, r):
            effect_score = abs(r) * 100
            p_boost = max(0, min(50, ((0.05 - p) / 0.05) * 50))
            return int(min(100, effect_score + p_boost))

        top_corrs['certainty'] = top_corrs.apply(lambda x: compute_certainty(x['p'], x['r']), axis=1)

        # 🚨 Only keep statistically significant correlations
        top_corrs = top_corrs[top_corrs['p'] < 0.05]

        # Define likely_real and possible after filtering
        likely_real = top_corrs[top_corrs['certainty'] >= 50].sort_values(by='certainty', ascending=False)
        possible = top_corrs[(top_corrs['certainty'] >= 20) & (top_corrs['certainty'] < 50)].sort_values(by='certainty', ascending=False)

        st.subheader("Likely real correlations")
        if not likely_real.empty:
            for i, row in likely_real.iterrows():
                description = diary_style_desc_with_threshold(row, daily)

                with st.expander(description):
                    f1 = row['Feature 1']
                    f2 = row['Feature 2']
                    df_plot = daily[[f1, f2]].dropna()

                    if len(df_plot) > 1:
                        x = df_plot[f1]
                        y = df_plot[f2]
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.scatter(x, y, alpha=0.6)
                        m, b = np.polyfit(x, y, 1)
                        ax.plot(x, m*x + b, color='red', linestyle='--')
                        ax.set_xlabel(human_readable_feature(f1))
                        ax.set_ylabel(human_readable_target(f2))
                        ax.set_title(f"r = {row['r']:.2f}, p = {row['p']:.3f}")
                        st.pyplot(fig)
                    else:
                        st.write("Not enough data to plot.")

        st.subheader("Possible correlations (borderline)")
        if not possible.empty:
            for i, row in possible.iterrows():
                description = diary_style_desc_with_threshold(row, daily)

                with st.expander(description):
                    f1 = row['Feature 1']
                    f2 = row['Feature 2']
                    df_plot = daily[[f1, f2]].dropna()

                    if len(df_plot) > 1:
                        x = df_plot[f1]
                        y = df_plot[f2]
                        fig, ax = plt.subplots(figsize=(6,4))
                        ax.scatter(x, y, alpha=0.6)
                        m, b = np.polyfit(x, y, 1)
                        ax.plot(x, m*x + b, color='red', linestyle='--')
                        ax.set_xlabel(human_readable_feature(f1))
                        ax.set_ylabel(human_readable_target(f2))
                        ax.set_title(f"r = {row['r']:.2f}, p = {row['p']:.3f}")
                        st.pyplot(fig)
                    else:
                        st.write("Not enough data to plot.")
        else:
            st.write("No borderline correlations found.")
    else:
        st.write("No correlations found.")

    # --- Migraine episode analysis ---
    st.markdown("---")
    st.header("Migraine Episode Analysis")

    migraine_days = detect_migraine_days(daily)
    st.write(f"Detected {len(migraine_days)} migraine days.")

    if migraine_days:
        prodrome_signs = identify_prodrome_signs(daily)
        if not prodrome_signs.empty:
            st.subheader("Likely Prodrome Warning Signs (1 day before)")
            st.dataframe(prodrome_signs, use_container_width=True)
        else:
            st.info("No strong prodrome patterns detected.")
    else:
        st.info("No migraine days found in dataset.")
