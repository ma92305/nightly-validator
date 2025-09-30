
# correlations_app.py
import streamlit as st
import pandas as pd
import numpy as np
from correlations import compute_daily_aggregates, compute_pairwise_cross_group_matrix
import matplotlib.pyplot as plt

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

# -----------------------------
# Migraine detection functions (weighted)
# -----------------------------

# Define symptoms with weights
SYMPTOM_WEIGHTS = {
    "Headache": 2,
    "Right side headache": 2,
    "Left side headache": 2,
    "Nausea": 1.5,
    "Sensory sensitivity": 1.5,
    "Vision issues": 1.5,
    "Eye pain": 1.5,
    "Base of head pain": 1.5,
    "Neck pain": 1.2,
    "Fatigue": 0.5,
    "Dizziness": 0.5,
    "Fuzziness": 0.5,
    "Weakness/shakiness": 0.5,
    "Brain fog": 0.5,
    "Heavy eyes": 0.5,
    "Negative mood": 0.3,
    "Overheating": 0.3,
    "Bloating": 0.2,
    "Low appetite/early satiety": 0.2,
    "Stomach cramping": 0.2
}

MIGRAINE_SYMPTOMS = list(SYMPTOM_WEIGHTS.keys())

SEVERITY_MAP = {"⚪️":0, "🟡":1, "🟠":2, "🔴":3, "🟣":4, "none":0, None:0}

def preprocess_symptom_matrix(symptom_df):
    """
    Convert symptom DataFrame to numeric matrix.
    Handles both wide-format and long-format sheets.
    """
    # Check if sheet is wide format
    missing_cols = [s for s in MIGRAINE_SYMPTOMS if s not in symptom_df.columns]
    if not missing_cols:
        # Already wide format
        num_df = symptom_df[MIGRAINE_SYMPTOMS].replace(SEVERITY_MAP)
        # Convert index to datetime if possible
        if not pd.api.types.is_datetime64_any_dtype(num_df.index):
            num_df.index = pd.to_datetime(num_df.index, errors='coerce')
    else:
        # Likely long format: columns "time", "item", "severity"
        if not {"time", "item", "severity"}.issubset(symptom_df.columns):
            raise ValueError(
                "Sheet does not contain expected columns for long format: 'time', 'item', 'severity'"
            )
        # Pivot to wide format
        symptom_wide = symptom_df.pivot_table(
            index="time",
            columns="item",
            values="severity",
            aggfunc="first"  # take first if duplicates
        )
        # Convert index to datetime
        symptom_wide.index = pd.to_datetime(symptom_wide.index, errors='coerce')
        # Add missing symptom columns
        for s in MIGRAINE_SYMPTOMS:
            if s not in symptom_wide.columns:
                symptom_wide[s] = 0
        # Replace emojis with numbers
        num_df = symptom_wide[MIGRAINE_SYMPTOMS].replace(SEVERITY_MAP)

    # Force numeric type
    num_df = num_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return num_df

def compute_baseline(num_df):
    """
    Compute median and MAD for each symptom to define typical baseline.
    Uses manual MAD calculation to support newer pandas versions.
    """
    median = num_df.median()
    mad = (num_df - median).abs().median()  # median absolute deviation
    # Prevent division by zero
    mad = mad.replace(0, 1)
    return median, mad

def score_migraine(num_df, median, mad, weights=SYMPTOM_WEIGHTS):
    """
    Compute weighted migraine score per timestamp.
    Only unusually high severity counts; weights emphasize migraine-specific symptoms.
    """
    z_scores = (num_df - median) / mad
    z_scores = z_scores.clip(lower=0)  # ignore below-baseline symptoms

    # Apply weights
    weight_series = pd.Series(weights)
    weighted_scores = z_scores * weight_series

    # Sum across symptoms
    migraine_score = weighted_scores.sum(axis=1)
    return migraine_score

def detect_migraine_episodes(migraine_score, threshold=3, min_duration=1):
    """
    Identify consecutive timestamps where weighted migraine_score exceeds threshold.
    Returns list of dicts with start, end, and peak score timestamps.
    """
    episodes = []
    in_episode = False
    start_idx = None

    for idx, score in migraine_score.items():
        if score >= threshold:
            if not in_episode:
                in_episode = True
                start_idx = idx
        else:
            if in_episode:
                end_idx = idx
                # Only keep episodes with at least min_duration timestamps
                if (migraine_score.loc[start_idx:end_idx].shape[0] >= min_duration):
                    episodes.append({
                        "start": start_idx,
                        "end": end_idx,
                        "peak_score": migraine_score[start_idx:end_idx].max()
                    })
                in_episode = False
    if in_episode:
        end_idx = migraine_score.index[-1]
        if (migraine_score.loc[start_idx:end_idx].shape[0] >= min_duration):
            episodes.append({
                "start": start_idx,
                "end": end_idx,
                "peak_score": migraine_score[start_idx:].max()
            })
    return episodes

def display_migraine_episodes(st, num_df, migraine_score, episodes):
    """
    Streamlit display for detected migraine episodes.
    num_df: numeric symptom matrix with datetime index
    """
    st.subheader("Migraine Episode Analysis")
    if not episodes:
        st.write("No migraine episodes detected.")
        return

    for i, ep in enumerate(episodes, 1):
        st.markdown(f"**Episode {i}**")
        st.write(f"Start: {ep['start']}, End: {ep['end']}, Peak score: {ep['peak_score']:.2f}")

        # Find peak time
        peak_time = migraine_score[ep['start']:ep['end']].idxmax()

        # Safely slice numeric dataframe
        if peak_time in num_df.index:
            snapshot = num_df.loc[peak_time, MIGRAINE_SYMPTOMS]
        else:
            # fallback: nearest timestamp
            snapshot = num_df.iloc[(num_df.index - peak_time).abs().argmin()]
            st.write("(Used nearest timestamp for snapshot)")

        st.write("Snapshot of symptoms at peak:")
        st.dataframe(snapshot.to_frame("Severity"))

# --- Main Streamlit app ---
def show_correlation_page(sheets):
    st.title("Correlation Explorer — Health Logs")
    st.markdown(
        "This page aggregates your sheets into daily features and computes correlations. "
        "The strongest correlations are summarized in easy-to-read plain English with a certainty meter."
    )

    # 1) Build daily aggregates for correlations
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

    # -----------------------------
    # 3) Migraine detection
    # -----------------------------
    st.markdown("---")
    st.subheader("Migraine Detection")
    
    symptom_sheet_name = None
    symptom_df = None
    
    # Look for a wide-format sheet first
    for name, df in sheets.items():
        if all(s in df.columns for s in MIGRAINE_SYMPTOMS):
            symptom_sheet_name = name
            symptom_df = df.copy()
            break
    
    # If no wide-format sheet, look for long-format sheet
    if symptom_sheet_name is None:
        for name, df in sheets.items():
            if {'item', 'time', 'severity'}.issubset(df.columns):
                symptom_sheet_name = name
                symptom_df = df.copy()
                break
    
    if symptom_sheet_name is None:
        st.warning("No sheet with timestamped migraine symptom data found.")
        st.write("Expected columns: wide-format symptoms OR long-format ['item','time','severity']")
        return
    
    st.write(f"Using sheet '{symptom_sheet_name}' for migraine detection.")
    st.write("Sample data:")
    st.dataframe(symptom_df.head(10))
    
    # Preprocess (handles long or wide internally)
    num_df = preprocess_symptom_matrix(symptom_df)
    median, mad = compute_baseline(num_df)
    migraine_score = score_migraine(num_df, median, mad)
    episodes = detect_migraine_episodes(migraine_score, threshold=3)
    display_migraine_episodes(st, symptom_df, migraine_score, episodes)
