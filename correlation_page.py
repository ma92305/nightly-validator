# --- CORRELATION PAGE (updated) ---
import os
import pandas as pd
import numpy as np
import streamlit as st
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

CORR_CACHE_PATH = "/Users/melinaahmad/Library/Mobile Documents/com~apple~CloudDocs/Shortcuts/New/corr_cache.parquet"
HOUR_LAGS = [0,1,2,3,4,5,6,8,12] + [24*d for d in range(1,8)]  # 1-7 days

# --- Variable builder using new logic ---
@st.cache_data
def build_vars_new(data, hourly_index):
    # Use your optimized build_all_variables_optimized
    return build_all_variables_optimized(data, hourly_index)

# --- Cached correlations loader ---
def load_cached_correlations(cache_path=CORR_CACHE_PATH):
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except:
            return None
    return None

# --- Categorize variables ---
def categorize_var(name):
    if name.startswith('Symptom: ') or name == 'Symptom Total Score':
        return 'Symptoms'
    elif name.startswith('Condition: '):
        return 'Conditions'
    elif name.startswith(('temp_', 'humidity_', 'pressure_','Hourly ')):
        return 'Weather'
    elif name.startswith(('Sleep ', 'Wake ', 'Bedtime')):
        return 'Sleep'
    elif name.startswith(('Heart Rate', 'Tachycardia', 'HRV')):
        return 'Heart Rate'
    elif name.startswith(('Nutrition Item:', 'Meal Amount', 'Water ')):
        return 'Nutrition'
    elif name.startswith(('Stairs', 'Standing', 'Walking', 'Steps')):
        return 'General'
    else:
        return 'Other'

# --- Streamlit correlation page ---
# --- Streamlit correlation page (no cache) ---
def correlation_page():
    st.header("Correlation Explorer with Scenarios")

    # --- Determine global time range from loaded data ---
    all_times = []
    for name, df_ in data.items():
        for col in df_.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    t = pd.to_datetime(df_[col], errors='coerce')
                    if not t.empty:
                        all_times.append(t.min())
                        all_times.append(t.max())
                except:
                    pass
    all_times = [t for t in all_times if pd.notna(t)]
    if not all_times:
        st.write("No datetime data found in your sheets.")
        return
    start, end = min(all_times), max(all_times)
    st.write(f"Data range detected: {pd.to_datetime(start).date()} → {pd.to_datetime(end).date()}")

    # --- User selects date range ---
    col1, col2 = st.columns(2)
    with col1:
        user_start = st.date_input("Start date:", value=pd.to_datetime(start).date())
    with col2:
        user_end = st.date_input("End date:", value=pd.to_datetime(end).date())

    hourly_index = to_hourly_index(user_start, user_end)

    st.info("Building variables (optimized)...")
    vars_dict = build_all_variables_optimized(data, hourly_index)
    st.write(f"Built {len(vars_dict)} variables.")

    # --- Variable categorization ---
    var_categories = {v: categorize_var(v) for v in vars_dict.keys()}

    # --- Correlation scenario selection ---
    scenario = st.selectbox("Select Correlation Scenario:", list(CORRELATION_SCENARIOS.keys()))
    sc = CORRELATION_SCENARIOS[scenario]

    # --- Filter variables per scenario ---
    selected_vars = [v for v in vars_dict if var_categories.get(v,'Other') in sc['groups']]
    filtered_vars_dict = {k: vars_dict[k] for k in selected_vars}

    # --- Lags ---
    lags_choice = st.multiselect("Select lags (hours):", sc['lags'], default=sc['lags'])

    min_abs_r = st.slider("Minimum absolute correlation to show:", 0.0, 1.0, 0.25, 0.01)
    p_threshold = st.number_input("Max p-value to show (NaN = ignore):", value=0.05, format="%.3f")

    # --- Compute correlations on the fly ---
    st.info("Computing correlations (this may take a moment)...")
    all_corr = compute_all_pairwise_correlations_optimized(filtered_vars_dict, lags_hours=lags_choice)

    if all_corr is None or all_corr.empty:
        st.write("No correlations found.")
        return

    # --- Filtering as before ---
    all_corr = all_corr[~((all_corr['var_a']==all_corr['var_b']) & (all_corr['lag_hours']==0))]
    if not sc.get('allow_within_group', True):
        all_corr = all_corr[~all_corr.apply(lambda r: var_categories.get(r['var_a'])==var_categories.get(r['var_b']), axis=1)]
    all_corr = all_corr[all_corr['n']>=6]
    if p_threshold is not None:
        all_corr = all_corr[(all_corr['p'].isna()) | (all_corr['p']<=p_threshold)]
    all_corr = all_corr[all_corr['r'].abs()>=min_abs_r]

    st.write(f"{len(all_corr)} correlations matching filters.")

    # --- Summary generation (reuse old logic) ---
    corr_df = all_corr.rename(columns={'var_a':'Variable X','var_b':'Variable Y','r':'Correlation','p':'P-Value','n':'N'})

    def generate_summary(corr_df):
        summaries=[]
        filtered_df = corr_df[(corr_df['P-Value']<=0.05)&(corr_df['N']>=20)]
        grouped = filtered_df.groupby(['Variable X','Variable Y'])
        for (var1,var2), group in grouped:
            best_row = group.loc[group['P-Value'].idxmin()]
            corr = best_row['Correlation']
            p_value = best_row['P-Value']
            n = best_row['N']
            best_lag = best_row.get('lag_hours',0)
            direction = "increase" if corr>0 else "decrease"
            lag_str = f" {best_lag}h later" if best_lag>0 else f" {abs(best_lag)}h earlier" if best_lag<0 else ""
            summaries.append(f"An increase in {var1} is associated with a {direction} in {var2}{lag_str} (p={p_value:.3f}, n={n})")
        return summaries

    summary_list = generate_summary(corr_df)
    if summary_list:
        st.subheader("Summary of Likely Real Correlations")
        for s in summary_list:
            st.write(f"- {s}")
    else:
        st.info("No strong statistically significant correlations found.")

    # --- Display table or heatmap ---
    view = st.radio("View as:", ['Heatmap (single lag)','Top correlations table'], index=1)
    if view.startswith('Heatmap'):
        chosen_lag = st.selectbox("Choose lag (hours) to display:", sorted(all_corr['lag_hours'].unique()))
        chosen_method = st.selectbox("Choose method:", ['pearson','spearman'])
        subset = all_corr[(all_corr['lag_hours']==chosen_lag)&(all_corr['method']==chosen_method)]
        if subset.empty:
            st.write("No results for this lag/method with current filters.")
            return
        names = sorted(set(subset['var_a']).union(set(subset['var_b'])))
        mat = pd.DataFrame(np.nan, index=names, columns=names)
        for _, row in subset.iterrows():
            mat.loc[row['var_a'], row['var_b']] = row['r']
        fig, ax = plt.subplots(figsize=(12, max(6,len(names)*0.25)))
        sns.heatmap(mat.astype(float), cmap='coolwarm', center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
        ax.set_title(f"Correlation Matrix (lag={chosen_lag}h, method={chosen_method})")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        top_n = st.number_input("Show top N results (by |r|):", min_value=10, max_value=1000, value=100, step=10)
        show_only_real = st.checkbox("Show only likely real correlations (p≤0.05 and n≥20)", value=True)
        df_show = all_corr.copy()
        if show_only_real:
            df_show = df_show[(df_show['p']<=0.05)&(df_show['n']>=20)]
        df_show['abs_r'] = df_show['r'].abs()
        df_show = df_show.sort_values('abs_r', ascending=False).head(int(top_n))
        st.dataframe(df_show[['var_a','var_b','lag_hours','method','r','p','n']])
