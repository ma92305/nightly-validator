import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, time
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# --- Config ---
CORR_CACHE_PATH = "corr_cache.parquet"  # You may want to make this user-specific or use Streamlit's cache
HOUR_LAGS = [0,1,2,3,4,5,6,8,12] + [24 * d for d in range(1,8)]  # 0-7 days (hours)

# --- Helper Functions (should be in your utils if used elsewhere) ---
def to_hourly_index(start, end):
    start_h = pd.to_datetime(start).floor('H')
    end_h = pd.to_datetime(end).ceil('H')
    return pd.date_range(start=start_h, end=end_h, freq='H')

def point_series_to_hourly(series, hourly_index, method='nearest'):
    series = series.dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    out = pd.Series(index=hourly_index, dtype=float)
    if method == 'nearest':
        snapped = series.copy()
        snapped.index = snapped.index.round('H')
        snapped = snapped.groupby(snapped.index).mean()
        snapped = snapped.reindex(hourly_index, fill_value=np.nan)
        out.update(snapped)
    elif method == 'ffill':
        out = out.ffill()
    return out

# --- Variable Extraction (use your app's standard function) ---
def build_all_variables(data, hourly_index):
    # Call your standardized app-wide variable extraction here!
    # If you have a unified function, call it, else copy from your latest page.
    # For illustration, let's assume you already have build_all_variables elsewhere, possibly in utils.py.
    from utils import build_all_variables as build_vars
    return build_vars(data, hourly_index)

# --- Correlation computation ---
def compute_all_pairwise_correlations(vars_dict, lags_hours=HOUR_LAGS, methods=['pearson','spearman'], n_jobs=-1):
    names = sorted(vars_dict.keys())
    master = pd.DataFrame(vars_dict).sort_index()
    results = []

    def process_pair(a, b, lag):
        out = []
        try:
            b_shifted = master[b].shift(-int(lag))
            df_pair = pd.concat([master[a], b_shifted], axis=1).dropna()
            if df_pair.shape[0] < 6:
                return out
            df_pair = df_pair.apply(pd.to_numeric, errors='coerce').dropna()
            if df_pair.shape[0] < 6:
                return out
            for method in methods:
                if method == 'pearson':
                    try:
                        r_val, p_val = pearsonr(df_pair.iloc[:,0], df_pair.iloc[:,1])
                    except:
                        r_val, p_val = np.nan, np.nan
                elif method == 'spearman':
                    try:
                        r_val, p_val = spearmanr(df_pair.iloc[:,0], df_pair.iloc[:,1], nan_policy='omit')
                    except:
                        r_val, p_val = np.nan, np.nan
                out.append({
                    'var_a': a,
                    'var_b': b,
                    'lag_hours': int(lag),
                    'method': method,
                    'r': float(r_val) if not pd.isna(r_val) else np.nan,
                    'p': float(p_val) if not pd.isna(p_val) else np.nan,
                    'n': int(df_pair.shape[0])
                })
        except Exception:
            pass
        return out

    all_tasks = [(a, b, lag) for i,a in enumerate(names) for j,b in enumerate(names) for lag in lags_hours if not (a==b and lag==0)]
    processed = Parallel(n_jobs=n_jobs, backend='loky')(delayed(process_pair)(a,b,lag) for a,b,lag in all_tasks)
    for sublist in processed:
        results.extend(sublist)
    return pd.DataFrame(results)

def load_cached_correlations(cache_path=CORR_CACHE_PATH):
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            return None
    return None

def save_cached_correlations(df, cache_path=CORR_CACHE_PATH):
    try:
        df.to_parquet(cache_path)
    except Exception:
        pass

# --- Variable categorization and scenario selection ---
CORRELATION_SCENARIOS = {
    "Symptoms & Sleep": {
        "groups": ("Symptoms", "Sleep"),
        "lags": [0, 1, 2, 3, 6, 12, 24],
        "allow_within_group": True,
    },
    "Symptoms & Weather": {
        "groups": ("Symptoms", "Weather"),
        "lags": [0, 1, 2, 3],
        "allow_within_group": False,
    },
    "Heart Rate & Symptoms": {
        "groups": ("Heart Rate", "Symptoms"),
        "lags": [0, 1, 2, 3, 6],
        "allow_within_group": False,
    },
    "All Variables": {
        "groups": ("Symptoms", "Sleep", "Weather", "Heart Rate", "Nutrition", "General", "Conditions"),
        "lags": [0, 1, 2, 3, 6, 12, 24],
        "allow_within_group": True,
    }
}

def categorize_var(name):
    if name.startswith('Symptom: ') or name == 'Symptom Total Score':
        return 'Symptoms'
    elif name.startswith('Condition: '):
        return 'Conditions'
    elif name.startswith(('temp_', 'humidity_', 'pressure_')):
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

# --- Correlation Page (main) ---
def correlation_page():
    st.header("Correlation Explorer")

    # Load your data dict (should be global or passed in by your app)
    from data_loader import data  # or however you load your dict of DataFrames

    # Detect global time range
    all_times = []
    for name, df_ in data.items():
        for col in df_.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    t = pd.to_datetime(df_[col], errors='coerce')
                    if not t.empty:
                        all_times.append(t.min())
                        all_times.append(t.max())
                except Exception:
                    pass
    all_times = [t for t in all_times if pd.notna(t)]
    if not all_times:
        st.write("No datetime data found in your sheets.")
        return
    start = min(all_times)
    end = max(all_times)
    st.write(f"Data range detected: {pd.to_datetime(start).date()} → {pd.to_datetime(end).date()}")

    # User selects date range
    col1, col2 = st.columns(2)
    with col1:
        user_start = st.date_input("Start date:", value=pd.to_datetime(start).date())
    with col2:
        user_end = st.date_input("End date:", value=pd.to_datetime(end).date())
    hourly_index = to_hourly_index(user_start, user_end)

    st.info("Building variables (this can take a moment)...")
    vars_dict = build_all_variables(data, hourly_index)
    st.write(f"Built {len(vars_dict)} variables.")

    # Categorize variables
    var_categories = {v: categorize_var(v) for v in vars_dict.keys()}

    # Select scenario
    scenario = st.selectbox("Select Correlation Scenario:", list(CORRELATION_SCENARIOS.keys()))
    sc = CORRELATION_SCENARIOS[scenario]
    selected_vars = [v for v in vars_dict if var_categories.get(v, 'Other') in sc['groups']]
    filtered_vars_dict = {k: vars_dict[k] for k in selected_vars}

    # Lags for this scenario
    lags_choice = st.multiselect("Select lags (hours):", sc['lags'], default=sc['lags'])

    min_abs_r = st.slider("Minimum absolute correlation to show:", 0.0, 1.0, 0.25, 0.01)
    p_threshold = st.number_input("Max p-value to show (NaN = ignore):", value=0.05, format="%.3f")

    # ---- Correlation computation (cache per scenario, lag and variable set) ----
    cache_key = f"{scenario}_{user_start}_{user_end}_{','.join(map(str,lags_choice))}.parquet"
    cache_path = os.path.join("corr_cache", cache_key)
    os.makedirs("corr_cache", exist_ok=True)

    if os.path.exists(cache_path):
        all_corr = pd.read_parquet(cache_path)
    else:
        all_corr = compute_all_pairwise_correlations(filtered_vars_dict, lags_hours=lags_choice)
        all_corr.to_parquet(cache_path)

    # Remove self correlations at zero lag
    all_corr = all_corr[~((all_corr['var_a'] == all_corr['var_b']) & (all_corr['lag_hours'] == 0))]

    # Remove sleep-sleep correlations (normalize casing & spaces)
    sleep_vars = {
        "bedtime (sec)", "waketime (sec)", "time_in_bed (sec)",
        "sleep_duration (sec)", "sleep_quality", "sleep_efficiency"
    }
    sleep_vars = {v.lower().strip() for v in sleep_vars}
    all_corr = all_corr[~(
        all_corr['var_a'].str.strip().str.lower().isin(sleep_vars) &
        all_corr['var_b'].str.strip().str.lower().isin(sleep_vars)
    )]

    # Filter out within-group pairs if scenario forbids it
    if not sc.get('allow_within_group', True):
        def same_group(row):
            return var_categories.get(row['var_a']) == var_categories.get(row['var_b'])
        all_corr = all_corr[~all_corr.apply(same_group, axis=1)]

    # Filter correlations by sample size, p-value, and correlation magnitude
    all_corr = all_corr[all_corr['n'] >= 6]
    if p_threshold is not None:
        all_corr = all_corr[(all_corr['p'].isna()) | (all_corr['p'] <= p_threshold)]
    all_corr = all_corr[all_corr['r'].abs() >= min_abs_r]

    st.write(f"{len(all_corr)} correlations matching filters.")

    # --- Display ---
    corr_df = all_corr.rename(columns={
        'var_a': 'Variable X',
        'var_b': 'Variable Y',
        'r': 'Correlation',
        'p': 'P-Value',
        'n': 'N'
    })

    # Summaries
    def generate_summary(corr_df):
        summaries = []
        filtered_df = corr_df[(corr_df['P-Value'] <= 0.05) & (corr_df['N'] >= 20)]
        grouped = filtered_df.groupby(['Variable X', 'Variable Y'])
        for (var1, var2), group in grouped:
            best_row = group.loc[group['P-Value'].idxmin()]
            corr = best_row['Correlation']
            p_value = best_row['P-Value']
            n = best_row['N']
            best_lag = best_row.get('lag_hours', 0)
            lag_min = int(group['lag_hours'].min())
            lag_max = int(group['lag_hours'].max())
            direction = "increase" if corr > 0 else "decrease"
            lag_str = f" {best_lag} hours later" if best_lag != 0 else ""
            range_str = f" observed across {lag_min}–{lag_max} hours" if lag_min != lag_max else ""
            summaries.append(
                f"An increase in {var1} is associated with a {direction} in {var2} "
                f"by {abs(corr) * 100:.1f}%{lag_str} (most significant, p = {p_value:.3f}, n = {n}){range_str}."
            )
        return summaries

    summary_list = generate_summary(corr_df)
    if summary_list:
        st.subheader("Summary of Likely Real Correlations")
        for item in summary_list:
            st.write(f"- {item}")
    else:
        st.info("No strong statistically significant correlations found.")

    # --- Visualization ---
    view = st.radio("View as:", ['Heatmap (single lag)','Top correlations table'], index=1)
    if view.startswith('Heatmap'):
        chosen_lag = st.selectbox("Choose lag (hours) to display:", sorted(all_corr['lag_hours'].unique()))
        chosen_method = st.selectbox("Choose method:", ['pearson','spearman'])
        subset = all_corr[(all_corr['lag_hours']==chosen_lag) & (all_corr['method']==chosen_method)]
        if subset.empty:
            st.write("No results for this lag/method with current filters.")
            return
        names = sorted(set(subset['var_a']).union(set(subset['var_b'])))
        mat = pd.DataFrame(np.nan, index=names, columns=names)
        for _, row in subset.iterrows():
            mat.loc[row['var_a'], row['var_b']] = row['r']
        fig, ax = plt.subplots(figsize=(12, max(6, len(names)*0.25)))
        sns.heatmap(mat.astype(float), cmap='coolwarm', center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
        ax.set_title(f"Correlation Matrix (lag={chosen_lag}h, method={chosen_method})")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        top_n = st.number_input("Show top N results (by |r|):", min_value=10, max_value=1000, value=100, step=10)
        show_only_real = st.checkbox(
            "Show only likely real correlations (p ≤ 0.05 and n ≥ 20)", 
            value=True
        )
        df_show = all_corr.copy()
        if show_only_real:
            df_show = df_show[(df_show['p'] <= 0.05) & (df_show['n'] >= 20)]
        df_show['abs_r'] = df_show['r'].abs()
        df_show = df_show.sort_values('abs_r', ascending=False).head(int(top_n))
        st.dataframe(df_show[['var_a','var_b','lag_hours','method','r','p','n']])
        idx = st.number_input("Inspect row index (0..n-1):", min_value=0, max_value=max(0, len(df_show)-1), value=0, step=1)
        if len(df_show) > 0:
            row = df_show.iloc[int(idx)]
            st.write("Selected:", row.to_dict())
            a = filtered_vars_dict[row['var_a']]
            b = filtered_vars_dict[row['var_b']].shift(-int(row['lag_hours']))
            plot_df = pd.concat([a, b], axis=1).dropna()
            if plot_df.shape[0] < 6:
                st.write("Not enough points to plot.")
            else:
                x = plot_df.iloc[:,0]
                y = plot_df.iloc[:,1]
                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(x, y, alpha=0.5)
                ax.set_xlabel(row['var_a'])
                ax.set_ylabel(f"{row['var_b']} (shifted by {int(row['lag_hours'])}h)")
                X = sm.add_constant(x)
                model = sm.OLS(y, X, missing='drop').fit()
                intercept, slope = model.params.iloc[0], model.params.iloc[1]
                xs = np.linspace(np.nanpercentile(x,1), np.nanpercentile(x,99), 100)
                ax.plot(xs, intercept + slope * xs, color='red', linewidth=2)
                ax.set_title(f"r={row['r']:.3f}, p={row['p']:.3f}, n={row['n']}")
                st.pyplot(fig)
    st.success("Correlation computation & display complete.")
