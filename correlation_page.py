import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from load_excel import load_excel_from_dropbox  # Dropbox loader

# --- Config ---
CORR_CACHE_DIR = "corr_cache"  # folder to store cached correlations
HOUR_LAGS = [0,1,2,3,4,5,6,8,12] + [24*d for d in range(1,8)]  # 0-7 days

# --- Helper Functions ---
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
        # Convert to numeric, coerce errors
        snapped = pd.to_numeric(snapped, errors='coerce')
        snapped = snapped.groupby(snapped.index).mean()
        snapped = snapped.reindex(hourly_index, fill_value=np.nan)
        out.update(snapped)
    elif method == 'ffill':
        out = out.ffill()
    return out

def build_all_variables(data, hourly_index):
    # example: convert your data dict into hourly variables
    vars_dict = {}
    for sheet_name, df in data.items():
        for col in df.columns:
            if col.lower() in ("time","date"):
                continue
            series = df[col]
            vars_dict[f"{sheet_name}: {col}"] = point_series_to_hourly(series, hourly_index)
    return vars_dict

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
    if not results:
        # Return empty dataframe with expected schema
        return pd.DataFrame(columns=['var_a','var_b','lag_hours','method','r','p','n'])
    else:
        return pd.DataFrame(results)

# --- Variable categorization ---
CORRELATION_SCENARIOS = {
    "Symptoms & Sleep": {"groups": ("Symptoms","Sleep"), "lags":[0,1,2,3,6,12,24], "allow_within_group": True},
    "Symptoms & Weather": {"groups": ("Symptoms","Weather"), "lags":[0,1,2,3], "allow_within_group": False},
    "Heart Rate & Symptoms": {"groups": ("Heart Rate","Symptoms"), "lags":[0,1,2,3,6], "allow_within_group": False},
    "All Variables": {"groups": ("Symptoms","Sleep","Weather","Heart Rate","Nutrition","General","Conditions"), "lags":[0,1,2,3,6,12,24], "allow_within_group": True}
}

def categorize_var(name):
    if name.startswith('Symptom: ') or name=='Symptom Total Score': return 'Symptoms'
    elif name.startswith('Condition: '): return 'Conditions'
    elif name.startswith(('temp_','humidity_','pressure_')): return 'Weather'
    elif name.startswith(('Sleep ','Wake ','Bedtime')): return 'Sleep'
    elif name.startswith(('Heart Rate','Tachycardia','HRV')): return 'Heart Rate'
    elif name.startswith(('Nutrition Item:','Meal Amount','Water ')): return 'Nutrition'
    elif name.startswith(('Stairs','Standing','Walking','Steps')): return 'General'
    else: return 'Other'

# --- Summary generation ---
def generate_summary(corr_df):
    summaries = []
    filtered_df = corr_df[(corr_df['p'] <= 0.05) & (corr_df['n'] >= 20)]
    grouped = filtered_df.groupby(['var_a','var_b'])
    for (var1,var2), group in grouped:
        best_row = group.loc[group['p'].idxmin()]
        corr = best_row['r']
        p_value = best_row['p']
        n = best_row['n']
        best_lag = best_row.get('lag_hours',0)
        lag_min = int(group['lag_hours'].min())
        lag_max = int(group['lag_hours'].max())
        direction = "increase" if corr>0 else "decrease"
        lag_str = f" {best_lag} hours later" if best_lag!=0 else ""
        range_str = f" observed across {lag_min}–{lag_max} hours" if lag_min!=lag_max else ""
        summaries.append(
            f"An increase in {var1} is associated with a {direction} in {var2} "
            f"by {abs(corr)*100:.1f}%{lag_str} (most significant, p={p_value:.3f}, n={n}){range_str}."
        )
    return summaries

# --- Main Correlation Page ---
def correlation_page(dbx):
    st.header("Correlation Explorer")

    # Load Excel from Dropbox
    data = load_excel_from_dropbox(dbx)
    if not data:
        st.error("No data loaded from Dropbox.")
        return

    # Detect global time range
    all_times = []
    for df_ in data.values():
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

    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        user_start = st.date_input("Start date:", value=pd.to_datetime(start).date())
    with col2:
        user_end = st.date_input("End date:", value=pd.to_datetime(end).date())
    hourly_index = to_hourly_index(user_start, user_end)

    st.info("Building variables...")
    vars_dict = build_all_variables(data, hourly_index)
    st.write(f"Built {len(vars_dict)} variables.")

    var_categories = {v: categorize_var(v) for v in vars_dict.keys()}

    # Scenario selection
    scenario = st.selectbox("Select Correlation Scenario:", list(CORRELATION_SCENARIOS.keys()))
    sc = CORRELATION_SCENARIOS[scenario]
    selected_vars = [v for v in vars_dict if var_categories.get(v,'Other') in sc['groups']]
    filtered_vars_dict = {k: vars_dict[k] for k in selected_vars}

    lags_choice = st.multiselect("Select lags (hours):", sc['lags'], default=sc['lags'])
    min_abs_r = st.slider("Minimum absolute correlation to show:", 0.0, 1.0, 0.25, 0.01)
    p_threshold = st.number_input("Max p-value to show (NaN = ignore):", value=0.05, format="%.3f")

    # ---- Correlation computation (always fresh, no cache) ----
    st.info("Computing correlations fresh (no cache)...")
    all_corr = compute_all_pairwise_correlations(filtered_vars_dict, lags_hours=lags_choice)
    st.info(f"Computed correlations ({len(all_corr)} rows)")

    st.write("DEBUG: correlation result shape =", all_corr.shape)
    st.write(all_corr.head())

    # Check expected columns exist
    expected_cols = ['var_a','var_b','lag_hours','method','r','p','n']
    missing = [c for c in expected_cols if c not in all_corr.columns]
    if missing:
        st.error(f"Missing expected columns in correlation DataFrame: {missing}")
        st.write(all_corr.head())
        return
    
    # Handle completely empty correlation set
    if all_corr.empty:
        st.warning("No correlations found for this variable set and lag range.")
        return
    
    # --- Remove self correlations at zero lag ---
    all_corr = all_corr[~((all_corr['var_a'] == all_corr['var_b']) & (all_corr['lag_hours'] == 0))]
    
    # --- Remove sleep-sleep correlations ---
    sleep_vars = {"bedtime (sec)", "waketime (sec)", "time_in_bed (sec)",
                  "sleep_duration (sec)", "sleep_quality", "sleep_efficiency"}
    sleep_vars = {v.lower().strip() for v in sleep_vars}
    all_corr = all_corr[~(
        all_corr['var_a'].str.strip().str.lower().isin(sleep_vars) &
        all_corr['var_b'].str.strip().str.lower().isin(sleep_vars)
    )]
    
    # --- Filter within-group if scenario forbids it ---
    if not sc.get('allow_within_group', True):
        def same_group(row):
            return var_categories.get(row['var_a']) == var_categories.get(row['var_b'])
        all_corr = all_corr[~all_corr.apply(same_group, axis=1)]
    
    # --- Filter correlations by sample size, p-value, magnitude ---
    all_corr = all_corr[all_corr['n'] >= 6]
    if p_threshold is not None:
        all_corr = all_corr[(all_corr['p'].isna()) | (all_corr['p'] <= p_threshold)]
    all_corr = all_corr[all_corr['r'].abs() >= min_abs_r]
    
    if all_corr.empty:
        st.warning("No correlations match your filters (sample size, p-value, minimum r).")
        return
    
    st.write(f"{len(all_corr)} correlations matching filters.")

    # Summary
    summary_list = generate_summary(all_corr)
    if summary_list:
        st.subheader("Summary of Likely Real Correlations")
        for item in summary_list:
            st.write(f"- {item}")
    else:
        st.info("No strong statistically significant correlations found.")

    # Display top table
    top_n = st.number_input("Show top N results (by |r|):", min_value=10, max_value=1000, value=100, step=10)
    df_show = all_corr.copy()
    df_show['abs_r'] = df_show['r'].abs()
    df_show = df_show.sort_values('abs_r', ascending=False).head(int(top_n))
    st.dataframe(df_show[['var_a','var_b','lag_hours','method','r','p','n']])

    # Scatter/OLS plot
    if len(df_show)>0:
        idx = st.number_input("Inspect row index (0..n-1):", min_value=0, max_value=max(0,len(df_show)-1), value=0, step=1)
        row = df_show.iloc[int(idx)]
        a = filtered_vars_dict[row['var_a']]
        b = filtered_vars_dict[row['var_b']].shift(-int(row['lag_hours']))
        plot_df = pd.concat([a,b], axis=1).dropna()
        if plot_df.shape[0] < 6:
            st.write("Not enough points to plot.")
        else:
            x = plot_df.iloc[:,0]
            y = plot_df.iloc[:,1]
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(x,y,alpha=0.5)
            ax.set_xlabel(row['var_a'])
            ax.set_ylabel(f"{row['var_b']} (shifted by {int(row['lag_hours'])}h)")
            X = sm.add_constant(x)
            model = sm.OLS(y,X,missing='drop').fit()
            intercept, slope = model.params.iloc[0], model.params.iloc[1]
            xs = np.linspace(np.nanpercentile(x,1), np.nanpercentile(x,99), 100)
            ax.plot(xs, intercept + slope*xs, color='red', linewidth=2)
            ax.set_title(f"r={row['r']:.3f}, p={row['p']:.3f}, n={row['n']}")
            st.pyplot(fig)

    st.success("Correlation computation & display complete.")
