# correlations.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

from utils import (
    to_datetime_if_possible,
    severity_to_numeric,
    safe_daily_date_from_col,
    maybe_numeric,
)

def compute_daily_aggregates(sheets):
    """
    Convert the dictionary of raw sheets to a single daily-aggregated DataFrame.
    Returns: daily_df (index = date (datetime.date), columns = aggregated features)
    Aggregations included (examples):
      - HR: HR_avg, HR_max, tachy_percent (daily)
      - Sleep: duration, score, rem_time, deep_time, awake_time
      - Weather: temp_avg, humidity_avg, pressure_avg, precipitation_total
      - Activity: steps total, walking minutes, standing minutes, stairs count
      - Symptoms count / severity average
      - Meds: count of meds taken that day (optionally by med name)
      - Nutrition: counts of sugar/spice, meals_by_amount (amount_num sum)
    Adjust aggregations to your needs.
    """
    # container for per-day features
    features = {}

    # Helper to ensure date index
    def df_with_date_index(df, date_col=None):
        df = df.copy()
        df = to_datetime_if_possible(df)
        if date_col is None:
            # try 'date' or any datetime-like
            date_col = None
            for c in df.columns:
                if c.lower() == 'date':
                    date_col = c
                    break
            if date_col is None:
                for c in df.columns:
                    if 'time' in c.lower() or 'date' in c.lower():
                        date_col = c
                        break
        if date_col is None:
            df['__date'] = pd.NaT
            return df.set_index('__date')
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['_day'] = df[date_col].dt.date
        return df.set_index('_day')

    # HR Stats
    if 'HR Stats' in sheets:
        hr = sheets['HR Stats'].copy()
        hr = to_datetime_if_possible(hr)
        # assume 'date' column (date only)
        if 'date' in hr.columns:
            hr['date'] = pd.to_datetime(hr['date'], errors='coerce').dt.date
        else:
            hr['date'] = safe_daily_date_from_col(hr)
        hr_indexed = hr.set_index('date')
        features['HR_avg'] = hr_indexed['HR_avg']
        features['HR_max'] = hr_indexed['HR_max']
        features['tachy_percent'] = hr_indexed.get('tachy_percent')

    # Sleep Stats
    if 'Sleep Stats' in sheets:
        s = sheets['Sleep Stats'].copy()
        s = to_datetime_if_possible(s)
        if 'date' in s.columns:
            s['date'] = pd.to_datetime(s['date'], errors='coerce').dt.date
        else:
            s['date'] = safe_daily_date_from_col(s)
        s_idx = s.set_index('date')
        for c in ['duration', 'score', 'rem_time', 'deep_time', 'awake_time']:
            if c in s_idx.columns:
                features[f"sleep_{c}"] = s_idx[c]

    # Weather Stats
    if 'Weather Stats' in sheets:
        w = sheets['Weather Stats'].copy()
        w['date'] = pd.to_datetime(w['date'], errors='coerce').dt.date
        w_idx = w.set_index('date')
        for c in ['temp_avg', 'humidity_avg', 'pressure_avg', 'precipitation_total']:
            if c in w_idx.columns:
                features[f"weather_{c}"] = w_idx[c]

    # Walking (aggregate steps)
    if 'Walking' in sheets:
        walk = sheets['Walking'].copy()
        walk = to_datetime_if_possible(walk)
        # ensure 'Date' or date column
        if 'Date' in walk.columns:
            walk['Date'] = pd.to_datetime(walk['Date'], errors='coerce').dt.date
        elif 'date' in walk.columns:
            walk['date'] = pd.to_datetime(walk['date'], errors='coerce').dt.date
            walk['Date'] = walk['date']
        else:
            walk['Date'] = safe_daily_date_from_col(walk)
        # steps total
        if 'Steps' in walk.columns:
            steps_by_day = walk.groupby('Date')['Steps'].sum()
            features['steps_total'] = steps_by_day

    # Standing
    if 'Standing' in sheets:
        st = sheets['Standing'].copy()
        st = to_datetime_if_possible(st)
        if 'Date' in st.columns:
            st['Date'] = pd.to_datetime(st['Date'], errors='coerce').dt.date
        elif 'date' in st.columns:
            st['Date'] = pd.to_datetime(st['date'], errors='coerce').dt.date
        else:
            st['Date'] = safe_daily_date_from_col(st)
        if 'Duration' in st.columns:
            standing_by_day = st.groupby('Date')['Duration'].sum()
            features['standing_minutes'] = standing_by_day

    # Stairs
    if 'Stairs' in sheets:
        st = sheets['Stairs'].copy()
        st = to_datetime_if_possible(st)
        date_col = 'Date' if 'Date' in st.columns else None
        st['Date'] = pd.to_datetime(st[date_col], errors='coerce').dt.date if date_col else safe_daily_date_from_col(st)
        if 'Quantity' in st.columns:
            stairs_by_day = st.groupby('Date')['Quantity'].sum()
            features['stairs_count'] = stairs_by_day

    # Symptoms: count and avg severity
    if 'Symptoms' in sheets:
        sym = sheets['Symptoms'].copy()
        sym = to_datetime_if_possible(sym)
        # convert time -> date
        time_col = None
        for c in sym.columns:
            if 'time' in c.lower():
                time_col = c
                break
        if time_col:
            sym['_date'] = pd.to_datetime(sym[time_col], errors='coerce').dt.date
        else:
            sym['_date'] = safe_daily_date_from_col(sym)
        # severity mapping
        if 'severity' in sym.columns:
            sym['severity_num'] = severity_to_numeric(sym['severity'])
            severity_mean = sym.groupby('_date')['severity_num'].mean()
            features['symptom_severity_mean'] = severity_mean
        symptom_count = sym.groupby('_date').size()
        features['symptom_count'] = symptom_count

    # Conditions (counts per day)
    if 'Conditions' in sheets:
        cond = sheets['Conditions'].copy()
        cond = to_datetime_if_possible(cond)
        time_col = None
        for c in cond.columns:
            if 'time' in c.lower():
                time_col = c
                break
        if time_col:
            cond['_date'] = pd.to_datetime(cond[time_col], errors='coerce').dt.date
        else:
            cond['_date'] = safe_daily_date_from_col(cond)
        cond_count = cond.groupby('_date').size()
        features['conditions_count'] = cond_count

    # Nutrition general (counts of items like sugar/spice)
    if 'Nutrition - General' in sheets:
        nut = sheets['Nutrition - General'].copy()
        nut = to_datetime_if_possible(nut)
        time_col = None
        for c in nut.columns:
            if 'time' in c.lower():
                time_col = c
                break
        if time_col:
            nut['_date'] = pd.to_datetime(nut[time_col], errors='coerce').dt.date
        else:
            nut['_date'] = safe_daily_date_from_col(nut)
        # count by item
        item_counts = nut.groupby(['_date', 'item']).size().unstack(fill_value=0)
        # flatten columns as nutrition_item::name
        for col in item_counts.columns:
            features[f"nutrition_{col}"] = item_counts[col]

    # Nutrition - Meals (use amount_num)
    if 'Nutrition - Meals' in sheets:
        nm = sheets['Nutrition - Meals'].copy()
        nm = to_datetime_if_possible(nm)
        if 'date' in nm.columns:
            nm['date'] = pd.to_datetime(nm['date'], errors='coerce').dt.date
        elif 'time' in nm.columns:
            nm['date'] = pd.to_datetime(nm['time'], errors='coerce').dt.date
        else:
            nm['date'] = safe_daily_date_from_col(nm)
        if 'amount_num' in nm.columns:
            meals_by_day = nm.groupby('date')['amount_num'].sum()
            features['meals_amount_sum'] = meals_by_day

    # Meds: count meds per day, optionally by medication name
    if 'Meds' in sheets:
        meds = sheets['Meds'].copy()
        meds = to_datetime_if_possible(meds)
        date_col = 'date' if 'date' in meds.columns else None
        if date_col:
            meds['date'] = pd.to_datetime(meds['date'], errors='coerce').dt.date
        elif 'time taken' in meds.columns:
            meds['date'] = pd.to_datetime(meds['time taken'], errors='coerce').dt.date
        else:
            meds['date'] = safe_daily_date_from_col(meds)
        meds_count = meds.groupby('date').size()
        features['meds_count'] = meds_count
        # meds by name
        if 'medication' in meds.columns:
            meds_by_name = meds.groupby(['date', 'medication']).size().unstack(fill_value=0)
            for col in meds_by_name.columns:
                features[f"med_{col}"] = meds_by_name[col]

    # Consolidate features into a single DataFrame
    # Create DataFrame from all series and outer-join on date index
    df_list = []
    for k, ser in features.items():
        series = ser.copy()
        # ensure index is datetime.date
        if isinstance(series.index, pd.DatetimeIndex):
            idx = series.index.date
        else:
            idx = series.index
        s = pd.Series(series.values, index=pd.to_datetime(idx), name=k)
        df_list.append(s)
    if not df_list:
        return pd.DataFrame()
    daily_df = pd.concat(df_list, axis=1)
    # Sort and keep daily frequency index
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df = daily_df.sort_index()
    return daily_df

def compute_pairwise_correlations(df, columns=None, method='pearson', min_periods=3):
    """
    Compute correlation matrix (Pearson or Spearman) and p-values matrix.
    Returns (corr_df, pval_df).
    """
    if columns is None:
        columns = df.columns.tolist()
    data = df[columns].copy()
    corr = data.corr(method='pearson' if method == 'pearson' else None)
    # compute pairwise with stats (if possible)
    pvals = pd.DataFrame(index=columns, columns=columns, data=np.nan)
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            a = data[columns[i]]
            b = data[columns[j]]
            valid = a.notna() & b.notna()
            if valid.sum() >= min_periods:
                try:
                    if method == 'pearson':
                        r, p = pearsonr(a[valid], b[valid])
                    else:
                        r, p = spearmanr(a[valid], b[valid])
                    corr.loc[columns[i], columns[j]] = r
                    corr.loc[columns[j], columns[i]] = r
                    pvals.loc[columns[i], columns[j]] = p
                    pvals.loc[columns[j], columns[i]] = p
                except Exception:
                    corr.loc[columns[i], columns[j]] = np.nan
                    pvals.loc[columns[i], columns[j]] = np.nan
            else:
                corr.loc[columns[i], columns[j]] = np.nan
                corr.loc[columns[j], columns[i]] = np.nan
                pvals.loc[columns[i], columns[j]] = np.nan
                pvals.loc[columns[j], columns[i]] = np.nan
    return corr.astype(float), pvals.astype(float)

def compute_lagged_correlations(series_x, series_y, max_lag=7, freq='D', method='pearson', min_periods=3):
    """
    Compute correlation between series_x and series_y for lags in [-max_lag..+max_lag].
    If freq == 'D', lag units are days; if 'H', hours.
    Returns DataFrame with columns ['lag', 'corr', 'pval', 'n'].
    Positive lag means y is shifted forward (i.e., x at day t correlates with y at day t+lag).
    """
    results = []
    # ensure same datetime index type
    x = series_x.sort_index().copy()
    y = series_y.sort_index().copy()
    for lag in range(-max_lag, max_lag + 1):
        if freq == 'D':
            y_shifted = y.shift(periods=lag, freq='D') if isinstance(y.index, pd.DatetimeIndex) else y.shift(lag)
        elif freq == 'H':
            y_shifted = y.shift(periods=lag, freq='H') if isinstance(y.index, pd.DatetimeIndex) else y.shift(lag)
        else:
            y_shifted = y.shift(lag)
        merged = pd.concat([x, y_shifted], axis=1).dropna()
        n = len(merged)
        if n >= min_periods:
            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]
            try:
                if method == 'pearson':
                    r, p = pearsonr(a, b)
                else:
                    r, p = spearmanr(a, b)
            except Exception:
                r, p = np.nan, np.nan
        else:
            r, p = np.nan, np.nan
        results.append({'lag': lag, 'corr': r, 'pval': p, 'n': n})
    return pd.DataFrame(results)
