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

# ----------------------------
# Helper: valid dates/files
# ----------------------------
def get_valid_dates(validated_keys_df, category_col):
    """
    Returns a set of dates (datetime.date) for which the category is valid (TRUE).
    """
    if category_col not in validated_keys_df.columns:
        return set()  # nothing is valid if column missing
    df = validated_keys_df[validated_keys_df[category_col].astype(str).str.upper() == 'TRUE'].copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        return set(df['date'].dropna())
    return set()

VARIABLE_A_GROUPS = [
    "Conditions", "Stairs", "Walking", "Standing", "Meds", 
    "Nutrition", "Liquids", "Meals", "HR", "Weather", "Sleep"
]

VARIABLE_B_GROUPS = [
    "Symptoms", "HR", "Digestion", "Sleep"
]

def categorize_columns_for_cross_group(columns):
    """
    Split columns into independent (predictors) and dependent (outcomes).
    Some categories (like digestion) can belong to both.
    """
    independent = []
    dependent = []

    for col in columns:
        lc = col.lower()

        # Independent categories
        if lc.startswith("med_"):
            independent.append(col)
        elif lc.startswith(("nutrition_", "meal_", "liquid_")):
            independent.append(col)
        elif lc.startswith(("weather_", "hourlyweather_")):
            independent.append(col)
        elif lc.startswith(("standing_", "walking_", "stairs_", "location_", "steps_")):
            independent.append(col)
        elif lc.startswith("condition_"):
            independent.append(col)
        elif lc.startswith("digestion_"):
            independent.append(col)  # digestion in both

        # Dependent categories
        if lc.startswith("symptom_"):  # each symptom separately
            dependent.append(col)
        elif lc.startswith("sleep_"):
            dependent.append(col)
        elif lc.startswith(("heartrate_", "hrv_")):
            dependent.append(col)
        elif lc.startswith("digestion_"):
            dependent.append(col)  # digestion in both

    return independent, dependent

def compute_daily_aggregates(sheets):
    """
    Convert the dictionary of raw sheets to a single daily-aggregated DataFrame.
    Returns: daily_df (index = date (datetime.date), columns = aggregated features)
    """
    # container for per-day features
    features = {}

    # Load Validated Keys
    validated_keys = sheets.get('Validated Keys', pd.DataFrame())
    nutrition_valid_dates = get_valid_dates(validated_keys, 'nutrition_valid')
    conditions_valid_dates = get_valid_dates(validated_keys, 'conditions_valid')
    symptoms_valid_dates = get_valid_dates(validated_keys, 'symptoms_valid')
    digestion_valid_dates = get_valid_dates(validated_keys, 'digestion_valid')
    meds_valid_dates = get_valid_dates(validated_keys, 'med_valid')
    reproductive_valid_dates = get_valid_dates(validated_keys, 'reproductive_valid')

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

    # HR Stats (daily, but guard against duplicates)
    if 'HR Stats' in sheets:
        hr = sheets['HR Stats'].copy()
        hr = to_datetime_if_possible(hr)

        if 'date' in hr.columns:
            hr['date'] = pd.to_datetime(hr['date'], errors='coerce').dt.date
        else:
            hr['date'] = safe_daily_date_from_col(hr)

        # Define which stats we care about
        agg_map = {}
        if 'HR_avg' in hr.columns:
            agg_map['HR_avg'] = 'mean'
        if 'HR_max' in hr.columns:
            agg_map['HR_max'] = 'max'
        if 'HR_min' in hr.columns:
            agg_map['HR_min'] = 'min'
        if 'HRV' in hr.columns:
            agg_map['HRV'] = 'mean'
        if 'tachy_percent' in hr.columns:
            agg_map['tachy_percent'] = 'mean'

        # Aggregate by day (safe against duplicates)
        if agg_map:
            hr_daily = hr.groupby('date').agg(agg_map)

            # Store results into features dict
            for col in hr_daily.columns:
                features[col] = hr_daily[col]

    if 'Sleep Stats' in sheets:
        s = sheets['Sleep Stats'].copy()
        s = to_datetime_if_possible(s)
    
        # Ensure date column exists
        if 'date' in s.columns:
            s['date'] = pd.to_datetime(s['date'], errors='coerce').dt.date
        else:
            s['date'] = safe_daily_date_from_col(s)
    
        # Drop duplicate dates just in case
        s = s.drop_duplicates(subset=['date'], keep='last').set_index('date')
    
        # Bedtime/waketime as datetime
        s['bedtime'] = pd.to_datetime(s['bedtime'], errors='coerce')
        s['waketime'] = pd.to_datetime(s['waketime'], errors='coerce')
    
        # Convert times to numeric hours since midnight
        def to_hours(t):
            return t.hour + t.minute / 60 if pd.notnull(t) else np.nan
    
        s['bedtime_hours'] = s['bedtime'].apply(to_hours)
        s['waketime_hours'] = s['waketime'].apply(to_hours)
    
        # 7-day rolling average for bedtime/waketime
        s['bedtime_avg7'] = s['bedtime_hours'].rolling(7, min_periods=3).mean()
        s['waketime_avg7'] = s['waketime_hours'].rolling(7, min_periods=3).mean()
    
        # Difference from rolling average
        s['bedtime_diff'] = s['bedtime_hours'] - s['bedtime_avg7']
        s['waketime_diff'] = s['waketime_hours'] - s['waketime_avg7']
    
        # Convert duration to numeric hours if needed
        if 'duration' in s.columns:
            if np.issubdtype(s['duration'].dtype, np.timedelta64):
                s['duration_hours'] = s['duration'].dt.total_seconds() / 3600
            else:
                s['duration_hours'] = s['duration']
        else:
            s['duration_hours'] = np.nan
    
        # Compute sleep stage percentages
        stage_cols = ['rem_time', 'core_time', 'deep_time', 'awake_time']
        for col in stage_cols:
            if col in s.columns:
                # Convert timedelta to hours if needed
                if np.issubdtype(s[col].dtype, np.timedelta64):
                    s[col] = s[col].dt.total_seconds() / 3600
                elif np.issubdtype(s[col].dtype, np.datetime64):
                    # Convert datetime to hours since midnight (fallback)
                    s[col] = s[col].dt.hour + s[col].dt.minute / 60 + s[col].dt.second / 3600
                # Percentage relative to total duration
                s[f'{col}_pct'] = s[col] / s['duration_hours'].replace({0: np.nan})
    
        # Store features in the main dictionary
        features['sleep_duration'] = s['duration_hours']
        features['sleep_score'] = s['score'] if 'score' in s.columns else np.nan
        for col in stage_cols:
            if col in s.columns:
                features[col] = s[col]
                features[f'{col}_pct'] = s[f'{col}_pct']
        features['bedtime_diff'] = s['bedtime_diff']
        features['waketime_diff'] = s['waketime_diff']

    # Weather Stats (all columns + rolling 7-day averages)
    if 'Weather Stats' in sheets:
        w = sheets['Weather Stats'].copy()
        w['date'] = pd.to_datetime(w['date'], errors='coerce').dt.date
    
        weather_cols = [
            'temp_high', 'temp_low', 'temp_avg', 
            'humidity_avg', 
            'pressure_avg', 'pressure_min', 'pressure_max',
            'precipitation_hours', 'precipitation_total'
        ]
    
        # Aggregate by day to guard against duplicates
        w_daily = w.groupby('date')[weather_cols].mean()
    
        # Add raw daily features
        for col in w_daily.columns:
            features[f"weather_{col}"] = w_daily[col]
    
        # Compute 7-day rolling averages (using past 6 days + today)
        w_rolling = w_daily.rolling(window=7, min_periods=1).mean()
    
        # Add deviation from 7-day rolling average as features
        for col in w_daily.columns:
            features[f"weather_{col}_dev_from_7day_avg"] = w_daily[col] - w_rolling[col]

    if 'Walking' in sheets:
        walk = sheets['Walking'].copy()
        walk = to_datetime_if_possible(walk)
        
        # Ensure we have a date column
        if 'Date' in walk.columns:
            walk['Date'] = pd.to_datetime(walk['Date'], errors='coerce').dt.date
        elif 'date' in walk.columns:
            walk['Date'] = pd.to_datetime(walk['date'], errors='coerce').dt.date
            walk['Date'] = walk['date']
        else:
            walk['Date'] = safe_daily_date_from_col(walk)
        
        # 1. Total steps per day
        if 'Steps' in walk.columns:
            features['steps_total'] = walk.groupby('Date')['Steps'].sum()
        
        # 2. Total walking time per day
        if 'status' in walk.columns and 'time' in walk.columns:
            # Pivot start/end into a duration
            walk['_time'] = pd.to_datetime(walk['time'], errors='coerce')
            # We'll compute durations per entry
            total_durations = []
            for date, group in walk.groupby('Date'):
                duration_sum = pd.Timedelta(0)
                items = group['item'].unique()
                for item in items:
                    sub = group[group['item'] == item].sort_values('_time')
                    # iterate in pairs of Start -> End
                    for i in range(0, len(sub) - 1, 2):
                        start = sub.iloc[i]
                        end = sub.iloc[i+1]
                        if start['status'].lower() == 'start' and end['status'].lower() == 'end':
                            duration_sum += (end['_time'] - start['_time'])
                total_durations.append((date, duration_sum.total_seconds() / 60))  # in minutes
            features['walking_minutes'] = pd.Series(dict(total_durations))
        
        # 3. Count of Long Walks / Brisk Walks
        for walk_type in ['🚶Long Walk', '🚶Brisk Walk']:
            if 'item' in walk.columns:
                count_by_day = walk[walk['item'] == walk_type].groupby('Date').size()
                features[f"{walk_type}_count"] = count_by_day

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

    # --- Symptoms ---
    if 'Symptoms' in sheets:
        sym = sheets['Symptoms'].copy()
        sym = to_datetime_if_possible(sym)
        time_col = None
        for c in sym.columns:
            if 'time' in c.lower():
                time_col = c
                break
        if time_col:
            sym['_date'] = pd.to_datetime(sym[time_col], errors='coerce').dt.date
        else:
            sym['_date'] = safe_daily_date_from_col(sym)
        # Filter by valid dates
        sym = sym[sym['_date'].isin(symptoms_valid_dates)]

        # Convert severity text/numeric → number
        if 'severity' in sym.columns:
            sym['severity_num'] = severity_to_numeric(sym['severity'])
        elif 'Severity' in sym.columns:
            sym['severity_num'] = severity_to_numeric(sym['Severity'])
        else:
            sym['severity_num'] = maybe_numeric(sym.iloc[:, 0])  # fallback if structure varies

        # Group by (date, symptom) → average severity
        if 'Symptom' in sym.columns:
            per_symptom = sym.groupby(['_date', 'Symptom'])['severity_num'].mean().unstack(fill_value=np.nan)

            # Total Symptom Score = sum across symptom averages
            total_score = per_symptom.sum(axis=1)
            features['Total_Symptom_Score'] = total_score

            # Add per-symptom averages as individual features
            for col in per_symptom.columns:
                features[f"{col}_severity_avg"] = per_symptom[col]

        else:
            # If no Symptom column, just compute global average as fallback
            severity_mean = sym.groupby('_date')['severity_num'].mean()
            features['symptom_severity_mean'] = severity_mean

    # --- Conditions ---
    if 'Conditions' in sheets:
        cond = sheets['Conditions'].copy()
        cond = to_datetime_if_possible(cond)
        if 'time' in cond.columns:
            cond['_date'] = pd.to_datetime(cond['time'], errors='coerce').dt.date
        else:
            cond['_date'] = safe_daily_date_from_col(cond)
        # Filter by valid dates
        cond = cond[cond['_date'].isin(conditions_valid_dates)]
    
        # Count per condition type (column 'item')
        if 'item' in cond.columns:
            cond_counts = cond.groupby(['_date', 'item']).size().unstack(fill_value=0)
    
            # Store each condition type as a separate feature
            for col in cond_counts.columns:
                features[f"condition_{col}"] = cond_counts[col]
    
            # Compute 7-day rolling mean for deviation
            cond_counts.index = pd.to_datetime(cond_counts.index)
            rolling_avg = cond_counts.rolling(window=7, min_periods=1).mean()
    
            # Calculate deviation: today's count - weekly average
            cond_deviation = cond_counts - rolling_avg
            for col in cond_counts.columns:
                features[f"condition_{col}_dev"] = cond_deviation[col]
        else:
            # fallback: total count per day
            total_counts = cond.groupby('_date').size()
            features['conditions_count'] = total_counts
            rolling_avg = total_counts.rolling(window=7, min_periods=1).mean()
            features['conditions_count_dev'] = total_counts - rolling_avg

    # Locations: total duration per item per day
    if 'Locations' in sheets:
        loc = sheets['Locations'].copy()
        loc = to_datetime_if_possible(loc)
        
        # Ensure datetime
        loc['time'] = pd.to_datetime(loc['time'], errors='coerce')
        loc = loc.sort_values(['item', 'time'])  # sort by item and time
        
        # Determine the date for each row
        loc['date'] = loc['time'].dt.date
        
        # Filter by valid dates (Conditions-valid)
        loc = loc[loc['date'].isin(conditions_valid_dates)]
        
        # Container for daily durations
        daily_durations = {}
        
        # Process each item type separately
        for item_name, df_item in loc.groupby('item'):
            total_per_day = {}
            stack = []  # keep track of unmatched 'Start' times
            
            for _, row in df_item.iterrows():
                if row['status'].lower() == 'start':
                    stack.append(row['time'])
                elif row['status'].lower() == 'end' and stack:
                    start_time = stack.pop(0)  # pair the earliest unmatched start
                    duration = (row['time'] - start_time).total_seconds() / 3600.0  # hours
                    day = start_time.date()
                    if day in total_per_day:
                        total_per_day[day] += duration
                    else:
                        total_per_day[day] = duration
            
            # Convert to pandas Series and store in features
            if total_per_day:
                s = pd.Series(total_per_day)
                s.index = pd.to_datetime(s.index)
                features[f"location_{item_name}_hours"] = s
    
    # Nutrition general (counts of items like sugar/spice)
    # --- Example for Nutrition - General ---
    if 'Nutrition - General' in sheets:
        nut = sheets['Nutrition - General'].copy()
        nut = to_datetime_if_possible(nut)
        date_col = None
        for c in nut.columns:
            if 'time' in c.lower():
                date_col = c
                break
        if date_col:
            nut['_date'] = pd.to_datetime(nut[date_col], errors='coerce').dt.date
        else:
            nut['_date'] = safe_daily_date_from_col(nut)
        # Filter by valid dates
        nut = nut[nut['_date'].isin(nutrition_valid_dates)]
        # Count by item
        item_counts = nut.groupby(['_date', 'item']).size().unstack(fill_value=0)
        for col in item_counts.columns:
            features[f"nutrition_{col}"] = item_counts[col]
            features[f"nutrition_{col}_7d_avg"] = item_counts[col].rolling(window=7, min_periods=1).mean()

    # --- Nutrition - Meals ---
    if 'Nutrition - Meals' in sheets and 'Sleep Stats' in sheets:
        nm = sheets['Nutrition - Meals'].copy()
        nm = to_datetime_if_possible(nm)
        if 'time' in nm.columns:
            nm['date'] = pd.to_datetime(nm['time'], errors='coerce').dt.date
            nm['meal_time'] = pd.to_datetime(nm['time'], errors='coerce')
        elif 'date' in nm.columns:
            nm['date'] = pd.to_datetime(nm['date'], errors='coerce').dt.date
            nm['meal_time'] = pd.to_datetime(nm['date'], errors='coerce')
        else:
            nm['date'] = safe_daily_date_from_col(nm)
            nm['meal_time'] = pd.to_datetime(nm.iloc[:,0], errors='coerce')
        # Filter by valid dates
        nm = nm[nm['date'].isin(nutrition_valid_dates)]
        
        # Total meal amount per day
        if 'amount_num' in nm.columns:
            meals_by_day = nm.groupby('date')['amount_num'].sum()
            features['meals_amount_sum'] = meals_by_day
    
        # Wake-relative features
        sleep_df = sheets['Sleep Stats'].copy()
        sleep_df = to_datetime_if_possible(sleep_df)
        sleep_df['date'] = pd.to_datetime(sleep_df['date'], errors='coerce').dt.date
        sleep_df['waketime'] = pd.to_datetime(sleep_df['waketime'], errors='coerce')
    
        # Merge waketime into meals
        nm = nm.merge(sleep_df[['date', 'waketime']], on='date', how='left')
        nm['hours_after_wake'] = (nm['meal_time'] - nm['waketime']).dt.total_seconds() / 3600
    
        # First meal after waking
        first_meal = nm.groupby('date')['hours_after_wake'].min()
        features['first_meal_after_wake_hours'] = first_meal
    
        # Meal amounts in time windows after wake
        def meal_amount_in_window(df, start_h, end_h):
            window = df[(df['hours_after_wake'] >= start_h) & (df['hours_after_wake'] < end_h)]
            return window.groupby('date')['amount_num'].sum()
    
        features['meal_0_1h_after_wake'] = meal_amount_in_window(nm, 0, 1)
        features['meal_1_3h_after_wake'] = meal_amount_in_window(nm, 1, 3)
        features['meal_3_6h_after_wake'] = meal_amount_in_window(nm, 3, 6)
    
        # 7-day rolling averages
        for col in ['first_meal_after_wake_hours', 'meal_0_1h_after_wake', 'meal_1_3h_after_wake', 'meal_3_6h_after_wake']:
            rolling_col = f"{col}_7day_avg"
            features[rolling_col] = features[col].rolling(7, min_periods=1).mean()

    # --- Nutrition - Liquids ---
    if 'Nutrition - Liquids' in sheets and 'Sleep Stats' in sheets:
        nl = sheets['Nutrition - Liquids'].copy()
        nl = to_datetime_if_possible(nl)
        if 'time' in nl.columns:
            nl['date'] = pd.to_datetime(nl['time'], errors='coerce').dt.date
            nl['liquid_time'] = pd.to_datetime(nl['time'], errors='coerce')
        elif 'date' in nl.columns:
            nl['date'] = pd.to_datetime(nl['date'], errors='coerce').dt.date
            nl['liquid_time'] = pd.to_datetime(nl['date'], errors='coerce')
        else:
            nl['date'] = safe_daily_date_from_col(nl)
            nl['liquid_time'] = pd.to_datetime(nl.iloc[:,0], errors='coerce')
        # Filter by valid dates
        nl = nl[nl['date'].isin(nutrition_valid_dates)]
    
        # Total liquid intake per day
        if 'amount_num' in nl.columns:
            liquids_by_day = nl.groupby('date')['amount_num'].sum()
            features['liquids_amount_sum'] = liquids_by_day
    
        # Wake-relative features
        sleep_df = sheets['Sleep Stats'].copy()
        sleep_df = to_datetime_if_possible(sleep_df)
        sleep_df['date'] = pd.to_datetime(sleep_df['date'], errors='coerce').dt.date
        sleep_df['waketime'] = pd.to_datetime(sleep_df['waketime'], errors='coerce')
    
        # Merge waketime into liquids
        nl = nl.merge(sleep_df[['date', 'waketime']], on='date', how='left')
        nl['hours_after_wake'] = (nl['liquid_time'] - nl['waketime']).dt.total_seconds() / 3600
    
        # First liquid after waking
        first_liquid = nl.groupby('date')['hours_after_wake'].min()
        features['first_liquid_after_wake_hours'] = first_liquid
    
        # Liquid amounts in time windows after wake
        def liquid_amount_in_window(df, start_h, end_h):
            window = df[(df['hours_after_wake'] >= start_h) & (df['hours_after_wake'] < end_h)]
            return window.groupby('date')['amount_num'].sum()
    
        features['liquid_0_1h_after_wake'] = liquid_amount_in_window(nl, 0, 1)
        features['liquid_1_3h_after_wake'] = liquid_amount_in_window(nl, 1, 3)
        features['liquid_3_6h_after_wake'] = liquid_amount_in_window(nl, 3, 6)
    
        # 7-day rolling averages
        for col in ['first_liquid_after_wake_hours', 'liquid_0_1h_after_wake', 'liquid_1_3h_after_wake', 'liquid_3_6h_after_wake']:
            rolling_col = f"{col}_7day_avg"
            features[rolling_col] = features[col].rolling(7, min_periods=1).mean()

    # --- Digestion ---
    if 'Digestion' in sheets:
        dig = sheets['Digestion'].copy()
        dig = to_datetime_if_possible(dig)
        if 'time' in dig.columns:
            dig['_date'] = pd.to_datetime(dig['time'], errors='coerce').dt.date
        else:
            dig['_date'] = safe_daily_date_from_col(dig)
        # Filter by valid dates
        dig = dig[dig['_date'].isin(digestion_valid_dates)]
        
        # Map Bristol stool descriptions to scores
        bristol_map = {
            "1: Separate hard lumps": 1,
            "2: Lumpy and sausage like": 2,
            "3: Cracked sausage": 3,
            "4: Smooth snake": 4,
            "5: Soft blobs": 5,
            "6: Mushy blobs": 6,
            "7: Liquid": 7
        }
        dig['bristol_score'] = dig['item'].map(bristol_map)
        
        # BM flag: 1 if any BM that day, else 0
        bm_flag = dig.groupby('_date').size().apply(lambda x: 1 if x > 0 else 0)
        features['BM_flag'] = bm_flag
        
        # BM count
        bm_count = dig.groupby('_date').size()
        features['BM_count'] = bm_count
        
        # BM average Bristol score
        bm_score_avg = dig.groupby('_date')['bristol_score'].mean()
        features['BM_score_avg'] = bm_score_avg
    
    # --- Meds ---
    if 'Meds' in sheets:
        meds = sheets['Meds'].copy()
        meds = to_datetime_if_possible(meds)
        if 'date' in meds.columns:
            meds['date'] = pd.to_datetime(meds['date'], errors='coerce').dt.date
        elif 'time taken' in meds.columns:
            meds['date'] = pd.to_datetime(meds['time taken'], errors='coerce').dt.date
        else:
            meds['date'] = safe_daily_date_from_col(meds)
        # Filter by valid dates
        meds = meds[meds['date'].isin(meds_valid_dates)]    
        # Total meds taken per day
        features['meds_count'] = meds.groupby('date').size()
    
        if 'medication' in meds.columns:
            meds['dose'] = maybe_numeric(meds['dose']) if 'dose' in meds.columns else 1
            meds['taken_flag'] = meds['status'].map(lambda x: 1 if str(x).lower() == 'taken' else 0)
    
            # Merge wake/sleep times if available
            if 'Sleep Stats' in sheets:
                sleep = sheets['Sleep Stats'].copy()
                sleep = to_datetime_if_possible(sleep)
                if 'date' in sleep.columns:
                    sleep['date'] = pd.to_datetime(sleep['date'], errors='coerce').dt.date
                else:
                    sleep['date'] = safe_daily_date_from_col(sleep)
                sleep_idx = sleep.set_index('date')
                wake_time_map = sleep_idx['waketime'].to_dict() if 'waketime' in sleep_idx.columns else {}
                bed_time_map = sleep_idx['bedtime'].to_dict() if 'bedtime' in sleep_idx.columns else {}
            else:
                wake_time_map = {}
                bed_time_map = {}
    
            # Container for per-med features
            med_features = {}
    
            for med in meds['medication'].unique():
                med_df = meds[meds['medication'] == med].copy()
    
                # Total dose
                total_dose = med_df.groupby('date')['dose'].sum()
                med_features[f"med_{med}_total_dose"] = total_dose
    
                # Taken/skipped flag (1 if any taken)
                taken_flag = med_df.groupby('date')['taken_flag'].max()
                med_features[f"med_{med}_taken_flag"] = taken_flag
    
                # Hours after waking
                hours_after_wake = []
                hours_before_bed = []
                for idx, row in med_df.iterrows():
                    dtime = pd.to_datetime(row['time taken']) if 'time taken' in row else pd.NaT
                    day = row['date']
                    wake = wake_time_map.get(day, pd.NaT)
                    bed = bed_time_map.get(day, pd.NaT)
    
                    if pd.notna(dtime) and pd.notna(wake):
                        hours_after_wake.append((dtime - pd.to_datetime(wake)).total_seconds() / 3600)
                    else:
                        hours_after_wake.append(np.nan)
    
                    if pd.notna(dtime) and pd.notna(bed):
                        hours_before_bed.append((pd.to_datetime(bed) - dtime).total_seconds() / 3600)
                    else:
                        hours_before_bed.append(np.nan)
    
                med_df['hours_after_wake'] = hours_after_wake
                med_df['hours_before_bed'] = hours_before_bed
    
                # Daily mean timing
                mean_after_wake = med_df.groupby('date')['hours_after_wake'].mean()
                mean_before_bed = med_df.groupby('date')['hours_before_bed'].mean()
                med_features[f"med_{med}_hours_after_wake_avg"] = mean_after_wake
                med_features[f"med_{med}_hours_before_bed_avg"] = mean_before_bed
    
                # Rolling averages (7-day)
                roll_after_wake = mean_after_wake.rolling(7, min_periods=1).mean()
                roll_before_bed = mean_before_bed.rolling(7, min_periods=1).mean()
                med_features[f"med_{med}_hours_after_wake_dev"] = mean_after_wake - roll_after_wake
                med_features[f"med_{med}_hours_before_bed_dev"] = mean_before_bed - roll_before_bed
    
            # Add all med features to main features dict
            for k, v in med_features.items():
                features[k] = v

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

def compute_pairwise_cross_group_matrix(df, method='pearson', min_periods=3):
    """
    Compute correlation matrix (Pearson or Spearman) and p-values matrix,
    but only for Variable A vs Variable B. Intra-group correlations are set to NaN.
    Returns: (corr_df, pval_df)
    """
    columns = df.columns.tolist()
    corr = pd.DataFrame(index=columns, columns=columns, data=np.nan)
    pvals = pd.DataFrame(index=columns, columns=columns, data=np.nan)

    # Categorize columns
    var_a_cols, var_b_cols = categorize_columns_for_cross_group(columns)

    for a_col in var_a_cols:
        for b_col in var_b_cols:
            # skip if same column
            if a_col == b_col:
                continue

            series_a = df[a_col]
            series_b = df[b_col]
            valid = series_a.notna() & series_b.notna()

            if valid.sum() >= min_periods:
                try:
                    if method == 'pearson':
                        r, p = pearsonr(series_a[valid], series_b[valid])
                    else:
                        r, p = spearmanr(series_a[valid], series_b[valid])
                except Exception:
                    r, p = np.nan, np.nan
            else:
                r, p = np.nan, np.nan

            # Fill symmetric positions in matrices
            corr.loc[a_col, b_col] = r
            corr.loc[b_col, a_col] = r
            pvals.loc[a_col, b_col] = p
            pvals.loc[b_col, a_col] = p

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
