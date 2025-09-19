import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import timedelta

# Symptom severity mapping
SYMPTOM_SCALE = {"⚪️": 1, "🟡": 2, "🟠": 3, "🔴": 4, "🟣": 5}

def daily_total_symptom_score(df):
    if df.empty:
        return pd.DataFrame(columns=["date", "total_symptom_score"])
    df["severity_num"] = df["severity"].map(SYMPTOM_SCALE).fillna(np.nan)
    grouped = df.groupby([df["time"].dt.date, "item"])["severity_num"].mean().reset_index()
    total = grouped.groupby("time")["severity_num"].sum().reset_index()
    total.rename(columns={"time": "date", "severity_num": "total_symptom_score"}, inplace=True)
    return total

def aggregate_symptoms(df, view, selected_period=None):
    df = df.copy()
    if df.empty:
        return df
    df['time'] = pd.to_datetime(df['time'])
    df["severity_num"] = df["severity"].map(SYMPTOM_SCALE).fillna(np.nan)

    if view == "Day":
        df['date_only'] = df['time'].dt.date
        if selected_period:
            df = df[df['date_only'] == selected_period]
        df['period'] = df['time'].dt.hour
        period_order = list(range(24))

    elif view == "Week":
        df['week_start'] = (df['time'] - pd.to_timedelta(df['time'].dt.weekday, unit='d')).dt.normalize()
        if selected_period:
            df = df[df['week_start'] == pd.to_datetime(selected_period).normalize()]
        df['day'] = df['time'].dt.dayofweek
        df['chunk'] = df['time'].dt.hour // 8
        df['period'] = df['day'] * 3 + df['chunk']
        period_order = list(range(21))

    elif view == "Month":
        df['month'] = df['time'].dt.to_period('M')
        if selected_period:
            df = df[df['month'] == selected_period]
        df['period'] = df['time'].dt.date
        period_order = sorted(df['period'].unique())

    elif view == "Year":
        df['year'] = df['time'].dt.year
        if selected_period:
            df = df[df['year'] == selected_period]
        df['period'] = df['time'].dt.to_period('2W').apply(lambda r: r.start_time)
        period_order = sorted(df['period'].unique())

    if df.empty:
        return df
    agg = df.groupby(['period', 'item'])['severity_num'].mean().reset_index()
    agg['period'] = pd.Categorical(agg['period'], categories=period_order, ordered=True)
    return agg.sort_values('period')

def format_labels(view, periods):
    if view == "Day":
        return [f"{p}:00" for p in periods]
    elif view == "Week":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return [f"{days[p//3]} {['0-8h','8-16h','16-24h'][p%3]}" for p in periods]
    elif view == "Month":
        return [str(p) for p in periods]
    elif view == "Year":
        return [p.strftime('%b %d') for p in periods]
    return periods

def plot_symptom_heatmap(df, view):
    pivot = df.pivot(index="item", columns="period", values="severity_num").fillna(0)
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot)/2)))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "", ["#ffffff", "#ffff00", "#ff9900", "#ff0000", "#800080"]
    )
    sns.heatmap(pivot, cmap=cmap, cbar=True, ax=ax, linewidths=0.5, linecolor='gray', vmin=1, vmax=5)
    ax.set_xlabel(view)
    ax.set_ylabel("Symptom")
    ax.set_title(f"Symptom Severity Heatmap ({view} View)")
    ax.set_xticklabels(format_labels(view, pivot.columns), rotation=45, ha="right")
    plt.tight_layout()
    return fig

def symptoms_page(symptoms_df):
    st.header("Total Symptom Score per Day")
    total_symptom_df = daily_total_symptom_score(symptoms_df)
    if not total_symptom_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(total_symptom_df["date"], total_symptom_df["total_symptom_score"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Score")
        ax.set_title("Total Symptom Score Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("No symptom data available.")

    st.subheader("Symptom Severity Heatmap")
    view = st.selectbox("Select view:", ["Day", "Week", "Month", "Year"], index=2)

    if view == "Day":
        available_periods = sorted(symptoms_df['time'].dt.date.unique(), reverse=True)
    elif view == "Week":
        available_periods = sorted(
            (symptoms_df['time'] - pd.to_timedelta(symptoms_df['time'].dt.weekday, unit='d')).dt.normalize().unique(),
            reverse=True
        )
    elif view == "Month":
        available_periods = sorted(symptoms_df['time'].dt.to_period('M').unique(), reverse=True)
    elif view == "Year":
        available_periods = sorted(symptoms_df['time'].dt.year.unique(), reverse=True)
    else:
        available_periods = []

    selected_period = st.selectbox("Select period:", available_periods)
    agg = aggregate_symptoms(symptoms_df, view, selected_period)
    if not agg.empty:
        st.pyplot(plot_symptom_heatmap(agg, view))
    else:
        st.write("No data for selection.")
