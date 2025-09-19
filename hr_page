import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from load_excel import load_excel_from_dropbox  # assuming you put the loader in its own file

# --- Heart Rate Page ---
def hr_page(_dbx):
    st.header("Heart Rate Overview")

    # Load from Dropbox Excel
    sheets = load_excel_from_dropbox(_dbx)
    if "Heart Rate" not in sheets:
        st.warning("No heart rate data available in the Excel file.")
        return

    hr_stats_df = sheets["Heart Rate"].copy()
    if hr_stats_df.empty:
        st.write("No heart rate data available.")
        return

    # Ensure 'date' column is datetime.date
    hr_stats_df['date'] = pd.to_datetime(hr_stats_df['date'], errors="coerce").dt.date
    hr_stats_df = hr_stats_df.dropna(subset=['date'])

    view = st.selectbox("Select view:", ["Monthly", "Yearly"])

    if view == "Monthly":
        hr_stats_df['month'] = pd.to_datetime(hr_stats_df['date']).dt.to_period('M')
        months = sorted(hr_stats_df['month'].unique(), reverse=True)
        selected_month = st.selectbox(
            "Select month:",
            months,
            format_func=lambda x: x.strftime("%B %Y")
        )
        plot_df = hr_stats_df[hr_stats_df['month'] == selected_month].sort_values('date')
        x_labels = plot_df['date']

    else:  # Yearly
        hr_stats_df['year'] = pd.to_datetime(hr_stats_df['date']).dt.year
        years = sorted(hr_stats_df['year'].unique(), reverse=True)
        selected_year = st.selectbox("Select year:", years)
        plot_df = hr_stats_df[hr_stats_df['year'] == selected_year].sort_values('date')
        x_labels = plot_df['date']

    if plot_df.empty:
        st.write("No data for this period.")
        return

    # --- Plot max HR ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_labels, plot_df['HR_max'], marker='o', linestyle='-', color='red')
    ax.set_title("Max Heart Rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("Max HR (bpm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Plot avg HR ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_labels, plot_df['HR_avg'], marker='o', linestyle='-', color='blue')
    ax.set_title("Average Heart Rate")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg HR (bpm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Plot tachy_percent ---
    if "tachy_percent" in plot_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_labels, plot_df['tachy_percent'], marker='o', linestyle='-', color='purple')
        ax.set_title("Tachycardia Percentage")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tachy %")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Tachycardia percentage not found in the dataset.")
