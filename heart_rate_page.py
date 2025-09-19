# heart_rate_page.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import calendar
from dateutil.relativedelta import relativedelta
from scipy.interpolate import make_interp_spline

# colors / markers for the three series
COLORS = {"HR_max": "#d62728", "HR_avg": "#1f77b4", "HR_min": "#2ca02c"}
MARKERS = {"HR_max": "o", "HR_avg": "s", "HR_min": "D"}
TACHY_COLORS = {"tachy_percent": "#ff7f0e", "HRV": "#2ca02c"}
TACHY_MARKERS = {"tachy_percent": "o", "HRV": "s"}


def _ensure_date_col(df, col="date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _get_sheets_from_dbx(dbx):
    try:
        from load_excel import load_excel_from_dropbox
    except Exception:
        try:
            from validate_logs import load_excel_from_dropbox  # noqa: F401
        except Exception:
            raise RuntimeError("Could not find load_excel_from_dropbox.")
    return load_excel_from_dropbox(dbx)


def _make_day_chunks(min_date, max_date):
    days = []
    cur = min_date
    while cur <= max_date:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _make_week_chunks(min_date, max_date):
    start = min_date - timedelta(days=(min_date.weekday() + 1) % 7)
    weeks = []
    cur = start
    while cur <= max_date:
        weeks.append(cur)
        cur += timedelta(weeks=1)
    return weeks


def _make_month_chunks(min_date, max_date):
    months = []
    cur = date(min_date.year, min_date.month, 1)
    last = date(max_date.year, max_date.month, 1)
    while cur <= last:
        months.append(cur)
        cur = (cur + relativedelta(months=1))
    return months


def _make_6m_chunks(min_date, max_date):
    months = _make_month_chunks(min_date, max_date)
    chunks = []
    for start in months:
        if start <= max_date:
            chunks.append(start)
    return chunks


def _make_year_chunks(min_date, max_date):
    years = []
    cur = date(min_date.year, 1, 1)
    last = date(max_date.year, 1, 1)
    while cur <= last:
        years.append(cur)
        cur = date(cur.year + 1, 1, 1)
    return years


def hr_page(dbx=None, sheets=None):
    st.header("Heart Rate Overview")

    # load sheets if not provided
    if sheets is None:
        if dbx is None:
            st.error("No data source provided (dbx or sheets).")
            return
        try:
            sheets = _get_sheets_from_dbx(dbx)
        except Exception as e:
            st.error(f"Failed to load Excel sheets: {e}")
            return

    hr_stats = sheets.get("HR Stats", pd.DataFrame()).copy()
    tachy = sheets.get("HR Stats", pd.DataFrame()).copy()  # tachy percent & HRV in same sheet

    hr_stats = _ensure_date_col(hr_stats, "date")
    tachy = _ensure_date_col(tachy, "date")

    # compute available date range
    dates = []
    if not hr_stats.empty:
        dates.append(min(hr_stats["date"]))
        dates.append(max(hr_stats["date"]))
    if not tachy.empty:
        dates.append(min(tachy["date"]))
        dates.append(max(tachy["date"]))
    if not dates:
        st.info("No heart rate data available.")
        return

    min_date = min(dates)
    max_date = max(dates)

    # --- Time-chunk controls ---
    col_t = st.columns([1, 1, 1, 1, 1])
    if "hr_view" not in st.session_state:
        st.session_state.hr_view = "D"
    view = st.session_state.hr_view
    labels = ["D", "W", "M", "6M", "Y"]
    for i, lab in enumerate(labels):
        if col_t[i].button(lab, key=f"hr_view_{lab}"):
            st.session_state.hr_view = lab
    view = st.session_state.hr_view

    day_chunks = _make_day_chunks(min_date, max_date)
    week_chunks = _make_week_chunks(min_date, max_date)
    month_chunks = _make_month_chunks(min_date, max_date)
    chunks6m = _make_6m_chunks(min_date, max_date)
    year_chunks = _make_year_chunks(min_date, max_date)
    chunk_map = {"D": day_chunks, "W": week_chunks, "M": month_chunks, "6M": chunks6m, "Y": year_chunks}
    current_chunks = chunk_map.get(view, day_chunks)

    state_key = f"hr_index_{view}"
    if state_key not in st.session_state:
        st.session_state[state_key] = max(0, len(current_chunks) - 1)

    nav_col_l, nav_col_c, nav_col_r = st.columns([1, 6, 1])
    if nav_col_l.button("⬅️ Prev"):
        if st.session_state[state_key] > 0:
            st.session_state[state_key] -= 1
    idx = st.session_state[state_key]
    idx = max(0, min(idx, len(current_chunks) - 1))
    st.session_state[state_key] = idx

    if view == "D":
        sel_start = current_chunks[idx]
        sel_end = sel_start
        nav_col_c.markdown(f"### {sel_start.strftime('%b %-d, %Y')}")
    elif view == "W":
        week_start = current_chunks[idx]
        week_end = week_start + timedelta(days=6)
        sel_start, sel_end = week_start, week_end
        nav_col_c.markdown(f"### Week: {sel_start.strftime('%b %-d, %Y')} → {sel_end.strftime('%b %-d, %Y')}")
    elif view == "M":
        month_start = current_chunks[idx]
        month_end = (month_start + relativedelta(months=1)) - timedelta(days=1)
        sel_start, sel_end = month_start, month_end
        nav_col_c.markdown(f"### {month_start.strftime('%B %Y')}")
    elif view == "6M":
        six_start = current_chunks[idx]
        six_end = (six_start + relativedelta(months=6)) - timedelta(days=1)
        sel_start, sel_end = six_start, six_end
        nav_col_c.markdown(f"### {six_start.strftime('%b %Y')} → {six_end.strftime('%b %Y')}")
    else:
        y_start = current_chunks[idx]
        y_end = date(y_start.year, 12, 31)
        sel_start, sel_end = y_start, y_end
        nav_col_c.markdown(f"### Year: {y_start.year}")

    most_recent_idx = len(current_chunks) - 1
    if idx < most_recent_idx:
        if nav_col_r.button("Next ➡️"):
            st.session_state[state_key] = min(most_recent_idx, st.session_state[state_key] + 1)
    else:
        nav_col_r.write("")

    hr_range = hr_stats[(hr_stats["date"] >= sel_start) & (hr_stats["date"] <= sel_end)].copy()
    tachy_range = tachy[(tachy["date"] >= sel_start) & (tachy["date"] <= sel_end)].copy()

    # --- HR chart ---
    st.subheader("Max / Avg / Min Heart Rate")
    if hr_range.empty:
        st.info("No HR stats for this period.")
    else:
        plot_df = hr_range.sort_values("date")
        # for 6M, average 3-day chunks
        if view == "6M":
            all_days = pd.date_range(sel_start, sel_end, freq="D").date
            chunks = []
            for i in range(0, len(all_days), 3):
                block = all_days[i:i + 3]
                vals = {"date": block[0]}
                subset = plot_df[plot_df["date"].isin(block)]
                for col in ["HR_max", "HR_avg", "HR_min"]:
                    if not subset.empty and col in subset:
                        vals[col] = subset[col].dropna().astype(float).mean()
                    else:
                        vals[col] = np.nan
                chunks.append(vals)
            plot_df = pd.DataFrame(chunks)
        elif view == "Y":
            # weekly aggregation
            week_starts = []
            cur = sel_start - timedelta(days=(sel_start.weekday() + 1) % 7)
            while cur <= sel_end:
                week_starts.append(cur)
                cur += timedelta(weeks=1)
            chunks = []
            for ws in week_starts:
                we = ws + timedelta(days=6)
                subset = plot_df[(plot_df["date"] >= ws) & (plot_df["date"] <= we)]
                vals = {"date": ws}
                for col in ["HR_max", "HR_avg", "HR_min"]:
                    if not subset.empty and col in subset:
                        vals[col] = subset[col].dropna().astype(float).mean()
                    else:
                        vals[col] = np.nan
                chunks.append(vals)
            plot_df = pd.DataFrame(chunks)

        fig, ax = plt.subplots(figsize=(12, 4))
        x_pos = np.arange(len(plot_df))
        for col in ["HR_max", "HR_avg", "HR_min"]:
            y = plot_df[col].to_numpy(dtype=float)
            ax.scatter(x_pos, y, label=col.replace("_", " "), color=COLORS[col], marker=MARKERS[col])
            mask = ~np.isnan(y)
            if mask.sum() >= 2:
                if mask.sum() >= 3:
                    spline = make_interp_spline(x_pos[mask], y[mask], k=3)
                    xs = np.linspace(x_pos[mask].min(), x_pos[mask].max(), 200)
                    ys = spline(xs)
                else:
                    xs = x_pos[mask]
                    ys = y[mask]
                ax.plot(xs, ys, color=COLORS[col], linewidth=1.6, alpha=0.8)

        # x-axis labels for 6M & Y
        if view == "6M":
            month_positions = []
            month_labels = []
            chunk_dates = pd.to_datetime(plot_df["date"]).dt.date.tolist()
            cur = sel_start
            while cur <= sel_end:
                deltas = [abs((cd - cur).days) for cd in chunk_dates]
                if deltas:
                    pos = int(np.argmin(deltas))
                    if not month_positions or month_positions[-1] != pos:
                        month_positions.append(pos)
                        month_labels.append(cur.strftime("%b"))
                cur = cur + relativedelta(months=1)
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels)
        elif view == "Y":
            month_positions = []
            month_labels = []
            chunk_dates = pd.to_datetime(plot_df["date"]).dt.date.tolist()
            months_initials = ["J","F","M","A","M","J","J","A","S","O","N","D"]
            for m in range(1, 13):
                mday = date(sel_start.year, m, 1)
                deltas = [abs((cd - mday).days) for cd in chunk_dates]
                if deltas:
                    pos = int(np.argmin(deltas))
                    if not month_positions or month_positions[-1] != pos:
                        month_positions.append(pos)
                        month_labels.append(months_initials[m-1])
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels)
        ax.set_xlabel("Time")
        ax.set_ylabel("BPM")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # --- Tachy % & HRV chart ---
    st.subheader("Tachy % & HRV")
    if tachy_range.empty or not {"tachy_percent","HRV"}.issubset(tachy_range.columns):
        st.info("No tachy or HRV data for this period.")
    else:
        plot_df2 = tachy_range.sort_values("date")
        # aggregate like HR chart
        if view == "6M":
            all_days = pd.date_range(sel_start, sel_end, freq="D").date
            chunks = []
            for i in range(0, len(all_days), 3):
                block = all_days[i:i+3]
                vals = {"date": block[0]}
                subset = plot_df2[plot_df2["date"].isin(block)]
                for col in ["tachy_percent","HRV"]:
                    if not subset.empty and col in subset:
                        vals[col] = subset[col].dropna().astype(float).mean()
                    else:
                        vals[col] = np.nan
                chunks.append(vals)
            plot_df2 = pd.DataFrame(chunks)
        elif view == "Y":
            week_starts = []
            cur = sel_start - timedelta(days=(sel_start.weekday() + 1) % 7)
            while cur <= sel_end:
                week_starts.append(cur)
                cur += timedelta(weeks=1)
            chunks = []
            for ws in week_starts:
                we = ws + timedelta(days=6)
                subset = plot_df2[(plot_df2["date"] >= ws) & (plot_df2["date"] <= we)]
                vals = {"date": ws}
                for col in ["tachy_percent","HRV"]:
                    if not subset.empty and col in subset:
                        vals[col] = subset[col].dropna().astype(float).mean()
                    else:
                        vals[col] = np.nan
                chunks.append(vals)
            plot_df2 = pd.DataFrame(chunks)

        fig, ax = plt.subplots(figsize=(12,4))
        x_pos = np.arange(len(plot_df2))
        for col in ["tachy_percent","HRV"]:
            y = plot_df2[col].to_numpy(dtype=float)
            ax.scatter(x_pos, y, label=col.replace("_"," "), color=TACHY_COLORS[col], marker=TACHY_MARKERS[col])
            mask = ~np.isnan(y)
            if mask.sum() >= 2:
                if mask.sum() >=3:
                    spline = make_interp_spline(x_pos[mask], y[mask], k=3)
                    xs = np.linspace(x_pos[mask].min(), x_pos[mask].max(), 200)
                    ys = spline(xs)
                else:
                    xs = x_pos[mask]
                    ys = y[mask]
                ax.plot(xs, ys, color=TACHY_COLORS[col], linewidth=1.6, alpha=0.8)

        # x-axis labels
        if view == "6M":
            month_positions = []
            month_labels = []
            chunk_dates = pd.to_datetime(plot_df2["date"]).dt.date.tolist()
            cur = sel_start
            while cur <= sel_end:
                deltas = [abs((cd - cur).days) for cd in chunk_dates]
                if deltas:
                    pos = int(np.argmin(deltas))
                    if not month_positions or month_positions[-1] != pos:
                        month_positions.append(pos)
                        month_labels.append(cur.strftime("%b"))
                cur = cur + relativedelta(months=1)
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels)
        elif view == "Y":
            month_positions = []
            month_labels = []
            chunk_dates = pd.to_datetime(plot_df2["date"]).dt.date.tolist()
            months_initials = ["J","F","M","A","M","J","J","A","S","O","N","D"]
            for m in range(1,13):
                mday = date(sel_start.year, m,1)
                deltas = [abs((cd - mday).days) for cd in chunk_dates]
                if deltas:
                    pos = int(np.argmin(deltas))
                    if not month_positions or month_positions[-1] != pos:
                        month_positions.append(pos)
                        month_labels.append(months_initials[m-1])
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels)
        ax.set_xlabel("Time")
        ax.set_ylabel("BPM / %")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
