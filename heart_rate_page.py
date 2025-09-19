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
            raise RuntimeError("Could not find load_excel_from_dropbox. Provide `sheets` or add loader module.")
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


def _plot_with_spline(ax, x, y, color, label):
    """Scatter + smooth spline line."""
    mask = ~np.isnan(y)
    x_valid = np.array(x)[mask]
    y_valid = np.array(y)[mask]
    if len(x_valid) >= 3:
        xnew = np.linspace(x_valid.min(), x_valid.max(), 300)
        spline = make_interp_spline(x_valid, y_valid, k=3)
        ynew = spline(xnew)
        ax.plot(xnew, ynew, color=color, alpha=0.8)
    elif len(x_valid) >= 2:
        ax.plot(x_valid, y_valid, color=color, alpha=0.6)
    ax.scatter(x_valid, y_valid, color=color, marker=MARKERS[label], s=40, label=label.replace("_", " "))


def hr_page(dbx=None, sheets=None):
    st.header("Heart Rate Overview")

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
    tachy = sheets.get("Tachy Events", pd.DataFrame()).copy()

    hr_stats = _ensure_date_col(hr_stats, "date")
    tachy = _ensure_date_col(tachy, "date")

    dates = []
    if not hr_stats.empty:
        dates.append(min(hr_stats["date"]))
        dates.append(max(hr_stats["date"]))
    if not tachy.empty:
        dates.append(min(tachy["date"]))
        dates.append(max(tachy["date"]))
    if not dates:
        st.info("No heart rate data available in the Excel file.")
        return

    min_date = min(dates)
    max_date = max(dates)

    col_t = st.columns([1, 1, 1, 1, 1])
    if "hr_view" not in st.session_state:
        st.session_state.hr_view = "D"
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
    idx = max(0, min(st.session_state[state_key], len(current_chunks) - 1))
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

    if view == "D":
        st.subheader(f"Tachy Events — {sel_start.strftime('%b %-d, %Y')}")
        if tachy_range.empty:
            st.info("No tachycardia events recorded for this day.")
        else:
            if "event_start" in tachy_range.columns:
                tachy_range["event_dt"] = pd.to_datetime(tachy_range["event_start"], errors="coerce")
            elif "event_start_epoch" in tachy_range.columns:
                tachy_range["event_dt"] = pd.to_datetime(tachy_range["event_start_epoch"], unit="s", errors="coerce")
            else:
                tachy_range["event_dt"] = pd.NaT
            events = tachy_range.dropna(subset=["event_dt"])
            if events.empty:
                st.info("No tachy events with valid times.")
            else:
                events["hour"] = (
                    events["event_dt"].dt.hour
                    + events["event_dt"].dt.minute / 60.0
                    + events["event_dt"].dt.second / 3600.0
                )
                max_bpm = events["max_bpm"].max()
                ymin = 100
                ymax = max(110, (max_bpm or ymin) + 5)

                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.scatter(events["hour"], events["max_bpm"], s=60, c="#d62728", edgecolor="k")
                for _, r in events.iterrows():
                    ax.text(r["hour"], r["max_bpm"] + 1, str(int(r.get("max_bpm", 0))), ha="center", va="bottom", fontsize=8)

                ax.set_xlim(0, 24)
                ax.set_ylim(ymin, ymax)
                ax.set_yticks(np.linspace(ymin, ymax, 5))
                ax.set_xticks([0, 6, 12, 18])
                ax.set_xticklabels(["12 AM", "6 AM", "12 PM", "6 PM"])
                ax.set_xlabel("Time of Day")
                ax.set_ylabel("Max BPM")
                ax.set_title("Tachy Events (start time vs max BPM)")
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.subheader("Max / Avg / Min Heart Rate")
        if hr_range.empty:
            st.info("No HR stats for this period.")
            return
        plot_df = hr_range.copy().sort_values("date")

        if view == "W":
            week_start = sel_start
            days = [week_start + timedelta(days=i) for i in range(7)]
            grouped = plot_df.set_index("date")
            hr_points = []
            for d in days:
                if d in grouped.index:
                    row = grouped.loc[[d]]
                    hr_points.append({"date": d, "HR_max": row["HR_max"].mean(), "HR_avg": row["HR_avg"].mean(), "HR_min": row["HR_min"].mean()})
                else:
                    hr_points.append({"date": d, "HR_max": np.nan, "HR_avg": np.nan, "HR_min": np.nan})
            hr_plot = pd.DataFrame(hr_points)

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                _plot_with_spline(ax, x_pos, hr_plot[col].values, COLORS[col], col)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([d.strftime("%a") for d in days])
            ax.set_xlabel("Day of Week")
            ax.set_ylabel("BPM")
            ax.set_title(f"Week: {sel_start.strftime('%b %-d, %Y')} → {sel_end.strftime('%b %-d, %Y')}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        elif view == "M":
            days = pd.date_range(sel_start, sel_end, freq="D").date
            grouped = plot_df.set_index("date")
            hr_points = []
            for d in days:
                if d in grouped.index:
                    row = grouped.loc[[d]]
                    hr_points.append({"date": d, "HR_max": row["HR_max"].mean(), "HR_avg": row["HR_avg"].mean(), "HR_min": row["HR_min"].mean()})
                else:
                    hr_points.append({"date": d, "HR_max": np.nan, "HR_avg": np.nan, "HR_min": np.nan})
            hr_plot = pd.DataFrame(hr_points)

            total_days = len(days)
            tick_days = [1, 7, 14, 21, 28]
            extra = [total_days] if total_days > 28 else []
            ticks_idx = [td - 1 for td in tick_days + extra if td <= total_days]
            labels = [str(td) for td in tick_days + extra if td <= total_days]

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                _plot_with_spline(ax, x_pos, hr_plot[col].values, COLORS[col], col)
            ax.set_xticks(ticks_idx)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Day of Month")
            ax.set_ylabel("BPM")
            ax.set_title(f"{sel_start.strftime('%B %Y')}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        elif view == "6M":
            all_days = pd.date_range(sel_start, sel_end, freq="D").date
            chunks = []
            for i in range(0, len(all_days), 3):
                block = all_days[i:i+3]
                subset = plot_df[plot_df["date"].isin(block)]
                chunks.append({
                    "date": block[0],
                    "HR_max": subset["HR_max"].mean() if not subset.empty else np.nan,
                    "HR_avg": subset["HR_avg"].mean() if not subset.empty else np.nan,
                    "HR_min": subset["HR_min"].mean() if not subset.empty else np.nan,
                })
            hr_plot = pd.DataFrame(chunks)

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                smoothed = hr_plot[col].rolling(window=2, min_periods=1).mean()
                ax.plot(x_pos, smoothed, color=COLORS[col], alpha=0.7)
                ax.scatter(x_pos, hr_plot[col], color=COLORS[col], marker=MARKERS[col], s=40, label=col.replace("_", " "))
            ax.set_xlabel("3-day chunks")
            ax.set_ylabel("BPM")
            ax.set_title(f"6-Month Window: {sel_start.strftime('%b %Y')} → {sel_end.strftime('%b %Y')}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        elif view == "Y":
            week_starts = []
            cur = sel_start - timedelta(days=(sel_start.weekday() + 1) % 7)
            while cur <= sel_end:
                week_starts.append(cur)
                cur += timedelta(weeks=1)

            chunks = []
            for ws in week_starts:
                we = ws + timedelta(days=6)
                subset = plot_df[(plot_df["date"] >= ws) & (plot_df["date"] <= we)]
                chunks.append({
                    "date": ws,
                    "HR_max": subset["HR_max"].mean() if not subset.empty else np.nan,
                    "HR_avg": subset["HR_avg"].mean() if not subset.empty else np.nan,
                    "HR_min": subset["HR_min"].mean() if not subset.empty else np.nan,
                })
            hr_plot = pd.DataFrame(chunks)

            months_initials = ["J","F","M","A","M","J","J","A","S","O","N","D"]
            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                smoothed = hr_plot[col].rolling(window=3, min_periods=1).mean()
                ax.plot(x_pos, smoothed, color=COLORS[col], alpha=0.7)
                ax.scatter(x_pos, hr_plot[col], color=COLORS[col], marker=MARKERS[col], s=40, label=col.replace("_", " "))

            month_positions = []
            month_labels = []
            for m in range(1, 13):
                mday = date(sel_start.year, m, 1)
                if not hr_plot.empty:
                    diffs = [abs((row - mday).days) for row in hr_plot["date"]]
                    if diffs:
                        pos = int(np.argmin(diffs))
                        month_positions.append(pos)
                        month_labels.append(months_initials[m-1])
            if month_positions:
                ax.set_xticks(month_positions)
                ax.set_xticklabels(month_labels)

            ax.set_xlabel("Weeks (approx)")
            ax.set_ylabel("BPM")
            ax.set_title(f"Year: {sel_start.year}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
