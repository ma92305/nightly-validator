
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
        # normalize to date objects (no time)
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def _get_sheets_from_dbx(dbx):
    # try to import loader (if you placed load_excel_from_dropbox in load_excel.py)
    try:
        from load_excel import load_excel_from_dropbox
    except Exception:
        # fallback to a function on validate_logs (if you kept it there)
        try:
            from validate_logs import load_excel_from_dropbox  # noqa: F401
        except Exception:
            raise RuntimeError("Could not find load_excel_from_dropbox. Provide `sheets` or add loader module.")
    # call loader
    return load_excel_from_dropbox(dbx)


def _make_day_chunks(min_date, max_date):
    days = []
    cur = min_date
    while cur <= max_date:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _make_week_chunks(min_date, max_date):
    # each chunk is a week starting Sunday -> Saturday
    # align min_date to previous Sunday
    start = min_date - timedelta(days=(min_date.weekday() + 1) % 7)
    weeks = []
    cur = start
    while cur <= max_date:
        weeks.append(cur)  # week start (Sunday)
        cur += timedelta(weeks=1)
    return weeks


def _make_month_chunks(min_date, max_date):
    # return first-of-months
    months = []
    cur = date(min_date.year, min_date.month, 1)
    last = date(max_date.year, max_date.month, 1)
    while cur <= last:
        months.append(cur)
        cur = (cur + relativedelta(months=1))
    return months


def _make_6m_chunks(min_date, max_date):
    # create rolling 6-month windows aligned to month boundaries (start at first of a month)
    months = _make_month_chunks(min_date, max_date)
    chunks = []
    for start in months:
        end = (start + relativedelta(months=6)) - timedelta(days=1)
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


def _safe_get(df, col):
    return df[col] if col in df.columns else pd.Series(dtype=float)


def _format_month_label(dt):
    return dt.strftime("%b")


def _format_year_months_label(start, end):
    return f"{start.strftime('%b %d, %Y')} → {end.strftime('%b %d, %Y')}"


def hr_page(dbx=None, sheets=None):
    """
    Main heart-rate page UI.
    Call as: hr_page(dbx) OR hr_page(sheets=your_sheets_dict)
    sheets = {"HR Stats": df, "Tachy Events": df, ...}
    """

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

    # normalize
    hr_stats = sheets.get("HR Stats", pd.DataFrame()).copy()
    tachy = sheets.get("Tachy Events", pd.DataFrame()).copy()

    hr_stats = _ensure_date_col(hr_stats, "date")
    tachy = _ensure_date_col(tachy, "date")

    # compute available date range from both sheets
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

    # --- Time-chunk controls ---
    col_t = st.columns([1, 1, 1, 1, 1])
    # keep selection in session_state
    if "hr_view" not in st.session_state:
        st.session_state.hr_view = "D"
    view = None
    labels = ["D", "W", "M", "6M", "Y"]
    for i, lab in enumerate(labels):
        if col_t[i].button(lab, key=f"hr_view_{lab}"):
            st.session_state.hr_view = lab
    view = st.session_state.hr_view

    # prepare chunk lists
    day_chunks = _make_day_chunks(min_date, max_date)
    week_chunks = _make_week_chunks(min_date, max_date)
    month_chunks = _make_month_chunks(min_date, max_date)
    chunks6m = _make_6m_chunks(min_date, max_date)
    year_chunks = _make_year_chunks(min_date, max_date)

    chunk_map = {"D": day_chunks, "W": week_chunks, "M": month_chunks, "6M": chunks6m, "Y": year_chunks}
    current_chunks = chunk_map.get(view, day_chunks)

    # session key for index for each view
    state_key = f"hr_index_{view}"
    if state_key not in st.session_state:
        # default to last (most recent) chunk
        st.session_state[state_key] = max(0, len(current_chunks) - 1)

    # navigation row (left arrow, label, right arrow)
    nav_col_l, nav_col_c, nav_col_r = st.columns([1, 6, 1])
    if nav_col_l.button("⬅️ Prev"):
        if st.session_state[state_key] > 0:
            st.session_state[state_key] -= 1
    # label
    idx = st.session_state[state_key]
    # Bound index
    idx = max(0, min(idx, len(current_chunks) - 1))
    st.session_state[state_key] = idx
    # compute display range for this chunk
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
    else:  # Y
        y_start = current_chunks[idx]
        y_end = date(y_start.year, 12, 31)
        sel_start, sel_end = y_start, y_end
        nav_col_c.markdown(f"### Year: {y_start.year}")

    # hide right arrow if current chunk is most recent (no future)
    most_recent_idx = len(current_chunks) - 1
    if idx < most_recent_idx:
        if nav_col_r.button("Next ➡️"):
            st.session_state[state_key] = min(most_recent_idx, st.session_state[state_key] + 1)
    else:
        # show disabled or nothing
        nav_col_r.write("")

    # filter data to selected range
    hr_range = hr_stats[(hr_stats["date"] >= sel_start) & (hr_stats["date"] <= sel_end)].copy()
    tachy_range = tachy[(tachy["date"] >= sel_start) & (tachy["date"] <= sel_end)].copy()

    # --- Plot for Day view: tachy events plotted at start time with y = max_bpm ---
    if view == "D":
        st.subheader(f"Tachy Events — {sel_start.strftime('%b %-d, %Y')}")
        if tachy_range.empty:
            st.info("No tachycardia events recorded for this day.")
        else:
            # parse event_start times to hour-of-day (float)
            def to_hour_frac(ts):
                ts = pd.to_datetime(ts)
                seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
                return seconds / 3600.0

            # get events on this single day
            events = tachy_range.copy()
            # prefer event_start column
            if "event_start" in events.columns:
                events["event_dt"] = pd.to_datetime(events["event_start"], errors="coerce")
            elif "event_start_epoch" in events.columns:
                events["event_dt"] = pd.to_datetime(events["event_start_epoch"], unit="s", errors="coerce")
            else:
                events["event_dt"] = pd.NaT

            events = events.dropna(subset=["event_dt"])
            if events.empty:
                st.info("No tachy events with valid times.")
            else:
                events["hour"] = events["event_dt"].dt.hour + events["event_dt"].dt.minute / 60.0 + events["event_dt"].dt.second / 3600.0
                # Y scale: start at 100 to day's max + cushion
                max_bpm = events["max_bpm"].max()
                ymin = 100
                ymax = max(110, (max_bpm or ymin) + 5)

                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.scatter(events["hour"], events["max_bpm"], s=60, c="#d62728", edgecolor="k")
                # annotate with time maybe
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

    # --- Plot for W / M / 6M / Y: plot HR_max/avg/min per-day (or aggregated) as dots + smoothed lines ---
    else:
        st.subheader("Max / Avg / Min Heart Rate")

        if hr_range.empty:
            st.info("No HR stats for this period.")
            return

        # for 6M: average over 3-day chunks per dot
        # for Y: average per calendar week
        plot_df = hr_range.copy()
        plot_df = plot_df.sort_values("date")

        # ---------- WEEK VIEW ----------
        if view == "W":
            week_start = sel_start
            days = [week_start + timedelta(days=i) for i in range(7)]
            grouped = plot_df.set_index("date")
            hr_points = []
            for d in days:
                row = grouped.loc[[d]] if d in grouped.index else pd.DataFrame()
                if not row.empty:
                    hr_points.append({
                        "date": d,
                        "HR_max": pd.to_numeric(row["HR_max"].dropna().astype(float)).mean() if "HR_max" in row else np.nan,
                        "HR_avg": pd.to_numeric(row["HR_avg"].dropna().astype(float)).mean() if "HR_avg" in row else np.nan,
                        "HR_min": pd.to_numeric(row["HR_min"].dropna().astype(float)).mean() if "HR_min" in row else np.nan,
                    })
                else:
                    hr_points.append({"date": d, "HR_max": np.nan, "HR_avg": np.nan, "HR_min": np.nan})
            hr_plot = pd.DataFrame(hr_points)

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                y = hr_plot[col].to_numpy(dtype=float)
                ax.scatter(x_pos, y, label=col.replace("_", " "), color=COLORS[col], marker=MARKERS[col])
                # smoothing: spline if >=3 points, linear if 2
                mask = ~np.isnan(y)
                if mask.sum() >= 3:
                    try:
                        xs = x_pos[mask]
                        ys = y[mask]
                        spline = make_interp_spline(xs, ys, k=3)
                        xs_smooth = np.linspace(xs.min(), xs.max(), 200)
                        ys_smooth = spline(xs_smooth)
                        ax.plot(xs_smooth, ys_smooth, color=COLORS[col], linewidth=1.6, alpha=0.8)
                    except Exception:
                        pass
                elif mask.sum() == 2:
                    xs = x_pos[mask]
                    ys = y[mask]
                    ax.plot(xs, ys, color=COLORS[col], linewidth=1.2, alpha=0.8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([d.strftime("%a") for d in days])
            ax.set_xlabel("Day of Week")
            ax.set_ylabel("BPM")
            ax.set_title(f"Week: {sel_start.strftime('%b %-d, %Y')} → {sel_end.strftime('%b %-d, %Y')}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        # ---------- MONTH VIEW ----------
        elif view == "M":
            month_start = sel_start
            month_end = sel_end
            days = pd.date_range(month_start, month_end, freq="D").date
            grouped = plot_df.set_index("date")
            hr_points = []
            for d in days:
                if d in grouped.index:
                    row = grouped.loc[[d]]
                    hr_points.append({
                        "date": d,
                        "HR_max": pd.to_numeric(row["HR_max"].dropna().astype(float)).mean() if "HR_max" in row else np.nan,
                        "HR_avg": pd.to_numeric(row["HR_avg"].dropna().astype(float)).mean() if "HR_avg" in row else np.nan,
                        "HR_min": pd.to_numeric(row["HR_min"].dropna().astype(float)).mean() if "HR_min" in row else np.nan,
                    })
                else:
                    hr_points.append({"date": d, "HR_max": np.nan, "HR_avg": np.nan, "HR_min": np.nan})
            hr_plot = pd.DataFrame(hr_points)

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                y = hr_plot[col].to_numpy(dtype=float)
                ax.scatter(x_pos, y, label=col.replace("_", " "), color=COLORS[col], marker=MARKERS[col])
                # smoothing: spline if >=3 points
                mask = ~np.isnan(y)
                if mask.sum() >= 3:
                    try:
                        xs = x_pos[mask]
                        ys = y[mask]
                        spline = make_interp_spline(xs, ys, k=3)
                        xs_smooth = np.linspace(xs.min(), xs.max(), 300)
                        ys_smooth = spline(xs_smooth)
                        ax.plot(xs_smooth, ys_smooth, color=COLORS[col], linewidth=1.6, alpha=0.8)
                    except Exception:
                        pass
                elif mask.sum() == 2:
                    xs = x_pos[mask]
                    ys = y[mask]
                    ax.plot(xs, ys, color=COLORS[col], linewidth=1.2, alpha=0.8)

            # x ticks at 1,7,14,21,28 and extras
            total_days = len(days)
            tick_days = [1, 7, 14, 21, 28]
            extra = []
            if total_days > 28:
                extra.append(total_days)
            ticks_idx = []
            labels = []
            for td in tick_days + extra:
                if td <= total_days:
                    ticks_idx.append(td - 1)
                    labels.append(str(td))

            ax.set_xticks(ticks_idx)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Day of Month")
            ax.set_ylabel("BPM")
            ax.set_title(f"{month_start.strftime('%B %Y')}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        # ---------- 6-MONTH VIEW ----------
        elif view == "6M":
            start = sel_start
            end = sel_end
            all_days = pd.date_range(start, end, freq="D").date
            chunks = []
            for i in range(0, len(all_days), 3):
                block = all_days[i:i + 3]
                vals = {"date": block[0]}
                subset = plot_df[plot_df["date"].isin(block)]
                if not subset.empty:
                    vals["HR_max"] = pd.to_numeric(subset["HR_max"].dropna().astype(float)).mean()
                    vals["HR_avg"] = pd.to_numeric(subset["HR_avg"].dropna().astype(float)).mean()
                    vals["HR_min"] = pd.to_numeric(subset["HR_min"].dropna().astype(float)).mean()
                else:
                    vals["HR_max"] = np.nan
                    vals["HR_avg"] = np.nan
                    vals["HR_min"] = np.nan
                chunks.append(vals)
            hr_plot = pd.DataFrame(chunks)

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            # scatter points
            for col in ["HR_max", "HR_avg", "HR_min"]:
                y = hr_plot[col].to_numpy(dtype=float)
                ax.scatter(x_pos, y, label=col.replace("_", " "), color=COLORS[col], marker=MARKERS[col])

            # rolling average line (window=3) for smoothing
            for col in ["HR_max", "HR_avg", "HR_min"]:
                series = pd.Series(hr_plot[col].to_numpy(dtype=float))
                if series.dropna().shape[0] >= 2:
                    y_roll = series.rolling(window=3, min_periods=1).mean().to_numpy()
                    # plot continuous line, ignoring leading/trailing NaNs by masking
                    mask = ~np.isnan(y_roll)
                    if mask.sum() >= 2:
                        ax.plot(x_pos[mask], y_roll[mask], color=COLORS[col], linewidth=1.6, alpha=0.9)

            # place month labels at closest chunk to 1st of each month in window
            month_positions = []
            month_labels = []
            if not hr_plot.empty:
                chunk_dates = pd.to_datetime(hr_plot["date"]).dt.date.tolist()
                cur = date(start.year, start.month, 1)
                while cur <= end:
                    # compute closest chunk index to this month start
                    deltas = [abs((cd - cur).days) for cd in chunk_dates]
                    if deltas:
                        pos = int(np.argmin(deltas))
                        # avoid duplicates (if two consecutive months map to same pos)
                        if not month_positions or month_positions[-1] != pos:
                            month_positions.append(pos)
                            month_labels.append(cur.strftime("%b"))
                    cur = (cur + relativedelta(months=1))

            if month_positions:
                ax.set_xticks(month_positions)
                ax.set_xticklabels(month_labels)
            else:
                ax.set_xticks([])
            ax.set_xlabel("Month (3-day chunks shown as dots)")
            ax.set_ylabel("BPM")
            ax.set_title(f"6-Month Window: {start.strftime('%b %Y')} → {end.strftime('%b %Y')}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        # ---------- YEAR VIEW ----------
        elif view == "Y":
            start = sel_start
            end = sel_end

            # build weeks starting at nearest Sunday before start
            week_starts = []
            cur = start - timedelta(days=(start.weekday() + 1) % 7)
            while cur <= end:
                week_starts.append(cur)
                cur += timedelta(weeks=1)

            chunks = []
            for ws in week_starts:
                we = ws + timedelta(days=6)
                subset = plot_df[(plot_df["date"] >= ws) & (plot_df["date"] <= we)]
                vals = {"date": ws}
                if not subset.empty:
                    vals["HR_max"] = pd.to_numeric(subset["HR_max"].dropna().astype(float)).mean()
                    vals["HR_avg"] = pd.to_numeric(subset["HR_avg"].dropna().astype(float)).mean()
                    vals["HR_min"] = pd.to_numeric(subset["HR_min"].dropna().astype(float)).mean()
                else:
                    vals["HR_max"] = np.nan
                    vals["HR_avg"] = np.nan
                    vals["HR_min"] = np.nan
                chunks.append(vals)

            hr_plot = pd.DataFrame(chunks)

            fig, ax = plt.subplots(figsize=(12, 4))
            x_pos = np.arange(len(hr_plot))
            for col in ["HR_max", "HR_avg", "HR_min"]:
                y = hr_plot[col].to_numpy(dtype=float)
                ax.scatter(x_pos, y, label=col.replace("_", " "), color=COLORS[col], marker=MARKERS[col])

            # rolling average smoothing (window=3)
            for col in ["HR_max", "HR_avg", "HR_min"]:
                series = pd.Series(hr_plot[col].to_numpy(dtype=float))
                if series.dropna().shape[0] >= 2:
                    y_roll = series.rolling(window=3, min_periods=1).mean().to_numpy()
                    mask = ~np.isnan(y_roll)
                    if mask.sum() >= 2:
                        ax.plot(x_pos[mask], y_roll[mask], color=COLORS[col], linewidth=1.6, alpha=0.9)

            # x ticks: months (J F M A M J J A S O N D)
            months_initials = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
            month_positions = []
            month_labels = []
            if not hr_plot.empty:
                chunk_dates = pd.to_datetime(hr_plot["date"]).dt.date.tolist()
                for m in range(1, 13):
                    mday = date(sel_start.year, m, 1)
                    deltas = [abs((cd - mday).days) for cd in chunk_dates]
                    if deltas:
                        pos = int(np.argmin(deltas))
                        if not month_positions or month_positions[-1] != pos:
                            month_positions.append(pos)
                            month_labels.append(months_initials[m - 1])

            if month_positions:
                ax.set_xticks(month_positions)
                ax.set_xticklabels(month_labels)

            ax.set_xlabel("Weeks (approx)")
            ax.set_ylabel("BPM")
            ax.set_title(f"Year: {sel_start.year}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

    st.info("Tip: use the D/W/M/6M/Y buttons and the arrows to move backward/forward through time windows.")
