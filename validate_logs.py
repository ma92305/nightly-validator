import streamlit as st
import json
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from collections import defaultdict
import dropbox
from dropbox.files import WriteMode

# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="Nightly Validator",
    page_icon="favicon.png",  # can also use an emoji like "🩺"
)

dbx = dropbox.Dropbox(
    oauth2_refresh_token=st.secrets["dropbox_refresh_token"],
    app_key=st.secrets["dropbox_app_key"],
    app_secret=st.secrets["dropbox_app_secret"]
)

DROPBOX_FOLDER = "/HealthLogs"  # your folder in Dropbox

SEVERITIES = ["None", "⚪️", "🟡", "🟠", "🔴", "🟣"]

COLOR_MAP = {
    "None": (0.6, 0.6, 0.6),
    "⚪️": (1.0, 1.0, 1.0),
    "🟡": (1.0, 1.0, 0.0),
    "🟠": (1.0, 0.65, 0.0),
    "🔴": (1.0, 0.0, 0.0),
    "🟣": (0.5, 0.0, 0.5),
}

MINUTES_IN_DAY = 24 * 60

# -----------------------------
# Refresh Button + Cached Loader
# -----------------------------
st.sidebar.button("🔄 Refresh Data", on_click=lambda: st.session_state.pop("cached_logs", None))

def get_log(date):
    """Load log for a date, but only fetch from Dropbox if not cached or if refreshed."""
    if "cached_logs" not in st.session_state:
        st.session_state.cached_logs = {}

    key = date.strftime("%Y-%m-%d")

    if key not in st.session_state.cached_logs:
        log = load_log(date) or {}
        st.session_state.cached_logs[key] = log

    return st.session_state.cached_logs[key]

# -----------------------------
# Dropbox Helpers (define first)
# -----------------------------
def dropbox_read_json(filename):
    """Read JSON file from Dropbox HealthLogs folder"""
    path = f"{DROPBOX_FOLDER}/{filename}"
    try:
        md, res = dbx.files_download(path)
        return json.loads(res.content.decode("utf-8"))
    except dropbox.exceptions.ApiError:
        return {}
    except Exception:
        return {}

def dropbox_write_json(filename, data):
    """Write JSON file to Dropbox HealthLogs folder"""
    path = f"{DROPBOX_FOLDER}/{filename}"
    dbx.files_upload(
        json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        path,
        mode=WriteMode.overwrite
    )

def load_log(date):
    filename = f"health_log_{date.strftime('%Y-%m-%d')}.txt"
    return dropbox_read_json(filename)

def save_log(date, data):
    """
    Save a day's JSON to Dropbox. This restores emoji/time formats
    and sets data['validated']=True (keeps your existing behavior).
    """
    # Ensure we don't mutate original object passed in (copy)
    data_to_save = json.loads(json.dumps(data))  # shallow deep-copy via JSON
    data_to_save["validated"] = True

    restored_data = restore_emojis_and_times(data_to_save)
    filename = f"health_log_{date.strftime('%Y-%m-%d')}.txt"
    dropbox_write_json(filename, restored_data)

# -----------------------------
# Load reference lists ONCE (cached)
# -----------------------------
if "all_lists" not in st.session_state:
    all_lists = dropbox_read_json("all_lists.json")
    if not all_lists:
        st.error("⚠️ all_lists.json not found in Dropbox /HealthLogs")
        all_lists = {}
    st.session_state.all_lists = all_lists

ALL_CONDITIONS = st.session_state.all_lists.get("conditions", [])
NUTRITION_OPTIONS = st.session_state.all_lists.get("nutrition", [])
STANDARD_SYMPTOMS = st.session_state.all_lists.get("symptoms", [])
CONDITION_OPTIONS = st.session_state.all_lists.get("conditions", [])
AMOUNT_OPTIONS = ["A little", "Some", "Moderate", "A lot"]

# -----------------------------
# Utils
# -----------------------------
def normalize_emoji(s):
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFKC", s).strip()

def parse_datetime_safe(time_str):
    """
    Accepts ISO dates and the custom format "<Mon> <D>, <YYYY> at <H>:<MM> <AM/PM>".
    """
    try:
        return datetime.fromisoformat(time_str)
    except Exception:
        try:
            time_str_clean = time_str.replace("\u202f", " ")
            return datetime.strptime(time_str_clean, "%b %d, %Y at %I:%M %p")
        except Exception as e:
            st.error(f"Failed to parse time: {time_str}")
            raise e

def restore_emojis_and_times(obj):
    """Ensure severity codes are emojis and times use the desired format."""
    if isinstance(obj, list):
        return [restore_emojis_and_times(x) for x in obj]
    elif isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if k == "severity" and isinstance(v, str):
                new_dict[k] = normalize_emoji(v)
            elif k in ("time", "time_taken") and isinstance(v, str):
                # try parse several formats, preserve if unknown
                dt = None
                try:
                    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    try:
                        dt = datetime.strptime(v, "%b %d, %Y at %I:%M %p")
                    except Exception:
                        try:
                            # med time format fallback
                            dt = datetime.strptime(v, "%Y-%m-%d %H:%M")
                        except Exception:
                            dt = None
                if dt:
                    # Format chosen to match original style
                    if k == "time_taken":
                        new_dict[k] = dt.strftime("%Y-%m-%d %H:%M")
                    else:
                        # use same format as original files
                        new_dict[k] = dt.strftime("%b %-d, %Y at %-I:%M %p")
                else:
                    new_dict[k] = v
            else:
                new_dict[k] = restore_emojis_and_times(v)
        return new_dict
    elif isinstance(obj, str):
        return normalize_emoji(obj)
    return obj

# -----------------------------
# Build minute timeline
# -----------------------------
def build_minute_timeline(entries, selected_date, prev_last_entry="⚪️", universal_start_min=None):
    timeline = ["⚪️"] * MINUTES_IN_DAY

    # Sort entries by time (guard entries lacking time)
    entries_sorted = sorted(entries, key=lambda e: parse_datetime_safe(e["time"]) if e.get("time") else datetime.min)

    # Apply real entries first
    for i, entry in enumerate(entries_sorted):
        if not entry.get("time"):
            continue
        entry_min = (parse_datetime_safe(entry["time"]) - datetime.combine(selected_date, datetime.min.time())).seconds // 60
        if i + 1 < len(entries_sorted):
            next_min = (parse_datetime_safe(entries_sorted[i + 1]["time"]) - datetime.combine(selected_date, datetime.min.time())).seconds // 60
        else:
            next_min = MINUTES_IN_DAY
        # guard bounds
        entry_min = max(0, min(entry_min, MINUTES_IN_DAY))
        next_min = max(0, min(next_min, MINUTES_IN_DAY))
        if next_min > entry_min:
            timeline[entry_min:next_min] = [entry.get("severity", "⚪️")] * (next_min - entry_min)

    # Midnight → 3AM: carry over previous night value only if still default
    for i in range(0, 3*60):
        if timeline[i] == "⚪️":
            timeline[i] = prev_last_entry

    # 3AM exactly: set to "None"
    timeline[3*60] = "None"

    # Determine universal start if not provided
    if universal_start_min is None:
        post_3am_entries = [
            e for e in entries_sorted
            if e.get("time") and (parse_datetime_safe(e["time"]) - datetime.combine(selected_date, datetime.min.time())).seconds // 60 > 3*60
        ]
        if post_3am_entries:
            universal_start_dt = min(parse_datetime_safe(e["time"]) for e in post_3am_entries)
            universal_start_min = (universal_start_dt - datetime.combine(selected_date, datetime.min.time())).seconds // 60
        else:
            universal_start_min = 3*60 + 1  # fallback

    # 3AM → universal_start: fill "None"
    for i in range(3*60 + 1, universal_start_min):
        if i < MINUTES_IN_DAY:
            timeline[i] = "None"

    # From universal_start onward: carry forward last known value
    last_known = timeline[universal_start_min] if universal_start_min < MINUTES_IN_DAY else "⚪️"
    for i in range(universal_start_min, MINUTES_IN_DAY):
        if timeline[i] not in ["None"]:
            last_known = timeline[i]
        else:
            timeline[i] = last_known

    # Cut off at current time if today
    now = datetime.now()
    if selected_date == now.date():
        timeline = timeline[:now.hour*60 + now.minute]

    return timeline

# -----------------------------
# Plot timeline with "None" support
# -----------------------------
def plot_timeline_matplotlib(timeline, symptom, fig_height=1):
    if not timeline:
        timeline = ["None"]
    arr_rgb = np.array([[COLOR_MAP.get(normalize_emoji(s), (0,0,0)) for s in timeline]])
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.imshow(arr_rgb, aspect='auto')
    ax.set_yticks([])

    # X-axis: tick for every hour
    max_min = len(timeline)
    xticks = [i*60 for i in range(25) if i*60 <= max_min]  # include 24*60
    xticklabels = [(h % 12) or 12 for h in range(len(xticks))]  # 12-hour clock
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # 12-hour clock labels
    xticklabels = [(h % 12) or 12 for h in range(len(xticks))]
    ax.set_xticklabels(xticklabels)

    ax.set_title(symptom, fontsize=10)
    ax.set_facecolor("lightgray")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Streamlit UI - Header / Date
# -----------------------------

# Initialize selected_date in session_state
if "selected_date" not in st.session_state:
    now = datetime.now()
    # If it's between 12:00 AM and 3:59 AM, default to yesterday
    if 0 <= now.hour < 4:
        st.session_state.selected_date = date.today() - timedelta(days=1)
    else:
        st.session_state.selected_date = date.today()

# Navigation button handlers
def go_prev_day():
    st.session_state.selected_date -= timedelta(days=1)

def go_next_day():
    st.session_state.selected_date += timedelta(days=1)

def go_today():
    now = datetime.now()
    if 0 <= now.hour < 4:
        st.session_state.selected_date = date.today() - timedelta(days=1)
    else:
        st.session_state.selected_date = date.today()

# ---- Navigation row ----
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.button("⬅️ Previous", on_click=go_prev_day)

with col2:
    st.button("📅 Today", on_click=go_today)

with col3:
    # Only show next-day button if it's not in the future
    if st.session_state.selected_date < date.today():
        st.button("Next ➡️", on_click=go_next_day)

# ---- Date input ----
selected_date = st.date_input(
    "Pick a date",
    value=st.session_state.selected_date,
    key="selected_date"
)

# -----------------------------
# Symptoms Section (full logic preserved)
# -----------------------------
with st.expander("Symptoms", expanded=False):

    # Load data on date change
    if "loaded_date" not in st.session_state or st.session_state.loaded_date != selected_date:
        st.session_state.loaded_date = selected_date
        data = get_log(selected_date)
        st.session_state.data = data
        symptom_entries = data.get("symptom_entries", [])

        # --- previous night last entries
        prev_date = selected_date - timedelta(days=1)
        prev_data = get_log(prev_date)
        prev_entries = prev_data.get("symptom_entries", [])
        prev_last_entry = {}
        for symptom in STANDARD_SYMPTOMS:
            entries_prev = [e for e in prev_entries if e.get("item") == symptom]
            if entries_prev:
                entries_prev_sorted = sorted(entries_prev, key=lambda e: parse_datetime_safe(e["time"]))
                prev_last_entry[symptom] = entries_prev_sorted[-1].get("severity", "⚪️")
            else:
                prev_last_entry[symptom] = "⚪️"

        # --- universal earliest across all symptoms (after 3AM)
        all_post_3am_entries = [
            e for e in symptom_entries
            if e.get("time") and (parse_datetime_safe(e["time"]) - datetime.combine(selected_date, datetime.min.time())).seconds // 60 > 3*60
        ]
        if all_post_3am_entries:
            universal_start_dt = min(parse_datetime_safe(e["time"]) for e in all_post_3am_entries)
            universal_start_min = (universal_start_dt - datetime.combine(selected_date, datetime.min.time())).seconds // 60
        else:
            universal_start_min = 3*60 + 1  # fallback if no entries after 3AM

        st.session_state.original_timelines = {}
        for symptom in STANDARD_SYMPTOMS:
            entries = [e for e in symptom_entries if e.get("item") == symptom]
            st.session_state.original_timelines[symptom] = build_minute_timeline(
                entries,
                selected_date,
                prev_last_entry=prev_last_entry.get(symptom, "⚪️"),
                universal_start_min=universal_start_min
            )

        st.session_state.timelines = {
            symptom: st.session_state.original_timelines[symptom].copy()
            for symptom in STANDARD_SYMPTOMS
        }

    # --- Display each symptom
    for symptom in STANDARD_SYMPTOMS:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{symptom}**", unsafe_allow_html=True)
        with col2:
            if st.button("Edit", key=f"edit_{symptom}"):
                st.session_state[f"expander_{symptom}"] = not st.session_state.get(f"expander_{symptom}", False)

        # plot
        plot_timeline_matplotlib(st.session_state.timelines[symptom], symptom)

        # Conditional editing UI
        if st.session_state.get(f"expander_{symptom}", False):
            st.markdown(f"**Edit Timeline for {symptom}**")
            col1, col2, col3 = st.columns(3)
            if f"temp_start_{symptom}" not in st.session_state:
                st.session_state[f"temp_start_{symptom}"] = 8
            if f"temp_end_{symptom}" not in st.session_state:
                st.session_state[f"temp_end_{symptom}"] = 9
            if f"temp_sev_{symptom}" not in st.session_state:
                st.session_state[f"temp_sev_{symptom}"] = "⚪️"

            # Hour labels for 24h + midnight next day
            hour_labels = [datetime.strptime(str(h % 24), "%H").strftime("%-I %p") for h in range(25)]

            # Start time select
            start_label = col1.selectbox(
                "Start Hour",
                hour_labels,
                index=st.session_state[f"temp_start_{symptom}"],
                key=f"input_start_{symptom}"
            )

            # End time select (allow midnight of next day)
            end_label = col2.selectbox(
                "End Hour",
                hour_labels,
                index=st.session_state[f"temp_end_{symptom}"],
                key=f"input_end_{symptom}"
            )

            # Convert labels → hours
            start_hour = datetime.strptime(start_label, "%I %p").hour
            end_hour = datetime.strptime(end_label, "%I %p").hour

            # Handle 12AM next day
            if end_label == "12 AM" and hour_labels.index(end_label) == len(hour_labels) - 1:
                end_hour = 24

            # Construct datetimes
            start_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=start_hour)
            if end_hour == 24:
                end_dt = datetime.combine(selected_date + timedelta(days=1), datetime.min.time())
            else:
                end_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=end_hour)

            severity = col3.selectbox(
                "Severity", SEVERITIES,
                index=SEVERITIES.index(st.session_state[f"temp_sev_{symptom}"]),
                key=f"input_sev_{symptom}"
            )

            st.session_state[f"temp_start_{symptom}"] = start_hour
            st.session_state[f"temp_end_{symptom}"] = end_hour
            st.session_state[f"temp_sev_{symptom}"] = severity

            if st.button(f"Save Change - {symptom}", key=f"save_change_{symptom}"):
                # Construct full datetimes for start and end
                start_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=start_hour)
    
                # Handle 12AM next day properly
                if end_hour == 24:  # you should have set this when selecting 12AM next day
                    end_dt = datetime.combine(selected_date + timedelta(days=1), datetime.min.time())
                else:
                    end_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=end_hour)

                # Convert to timeline minutes for the current day
                start_min = int((start_dt - datetime.combine(selected_date, datetime.min.time())).total_seconds() // 60)
                end_min = int((end_dt - datetime.combine(selected_date, datetime.min.time())).total_seconds() // 60)

                # Trim end_min to max of 24*60 so timeline array doesn't overflow
                end_min_timeline = min(end_min, MINUTES_IN_DAY)

                # Apply severity to timeline
                st.session_state.timelines[symptom][start_min:end_min_timeline] = [severity] * (end_min_timeline - start_min)

                st.success("Timeline updated in place.")
                st.session_state[f"expander_{symptom}"] = False
                st.rerun()

            if st.button(f"Reset Timeline - {symptom}", key=f"reset_{symptom}"):
                st.session_state.timelines[symptom] = st.session_state.original_timelines[symptom].copy()
                st.success("Timeline reset to original file data.")
                st.rerun()

    # Event Timeline
    st.markdown("### Event Timeline")
    event_entries = [
        e for e in st.session_state.data.get("symptom_entries", [])
        if e.get("category", "").lower() == "event"
    ]
    if event_entries:
        event_entries_sorted = sorted(event_entries, key=lambda e: parse_datetime_safe(e["time"]))
        for entry in event_entries_sorted:
            entry_time = parse_datetime_safe(entry["time"]).strftime("%-I:%M %p")
            st.markdown(f"{entry_time} - {entry['item']}")
    else:
        st.markdown("_No events recorded for this day_")

    # Symptom-only Validate
    if st.button("✅ Validate & Upload Symptoms Only"):
        data = st.session_state.data

        # --- Sync edited timelines into data first
        new_entries_all = []
        for symptom in STANDARD_SYMPTOMS:
            timeline = st.session_state.timelines[symptom]
            last_sev = timeline[0]
            start_min = 0
            for i, sev in enumerate(timeline):
                if sev != last_sev:
                    dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(minutes=start_min)
                    new_entries_all.append({
                        "item": symptom,
                        "time": dt.strftime("%b %-d, %Y at %-I:%M %p"),
                        "severity": last_sev
                    })
                    last_sev = sev
                    start_min = i
            dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(minutes=start_min)
            new_entries_all.append({
                "item": symptom,
                "time": dt.strftime("%b %-d, %Y at %-I:%M %p"),
                "severity": last_sev
            })

        # Preserve events
        event_entries = [
            e for e in data.get("symptom_entries", [])
            if e.get("category", "").lower() == "event"
        ]
        data["symptom_entries"] = new_entries_all + event_entries

        # previous night last entries
        prev_date = selected_date - timedelta(days=1)
        prev_data = get_log(prev_date)
        prev_entries = prev_data.get("symptom_entries", [])
        prev_last_entry = {}
        for symptom in STANDARD_SYMPTOMS:
            entries_prev = [e for e in prev_entries if e.get("item") == symptom]
            if entries_prev:
                entries_prev_sorted = sorted(entries_prev, key=lambda e: parse_datetime_safe(e["time"]))
                prev_last_entry[symptom] = entries_prev_sorted[-1].get("severity", "⚪️")
            else:
                prev_last_entry[symptom] = "⚪️"

        # universal earliest across all symptoms after 3AM
        symptom_entries = data.get("symptom_entries", [])
        all_post_3am_entries = [
            e for e in symptom_entries
            if e.get("time") and (parse_datetime_safe(e["time"]) - datetime.combine(selected_date, datetime.min.time())).seconds // 60 > 3*60
        ]
        if all_post_3am_entries:
            universal_earliest = min(parse_datetime_safe(e["time"]) for e in all_post_3am_entries)
            universal_earliest_min = (universal_earliest - datetime.combine(selected_date, datetime.min.time())).seconds // 60
        else:
            universal_earliest_min = 3*60  # fallback

        for symptom in STANDARD_SYMPTOMS:
            timeline = st.session_state.timelines[symptom]

            # Midnight → 3AM: fill only None minutes with previous night value
            for i in range(0, min(3*60, len(timeline))):
                if timeline[i] == "None":
                    timeline[i] = prev_last_entry.get(symptom, "⚪️")

            # 3AM → universal earliest: fill only None minutes
            for i in range(3*60, min(universal_earliest_min, len(timeline))):
                if timeline[i] == "None":
                    timeline[i] = "None"

            # Universal earliest → rest of timeline: carry forward last known value
            last_known = prev_last_entry.get(symptom, "⚪️")
            for i in range(universal_earliest_min, len(timeline)):
                if timeline[i] != "None":
                    last_known = timeline[i]
                elif last_known:
                    timeline[i] = last_known

        # Update validated_entries
        if "validated_entries" not in data or not data["validated_entries"]:
            data["validated_entries"] = [{
                "symptoms_valid": "true",
                "conditions_valid": "false",
                "nutrition_valid": "false",
                "digestion_valid": "false",
                "reproductive_valid": "false"
            }]
        else:
            data["validated_entries"][0]["symptoms_valid"] = "true"

        save_log(selected_date, data)
        st.success("✅ Symptom timelines validated & uploaded (events preserved)!")

# -----------------------------
# Streamlit App - Conditions / Activities Section (Simple List)
# -----------------------------
# NOTE: CONDITION_OPTIONS was loaded from st.session_state.all_lists earlier

def merge_activity_sessions(entries, merge_gap_minutes=15):
    display_list = []
    merged_sessions = []

    activities = [e for e in entries if e.get("category") == "Activities" and e.get("status") in ("Start", "End")]
    other_entries = [e for e in entries if not (e.get("category") == "Activities" and e.get("status") in ("Start", "End"))]

    activities_by_item = defaultdict(list)
    for e in activities:
        activities_by_item[e["item"]].append(e)

    for item, item_entries in activities_by_item.items():
        item_entries.sort(key=lambda e: parse_datetime_safe(e["time"]))
        sessions = []
        current_start = None
        current_end = None

        for e in item_entries:
            time = parse_datetime_safe(e["time"])
            status = e["status"]

            if status == "Start":
                if current_start is None:
                    current_start = time
                    current_end = time
                else:
                    if current_end is not None and (time - current_end) <= timedelta(minutes=merge_gap_minutes):
                        current_end = time
                    else:
                        if current_end is not None:
                            sessions.append({"start_time": current_start, "end_time": current_end, "item": item})
                            display_list.append({"time": current_start, "item": f"{item} (Start)"})
                            display_list.append({"time": current_end, "item": f"{item} (End)"})
                        else:
                            display_list.append({"time": current_start, "item": f"{item} (Start) ❌", "red": True})
                        current_start = time
                        current_end = time
            elif status == "End":
                if current_start is None:
                    display_list.append({"time": time, "item": f"{item} (End) ❌", "red": True})
                else:
                    current_end = time

        if current_start is not None:
            if current_end is not None:
                sessions.append({"start_time": current_start, "end_time": current_end, "item": item})
                display_list.append({"time": current_start, "item": f"{item} (Start)"})
                display_list.append({"time": current_end, "item": f"{item} (End)"})
            else:
                display_list.append({"time": current_start, "item": f"{item} (Start) ❌", "red": True})

        merged_sessions.extend(sessions)

    for e in other_entries:
        display_list.append({"time": parse_datetime_safe(e["time"]), "item": e["item"]})

    display_list.sort(key=lambda e: e["time"])
    return display_list, merged_sessions

with st.expander("Conditions & Activities", expanded=False):

    # Load data if not loaded yet
    if "loaded_conditions_date" not in st.session_state or st.session_state.loaded_conditions_date != selected_date:
        st.session_state.loaded_conditions_date = selected_date
        data = get_log(selected_date)
        st.session_state.data = data
        condition_entries = data.get("condition_entries", [])
        st.session_state.original_conditions = condition_entries.copy()
        st.session_state.conditions = condition_entries.copy()

    col_add, col_remove, col_reset = st.columns(3)
    with col_add:
        add_clicked = st.button("Add entry")
    with col_remove:
        remove_clicked = st.button("Remove entry")
    with col_reset:
        reset_clicked = st.button("Reset entries")

    # Add
    if add_clicked:
        st.session_state.show_add = True

    if st.session_state.get("show_add", False):
        st.markdown("### Add Condition / Activity Entry")
        new_condition = st.selectbox("Select Condition", CONDITION_OPTIONS, key="add_cond_select")

        hour_labels = [datetime.strptime(str(h), "%H").strftime("%-I %p") for h in range(24)]
        selected_label = st.selectbox("Select Hour", hour_labels, index=12, key="add_cond_hour")
        new_hour = datetime.strptime(selected_label, "%I %p").hour

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Confirm Add"):
                dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=new_hour)
                st.session_state.conditions.append({
                    "time": dt.strftime("%b %-d, %Y at %-I:%M %p"),
                    "item": new_condition,
                    "category": "Conditions"
                })
                st.session_state.conditions.sort(key=lambda e: parse_datetime_safe(e["time"]))
                st.success(f"Added {new_condition} at {dt.strftime('%-I:%M %p')}")

    # Remove
    if remove_clicked:
        st.session_state.show_remove = True

    if st.session_state.get("show_remove", False):
        st.markdown("### Remove Condition / Activity Entry")
        cond_times = [f"{parse_datetime_safe(e['time']).strftime('%-I:%M %p')} - {e['item']}" for e in st.session_state.conditions]
        to_remove = st.selectbox("Select entry to remove", cond_times, key="remove_cond_select")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Confirm Remove"):
                idx = cond_times.index(to_remove)
                removed = st.session_state.conditions.pop(idx)
                st.success(f"Removed {removed['item']} at {removed['time']}")
        with col2:
            if st.button("Close Remove Section"):
                st.session_state.show_remove = False

    # Reset Entries
    if reset_clicked:
        st.session_state.conditions = st.session_state.original_conditions.copy()
        st.success("Reset to original file entries")

    # Display Chronological List
    st.markdown("### Condition & Activity Timeline")
    display_list, merged_sessions = merge_activity_sessions(st.session_state.conditions)
    for entry in display_list:
        entry_time_str = entry["time"].strftime("%-I:%M %p")
        if entry.get("red"):
            st.markdown(f"<span style='color:red'>{entry_time_str} - {entry['item']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"{entry_time_str} - {entry['item']}")

    # Save & Validate (merged sessions)
    if st.button("✅ Save & Validate Conditions"):
        data = st.session_state.data

        new_entries = [
            e for e in st.session_state.conditions
            if not (e.get("category") == "Activities" and e.get("status") in ("Start", "End"))
        ]

        for session in merged_sessions:
            new_entries.append({
                "time": session["start_time"].strftime("%b %-d, %Y at %-I:%M %p"),
                "item": session["item"],
                "status": "Start",
                "category": "Activities"
            })
            new_entries.append({
                "time": session["end_time"].strftime("%b %-d, %Y at %-I:%M %p"),
                "item": session["item"],
                "status": "End",
                "category": "Activities"
            })

        new_entries.sort(key=lambda e: parse_datetime_safe(e["time"]))

        data["condition_entries"] = new_entries

        if "validated_entries" not in data or not data["validated_entries"]:
            data["validated_entries"] = [{
                "symptoms_valid": "false",
                "conditions_valid": "true",
                "nutrition_valid": "false",
                "digestion_valid": "false",
                "reproductive_valid": "false"
            }]
        else:
            data["validated_entries"][0]["conditions_valid"] = "true"

        save_log(selected_date, data)
        st.success("✅ Condition & Activity entries saved and validated (status preserved)!")


# -----------------------------
# Streamlit App - Nutrition Section (Simple List)
# -----------------------------

with st.expander("Nutrition", expanded=False):

    if "loaded_nutrition_date" not in st.session_state or st.session_state.loaded_nutrition_date != selected_date:
        st.session_state.loaded_nutrition_date = selected_date
        data = get_log(selected_date) or {}
        st.session_state.data = data
        nutrition_entries = data.get("nutrition_entries", [])
        st.session_state.original_nutrition = json.loads(json.dumps(nutrition_entries))
        st.session_state.nutrition = json.loads(json.dumps(nutrition_entries))

    col_add, col_remove, col_reset = st.columns(3)
    with col_add:
        add_clicked = st.button("Add entry", key="nutrition_add")
    with col_remove:
        remove_clicked = st.button("Remove entry", key="nutrition_remove")
    with col_reset:
        reset_clicked = st.button("Reset entries", key="nutrition_reset")

    # -----------------------------
    # Add Entry Section
    # -----------------------------
    if add_clicked:
        st.session_state.show_add_nutrition = True

    if st.session_state.get("show_add_nutrition", False):
        st.markdown("### Add Nutrition Entry")
        new_item = st.selectbox("Select Nutrition Item", NUTRITION_OPTIONS, key="add_nutrition_select")

        # Hours as 12 AM → 11 PM
        hour_labels = [datetime.strptime(str(h), "%H").strftime("%-I %p") for h in range(24)]
        selected_label = st.selectbox("Select Hour", hour_labels, index=12, key="add_nutrition_hour")
        new_hour = datetime.strptime(selected_label, "%I %p").hour

        # Only show amount for Meal or Liquid
        new_amount = ""
        if "Meal" in new_item or "Liquid" in new_item:
            new_amount = st.selectbox("Select Amount", AMOUNT_OPTIONS, index=1, key="add_nutrition_amount")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Confirm Add", key="confirm_add_nutrition"):
                dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=new_hour)
                entry = {
                    "time": dt.strftime("%b %-d, %Y at %-I:%M %p"),
                    "item": new_item,
                    "category": "Nutrition"
                }
                if new_amount:  # only include if set
                    entry["amount"] = new_amount
                st.session_state.nutrition.append(entry)
                # Sort chronologically
                st.session_state.nutrition.sort(key=lambda e: parse_datetime_safe(e["time"]))

                if new_amount:
                    st.success(f"Added {new_item} ({new_amount}) at {dt.strftime('%-I:%M %p')}")
                else:
                    st.success(f"Added {new_item} at {dt.strftime('%-I:%M %p')}")

    # -----------------------------
    # Remove Entry Section
    # -----------------------------
    if remove_clicked:
        st.session_state.show_remove_nutrition = True

    if st.session_state.get("show_remove_nutrition", False):
        st.markdown("### Remove Nutrition Entry")
        nutrition_times = [f"{parse_datetime_safe(e['time']).strftime('%-I:%M %p')} - {e['item']} ({e.get('amount','')})" for e in st.session_state.nutrition]
        to_remove = st.selectbox("Select entry to remove", nutrition_times, key="remove_nutrition_select")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Confirm Remove", key="confirm_remove_nutrition"):
                idx = nutrition_times.index(to_remove)
                removed = st.session_state.nutrition.pop(idx)
                st.success(f"Removed {removed['item']} ({removed.get('amount','')}) at {removed['time']}")
        with col2:
            if st.button("Close Remove Section", key="close_remove_nutrition"):
                st.session_state.show_remove_nutrition = False

    # -----------------------------
    # Reset Entries
    # -----------------------------
    if reset_clicked:
        st.session_state.nutrition = st.session_state.original_nutrition.copy()
        st.success("Reset to original file entries")

    # -----------------------------
    # Display Chronological Lists
    # -----------------------------
    st.markdown("### Nutrition Timeline")

    # Separate entries
    meals = [e for e in st.session_state.nutrition if "Meal" in e["item"]]
    liquids = [e for e in st.session_state.nutrition if "Liquid" in e["item"]]
    others = [e for e in st.session_state.nutrition if "Meal" not in e["item"] and "Liquid" not in e["item"]]

    # Helper to render timeline
    def render_nutrition_list(title, entries):
        st.markdown(f"#### {title}")
        if entries:
            for entry in entries:
                entry_time = parse_datetime_safe(entry["time"]).strftime("%-I:%M %p")
                amt = f" ({entry['amount']})" if entry.get("amount") else ""
                st.markdown(f"{entry_time} - {entry['item']}{amt}")
        else:
            st.markdown("_No entries_")

    render_nutrition_list("Meals", meals)
    render_nutrition_list("Liquids", liquids)
    render_nutrition_list("Other Nutrition", others)

    # -----------------------------
    # Save & Validate
    # -----------------------------
    if st.button("✅ Save & Validate Nutrition"):
        data = st.session_state.data
        data["nutrition_entries"] = st.session_state.nutrition.copy()

        # --- Add or update validated_entries top key
        if "validated_entries" not in data or not data["validated_entries"]:
            data["validated_entries"] = [{
                "symptoms_valid": "false",
                "conditions_valid": "false",
                "nutrition_valid": "true",
                "digestion_valid": "false",
                "reproductive_valid": "false"
            }]
        else:
            # Only update nutrition_valid, preserve other keys
            data["validated_entries"][0]["nutrition_valid"] = "true"

        save_log(selected_date, data)
        st.success("✅ Nutrition entries saved and validated!")

# -----------------------------
# Streamlit App - Digestion Section (Simple List)
# -----------------------------
DIGESTION_OPTIONS = [
    "1: Separate hard lumps",
    "2: Lumpy and sausage like",
    "3: Cracked sausage",
    "4: Smooth snake",
    "5: Soft blobs",
    "6: Mushy blobs",
    "7: Liquid"
]

# Initialize session state keys
for key, default in {
    "digestion": [],
    "original_digestion": [],
    "show_add_digestion": False,
    "show_remove_digestion": False,
    "loaded_digestion_date": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.expander("Digestion", expanded=False):

    if st.session_state.loaded_digestion_date != selected_date:
        st.session_state.loaded_digestion_date = selected_date
        # Load from Dropbox instead of local iCloud
        data = get_log(selected_date) or {}
        st.session_state.data = data
        digestion_entries = data.get("digestion_entries", [])
        st.session_state.original_digestion = json.loads(json.dumps(digestion_entries))
        st.session_state.digestion = json.loads(json.dumps(digestion_entries))
        st.session_state.show_add_digestion = False
        st.session_state.show_remove_digestion = False

    col_add, col_remove, col_reset = st.columns(3)
    with col_add:
        if st.button("Add entry", key="add_digestion_btn"):
            st.session_state.show_add_digestion = True
            st.session_state.show_remove_digestion = False  # hide remove section

    with col_remove:
        if st.button("Remove entry", key="remove_digestion_btn"):
            st.session_state.show_remove_digestion = True
            st.session_state.show_add_digestion = False  # hide add section

    with col_reset:
        if st.button("Reset entries", key="reset_digestion_btn"):
            st.session_state.digestion = st.session_state.original_digestion.copy()
            st.session_state.show_add_digestion = False
            st.session_state.show_remove_digestion = False

    # -----------------------------
    # Add Entry Section
    # -----------------------------
    if st.session_state.show_add_digestion:
        st.markdown("### Add Digestion Entry")
        new_item = st.selectbox("Select Digestion Type", DIGESTION_OPTIONS, key="add_digestion_select")

        # Hours as 12 AM → 11 PM
        hour_labels = [datetime.strptime(str(h), "%H").strftime("%-I %p") for h in range(24)]
        selected_label = st.selectbox("Select Hour", hour_labels, index=12, key="add_digestion_hour")
        new_hour = datetime.strptime(selected_label, "%I %p").hour

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Confirm Add", key="confirm_add_digestion"):
                dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=new_hour)
                st.session_state.digestion.append({
                    "time": dt.strftime("%b %-d, %Y at %-I:%M %p"),
                    "item": new_item,
                    "category": "Digestion"
                })
                st.session_state.digestion.sort(key=lambda e: parse_datetime_safe(e["time"]))
                st.success(f"Added {new_item} at {dt.strftime('%-I:%M %p')}")
                st.session_state.show_add_digestion = False  # auto-close after adding

    # -----------------------------
    # Remove Entry Section
    # -----------------------------
    if st.session_state.show_remove_digestion:
        st.markdown("### Remove Digestion Entry")
        if st.session_state.digestion:
            dig_times = [
                f"{parse_datetime_safe(e['time']).strftime('%-I:%M %p')} - {e['item']}"
                for e in st.session_state.digestion
            ]
            to_remove = st.selectbox("Select entry to remove", dig_times, key="remove_digestion_select")
            if st.button("Confirm Remove", key="confirm_remove_digestion"):
                idx = dig_times.index(to_remove)
                removed = st.session_state.digestion.pop(idx)
                st.success(f"Removed {removed['item']} at {removed['time']}")
                st.session_state.show_remove_digestion = False  # auto-close after removing
        else:
            st.info("No digestion entries to remove.")
            st.session_state.show_remove_digestion = False

    # -----------------------------
    # Reset Entries
    # -----------------------------
    if reset_clicked:
        st.session_state.digestion = st.session_state.original_digestion.copy()
        st.success("Reset to original file entries")

    # -----------------------------
    # Display Chronological List
    # -----------------------------
    st.markdown("### Digestion Timeline")
    for entry in st.session_state.digestion:
        entry_time = parse_datetime_safe(entry["time"]).strftime("%-I:%M %p")
        st.markdown(f"{entry_time} - {entry['item']}")

    # -----------------------------
    # Save & Validate
    # -----------------------------
    if st.button("✅ Save & Validate Digestion", key="save_digestion_btn"):
        data = st.session_state.data
        data["digestion_entries"] = st.session_state.digestion.copy()

        # --- Add or update validated_entries top key
        if "validated_entries" not in data or not data["validated_entries"]:
            data["validated_entries"] = [{
                "symptoms_valid": "false",
                "conditions_valid": "false",
                "nutrition_valid": "false",
                "digestion_valid": "true",
                "reproductive_valid": "false"
            }]
        else:
            # Only update digestion_valid, preserve other keys
            data["validated_entries"][0]["digestion_valid"] = "true"

        save_log(selected_date, data)
        st.success("✅ Digestion entries saved and validated!")

# -----------------------------
# Reproductive Section
# -----------------------------
with st.expander("Reproductive", expanded=False):
    st.subheader("Reproductive Tracking")

    # Inputs
    bleeding = st.checkbox("Bleeding")
    spotting = st.checkbox("Spotting")
    menstrual_pain = st.checkbox("Menstrual Pain")
    ovarian_pain = st.checkbox("Ovarian Pain")

    bleeding_severity = None
    menstrual_pain_severity = None

    if bleeding:
        bleeding_severity = st.selectbox(
            "Bleeding Severity",
            ["🟡", "🟠", "🔴", "🟣"],
            key="bleeding_severity_select"
        )

    if menstrual_pain:
        menstrual_pain_severity = st.selectbox(
            "Menstrual Pain Severity",
            ["🟡", "🟠", "🔴", "🟣"],
            key="menstrual_pain_severity_select"
        )

    # Display existing reproductive data
    existing_repro = st.session_state.data.get("reproductive_entries", {})
    if existing_repro:
        st.markdown("### Existing Reproductive Data")
        for key, value in existing_repro.items():
            if value is not False:
                if key in ["bleeding", "menstrual_pain"] and f"{key}_severity" in existing_repro:
                    severity = existing_repro[f"{key}_severity"]
                    st.markdown(f"**{key.replace('_', ' ').capitalize()}:** {severity}")
                elif not key.endswith("_severity"):
                    st.markdown(f"**{key.replace('_', ' ').capitalize()}:** Yes")
    else:
        st.markdown("No reproductive data entered yet.")

    if st.button("✅ Save & Validate Reproductive"):
        data = st.session_state.data
        reproductive_entries = {
            "bleeding": bleeding,
            "spotting": spotting,
            "menstrual_pain": menstrual_pain,
            "ovarian_pain": ovarian_pain
        }
        if bleeding and bleeding_severity:
            reproductive_entries["bleeding_severity"] = bleeding_severity
        if menstrual_pain and menstrual_pain_severity:
            reproductive_entries["menstrual_pain_severity"] = menstrual_pain_severity

        data["reproductive_entries"] = reproductive_entries
        if "validated_entries" not in data or not data["validated_entries"]:
            data["validated_entries"] = [{
                "symptoms_valid": "false",
                "conditions_valid": "false",
                "nutrition_valid": "false",
                "digestion_valid": "false",
                "reproductive_valid": "true"
            }]
        else:
            data["validated_entries"][0]["reproductive_valid"] = "true"

        save_log(selected_date, data)
        st.success("✅ Reproductive data saved and validated!")

# -----------------------------
# Medication Section
# -----------------------------
with st.expander("Meds", expanded=False):

    STATUS_OPTIONS = ["taken", "skipped", "no data"]

    # -----------------------------
    # Initialize session state and load log
    # -----------------------------
    if "loaded_med_date" not in st.session_state or st.session_state.loaded_med_date != selected_date:
        st.session_state.loaded_med_date = selected_date
        data = get_log(selected_date)
        st.session_state.data = data
        med_entries = data.get("med_entries", [])
        st.session_state.original_med = med_entries.copy()
        st.session_state.med = med_entries.copy()  # initialize med session key

        # Initialize UI flags
        st.session_state.show_add_med = False
        st.session_state.show_remove_med = False

    # -----------------------------
    # Load all_lists.json and build MED_GROUPS
    # -----------------------------

    MED_GROUPS = {}
    all_lists = dropbox_read_json("all_lists.json")

    if all_lists:
        if "morning_med" in all_lists:
            MED_GROUPS["☀️Morning Meds"] = [json.loads(med_str) for med_str in all_lists["morning_med"]]
        if "night_med" in all_lists:
            MED_GROUPS["✨Night Meds"] = [json.loads(med_str) for med_str in all_lists["night_med"]]
        if "as_needed_med" in all_lists:
            MED_GROUPS["❔As Needed Meds"] = [json.loads(med_str) for med_str in all_lists["as_needed_med"]]
    else:
        st.error("all_lists.json not found in Dropbox/HealthLogs!")

    # -----------------------------
    # Add / Remove / Reset buttons
    # -----------------------------
    col_add, col_remove, col_reset = st.columns(3)
    with col_add:
        add_clicked = st.button("Add entry", key="med_add")
    with col_remove:
        remove_clicked = st.button("Remove entry", key="med_remove")
    with col_reset:
        reset_clicked = st.button("Reset entries", key="med_reset")

    # -----------------------------
    # Add Entry Section (Grouped Meds)
    # -----------------------------
    if add_clicked:
        st.session_state.show_add_med = True

    if st.session_state.get("show_add_med", False):
        st.markdown("### Add Med Entry")

        if MED_GROUPS:
            # Step 1: select med group
            selected_group = st.selectbox("Select Med Group", list(MED_GROUPS.keys()), key="add_med_group")

            # Step 2: select meds in that group
            group_meds = MED_GROUPS[selected_group]

            default_checked = "as needed" not in selected_group.lower()
            med_selections = {}
            for med in group_meds:
                # Parse dose safely
                dose_val = 0
                dose_str = med.get("dose", "")
                if dose_str:
                    try:
                        dose_val = float(dose_str.replace(" mg", ""))
                    except ValueError:
                        dose_val = 0

                label = f"{med['emoji']} {med['name']} ({dose_val} mg)" if dose_val > 0 else f"{med['emoji']} {med['name']}"
                med_selections[med['name'] + med['emoji']] = st.checkbox(label, value=default_checked, key=f"checkbox_{med['name']}_{med['emoji']}")

            # Step 3: group status
            new_status = st.selectbox("Status for this group", STATUS_OPTIONS, index=0, key="group_status")

            # Step 4: group time (only if taken)
            if new_status == "taken":
                hour_labels = [datetime.strptime(str(h), "%H").strftime("%-I %p") for h in range(0, 24)]
                minute_labels = ["00", "15", "30", "45"]

                col_hr, col_min = st.columns([1, 1])
                with col_hr:
                    selected_hour_label = st.selectbox("Hour", hour_labels, index=8, key="group_hour")  # default 8 AM
                with col_min:
                    selected_minute_label = st.selectbox("Minute", minute_labels, index=0, key="group_minute")

                new_hour = datetime.strptime(selected_hour_label, "%I %p").hour
                new_minute = int(selected_minute_label)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Confirm Add Group", key="confirm_add_group"):
                    for med in group_meds:
                        key = med['name'] + med['emoji']
                        if med_selections[key]:
                            entry_date = selected_date
                            entry_dt = None
                            if new_status == "taken":
                                entry_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(
                                    hours=new_hour, minutes=new_minute
                                )

                            # Parse dose safely again for entry
                            dose_val = 0
                            dose_str = med.get("dose", "")
                            if dose_str:
                                try:
                                    dose_val = float(dose_str.replace(" mg", ""))
                                except ValueError:
                                    dose_val = 0

                            entry = {
                                "status": new_status,
                                "emoji": med['emoji'],
                                "name": med['name'],
                                "dose": f"{dose_val} mg" if dose_val > 0 else "",
                                "time_taken": entry_dt.strftime("%Y-%m-%d %H:%M") if entry_dt else ""
                            }

                            st.session_state.med.append(entry)

                    st.session_state.med.sort(
                        key=lambda e: parse_datetime_safe(e["time_taken"]) if e["time_taken"] else datetime.min
                    )
                    st.success(f"Added selected meds from {selected_group} [{new_status}]")
            with col2:
                if st.button("Close Add Section", key="close_add_med"):
                    st.session_state.show_add_med = False
        else:
            st.warning("No med groups found in all_lists.json.")

    # -----------------------------
    # Remove, Reset, Display & Save
    # -----------------------------
    if remove_clicked:
        st.session_state.show_remove_med = True

    if st.session_state.get("show_remove_med", False):
        st.markdown("### Remove Medication")
        med_times = [
            f"{parse_datetime_safe(e['time_taken']).strftime('%-I:%M %p') if e['time_taken'] else ''} - {e['emoji']}{e['name']} ({e.get('dose','')}) [{e['status']}]"
            for e in st.session_state.med
        ]
        to_remove = st.selectbox("Select entry to remove", med_times, key="remove_med_select")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Confirm Remove", key="confirm_remove_med"):
                idx = med_times.index(to_remove)
                removed = st.session_state.med.pop(idx)
                st.success(f"Removed {removed['emoji']}{removed['name']} ({removed.get('dose','')}) [{removed['status']}] at {removed['time_taken']}")
        with col2:
            if st.button("Close Remove Section", key="close_remove_med"):
                st.session_state.show_remove_med = False

    if reset_clicked:
        st.session_state.med = st.session_state.original_med.copy()
        st.success("Reset to original file entries")

    st.markdown("### Medication Timeline")

    def render_med_list(title, entries):
        st.markdown(f"#### {title}")
        if entries:
            for entry in entries:
                entry_time = parse_datetime_safe(entry["time_taken"]).strftime("%-I:%M %p") if entry["time_taken"] else ""
                amt = f" ({entry['dose']})" if entry.get("dose") else ""
                st.markdown(f"{entry_time} - {entry['emoji']}{entry['name']}{amt} [{entry['status']}]")
        else:
            st.markdown("_No entries_")

    render_med_list("Current Entries", st.session_state.med)

    # -----------------------------
    # Save & Validate Medications
    # -----------------------------
    if st.button("✅ Save & Validate Medications"):
        data = st.session_state.data
        data["med_entries"] = st.session_state.med.copy()

        # Ensure validated_entries works even if [{}] or {}
        if "validated_entries" not in data or not data["validated_entries"] or data["validated_entries"] == [{}]:
            data["validated_entries"] = [{
                "symptoms_valid": "false",
                "conditions_valid": "false",
                "nutrition_valid": "false",
                "digestion_valid": "false",
                "reproductive_valid": "false",
                "med_valid": "true"
            }]
        else:
            # Only update med_valid, preserve other keys
            if isinstance(data["validated_entries"], list) and len(data["validated_entries"]) > 0:
                data["validated_entries"][0]["med_valid"] = "true"
            else:
                # fallback for empty dict
                data["validated_entries"] = [{
                    "symptoms_valid": "false",
                    "conditions_valid": "false",
                    "nutrition_valid": "false",
                    "digestion_valid": "false",
                    "reproductive_valid": "false",
                    "med_valid": "true"
                }]

        save_log(selected_date, data)
        st.success("✅ Medication entries saved and validated!")
