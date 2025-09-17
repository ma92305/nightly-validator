import json
import pandas as pd
import warnings
from io import BytesIO
import dropbox
from concurrent.futures import ThreadPoolExecutor, as_completed

EXCEL_FILENAME = "combined_data.xlsx"

quantity_map = {
    "a little": 1,
    "some": 2,
    "moderate": 3,
    "a lot": 4
}

def map_amount_to_number(amount_str):
    if not isinstance(amount_str, str):
        return 0
    return quantity_map.get(amount_str.strip().lower(), 0)

def clean_entries(entries, required_fields=None):
    cleaned = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"item": entry}
        if not isinstance(entry, dict):
            continue
        if required_fields and any(field not in entry or not entry.get(field) for field in required_fields):
            continue
        if all(v in [None, "", [], {}] for v in entry.values()):
            continue
        cleaned.append(entry)
    return cleaned

def parse_tachy_events(tachy_input):
    keys = ["event_start_epoch", "event_end_epoch", "event_start", "event_end", "duration_seconds", "max_bpm"]
    events = {k: [] for k in keys}

    if not tachy_input:
        return events

    try:
        if isinstance(tachy_input, list):
            # Correct: your JSON is a list of dicts
            for entry in tachy_input:
                if not isinstance(entry, dict):
                    continue
                for k in keys:
                    val = entry.get(k)
                    if val is None:
                        continue
                    # unwrap single-item lists
                    if isinstance(val, list) and len(val) == 1:
                        val = val[0]
                    events[k].append(val)
        elif isinstance(tachy_input, dict):
            for k in keys:
                val = tachy_input.get(k)
                if val is None:
                    continue
                if isinstance(val, list):
                    events[k].extend([v[0] if isinstance(v, list) and len(v)==1 else v for v in val])
                else:
                    events[k].append(val)
        elif isinstance(tachy_input, str):
            for line in tachy_input.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if isinstance(d, dict):
                        for k in keys:
                            val = d.get(k)
                            if val is None:
                                continue
                            if isinstance(val, list) and len(val) == 1:
                                val = val[0]
                            events[k].append(val)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print("⚠️ Error parsing tachy events:", e)

    return events
    
def extract_weather_stats(weather_entries):
    temperature = weather_entries.get("temperature", {})
    humidity = weather_entries.get("humidity", {})
    pressure = weather_entries.get("pressure", {})
    precipitation = weather_entries.get("precipitation", {})
    return {
        "temp_high": temperature.get("temp_high"),
        "temp_low": temperature.get("temp_low"),
        "temp_avg": temperature.get("temp_avg"),
        "humidity_avg": humidity.get("humidity_avg"),
        "pressure_avg": pressure.get("pressure_avg"),
        "pressure_min": pressure.get("pressure_min"),
        "pressure_max": pressure.get("pressure_max"),
        "precipitation_hours": precipitation.get("precipitation_hours"),
        "precipitation_total": precipitation.get("precipitation_total"),
    }

def update_combined_excel(dbx, dropbox_folder_path: str, max_workers=5, force_rebuild=False):
    """
    Full incremental Excel builder with:
    - Parallel JSON loading
    - DataFrame caching for speed
    - All sheets included (HR, Sleep, Weather, Symptoms, Meds, Nutrition, Digestion, Conditions, Activities, Stairs, Standing, Walking, Validated Keys)
    """

    cache_file = f"{dropbox_folder_path}/combined_cache.json"

    # --- Load existing cache ---
    try:
        md, res = dbx.files_download(cache_file)
        cache = json.loads(res.content.decode("utf-8"))
    except Exception:
        cache = {"logs": {}}

    updated = False

    # --- List all health log files ---
    try:
        res = dbx.files_list_folder(dropbox_folder_path)
    except Exception as e:
        print("Dropbox folder error:", e)
        return

    logs_to_process = []
    for entry in res.entries:
        if not entry.name.startswith("health_log_") or not entry.name.endswith(".txt"):
            continue
        date_str = entry.name.replace("health_log_", "").replace(".txt", "")
        last_mod = entry.server_modified.isoformat()
        cached = cache["logs"].get(date_str)
        if cached and cached.get("last_modified") == last_mod:
            continue
        logs_to_process.append((entry.path_lower, entry.name, date_str, last_mod))

    if not logs_to_process and cache["logs"] and not force_rebuild:
        print("⚡ No new/changed logs. Skipping Excel rebuild.")
        return

    # --- Parallel download & parse ---
    def fetch_log(entry_path, filename, date_str, last_mod):
        try:
            _, f = dbx.files_download(entry_path)
            data = json.loads(f.content.decode("utf-8"))
            return date_str, {"data": data, "last_modified": last_mod, "filename": filename}
        except Exception as e:
            print(f"Failed to read {filename}:", e)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_log, *args) for args in logs_to_process]
        for future in as_completed(futures):
            result = future.result()
            if result:
                date_str, log_data = result
                cache["logs"][date_str] = log_data
                updated = True

    if updated:
        dbx.files_upload(
            json.dumps(cache, ensure_ascii=False, indent=2).encode("utf-8"),
            cache_file,
            mode=dropbox.files.WriteMode.overwrite,
        )

    # --- Containers for all sheets ---
    sheets = {
        "hourly_weather": [],
        "symptoms": [],
        "conditions": [],
        "activities": [],
        "stairs": [],
        "standing": [],
        "walking": [],
        "nutrition_general": [],
        "nutrition_meals": [],
        "nutrition_liquids": [],
        "digestion": [],
        "meds": [],
        "validated_keys": []
    }

    # --- Process cached logs ---
    for date_str, payload in cache["logs"].items():
        data = payload["data"]
        filename = payload["filename"]
        validated = data.get("validated_entries", {})
        val_dict = validated[0] if isinstance(validated, list) and validated else validated
        val_dict = {k: str(v).lower() == "true" for k, v in val_dict.items()} if isinstance(val_dict, dict) else {}

        # Validated flags
        row_flags = {"File": filename}
        for k, v in val_dict.items():
            row_flags[k] = bool(v)
        sheets["validated_flags"].append(row_flags)

        file_date = pd.to_datetime(date_str, errors="coerce").date()

        # --- HR & Tachy Events ---
        hr = data.get("heartrate_entries", {})
        if isinstance(hr, list):
            hr = hr[0] if hr else {}

        hr_max = hr.get("HR_max")
        tachy_percent = hr.get("tachy_percent")

        # Apply rule: if HR_max < 100 and tachy_percent is blank, set to 0
        try:
            hr_max_val = float(hr_max)
        except (TypeError, ValueError):
            hr_max_val = None

        if (hr_max_val is not None) and hr_max_val < 100 and (tachy_percent in [None, ""]):
            tachy_percent = 0

        sheets["hr_stats"].append({
            "File": filename,
            "HR_max": hr_max,
            "HR_avg": hr.get("HR_avg"),
            "HR_min": hr.get("HR_min"),
            "tachy_percent": tachy_percent,
            "HRV": hr.get("HRV"),
            "date": file_date,
        })

        # --- Tachy Events ---
        tachy_dict = parse_tachy_events(hr.get("tachy_events", {}))
        if any(len(v) > 0 for v in tachy_dict.values()):
                try:
                        with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=UserWarning)
                                tachy_df = pd.DataFrame(tachy_dict)

                                # Parse datetime columns
                                tachy_df["event_start"] = pd.to_datetime(tachy_df.get("event_start"), errors="coerce")
                                tachy_df["event_end"] = pd.to_datetime(tachy_df.get("event_end"), errors="coerce")

                        # Add file info and the date column
                        tachy_df["File"] = filename
                        tachy_df["date"] = file_date

                        # Optional: sort events within the day by start time
                        if "event_start" in tachy_df.columns:
                                tachy_df = tachy_df.sort_values("event_start", ascending=True, na_position="last")

                        sheets["tachy_events"].append(tachy_df)

                except Exception as e:
                        print(f"⚠️ Failed to create tachy DataFrame for {filename}: {e}")

        # --- Sleep & Weather ---
        sleep = data.get("sleep_entries", {})
        sleep_date = pd.to_datetime(sleep.get("bedtime"), errors='coerce').date() if sleep.get("bedtime") else file_date
        sheets["sleep"].append({
            "File": filename,
            "date": sleep_date,
            "duration": sleep.get("asleep_time"),
            "score": sleep.get("sleep_score"),
            "bedtime": sleep.get("bedtime"),
            "waketime": sleep.get("waketime"),
            "rem_time": sleep.get("rem_time"),
            "core_time": sleep.get("core_time"),
            "deep_time": sleep.get("deep_time"),
            "awake_time": sleep.get("awake_time"),
        })
        weather = data.get("weather_entries", {})
        if weather:
            stats = extract_weather_stats(weather)
            stats["File"] = filename
            stats["date"] = sleep_date
            sheets["weather"].append(stats)

        # --- HOURLY WEATHER ---
        hourly_data = weather.get("hourly", [])
        if hourly_data:
                df_hourly = pd.DataFrame(hourly_data)
                df_hourly['time'] = pd.to_datetime(df_hourly['time'])
                df_hourly['Date'] = df_hourly['time'].dt.date
                df_hourly['Hour'] = df_hourly['time'].dt.hour
                df_hourly['pressure'] = pd.to_numeric(df_hourly['pressure'], errors='coerce')

                # Calculate cumulative pressure change over 6 hours
                df_hourly['Pressure Change (inHg)'] = df_hourly['pressure'].diff(periods=6).fillna(0)

                # Flag rapid changes
                df_hourly['Pressure Trend'] = df_hourly['Pressure Change (inHg)'].apply(
                        lambda x: 'Rapid Rise' if x > 0.09 else ('Rapid Fall' if x < -0.09 else 'Stable')
                )

                # Flag low pressure
                df_hourly['Low Pressure Flag'] = df_hourly['pressure'].apply(lambda x: 'Yes' if x <= 30.00 else 'No')

                df_hourly['File'] = filename
                sheets["hourly_weather"].append(df_hourly)
        
        # --- SYMPTOMS ---
        if val_dict.get("symptoms_valid", False):
            symptoms = clean_entries(data.get("symptom_entries", []), required_fields=["item", "time"])
            if symptoms:
                df_symp = pd.DataFrame(symptoms)
                if "category" not in df_symp.columns:
                    df_symp["category"] = ""
                df_symp["category"] = df_symp["category"].astype(str).str.lower()
                df_events = df_symp[df_symp["category"] == "event"].copy()
                df_regular = df_symp[df_symp["category"] != "event"].copy()
                if not df_regular.empty:
                    df_regular = df_regular.drop(columns=["category"], errors="ignore")
                    df_regular["time"] = pd.to_datetime(df_regular["time"], errors='coerce')
                    df_regular["File"] = filename
                    sheets["symptoms"].append(df_regular.sort_values("time", ascending=False))
                if not df_events.empty:
                    df_events = df_events[["item", "time"]].copy()
                    df_events["time"] = pd.to_datetime(df_events["time"], errors='coerce')
                    df_events["File"] = filename
                    sheets["symptom_events"].append(df_events.sort_values("time", ascending=False))

        # --- CONDITIONS & ACTIVITIES (Locations) ---
        if val_dict.get("conditions_valid", False):
                cond_entries = clean_entries(
                        data.get("condition_entries", []),
                        required_fields=["item","time"]
                )
                if cond_entries:
                        df_cond = pd.DataFrame(cond_entries)
                        df_cond["time"] = pd.to_datetime(df_cond["time"], errors='coerce')
                        df_cond["File"] = filename
                        if "category" not in df_cond.columns:
                                df_cond["category"] = ""
                        df_cond["category"] = df_cond["category"].astype(str).str.lower()

                        # Activities / Locations (keep status, drop quantity)
                        df_activities = df_cond[df_cond["category"] == "activities"].copy()
                        if not df_activities.empty:
                                df_activities = df_activities.drop(columns=["quantity"], errors="ignore")
                                sheets["activities"].append(df_activities.sort_values("time", ascending=False))

        # --- ACTIVITY ENTRIES (new dedicated section) ---
        activity_entries = clean_entries(
            data.get("activity_entries", []),
            required_fields=["item","time"]
        )
        if activity_entries:
            df_activity = pd.DataFrame(activity_entries)
            df_activity["time"] = pd.to_datetime(df_activity["time"], errors='coerce')
            df_activity["File"] = filename

            walking_df = df_activity[df_activity["item"].astype(str).str.contains("🚶", na=False)]
            standing_df = df_activity[df_activity["item"].astype(str).str.contains("🧍", na=False)]
            stairs_df  = df_activity[df_activity["item"].astype(str).str.contains("🪜", na=False)]

            if not walking_df.empty:
                sheets["walking"].append(walking_df.sort_values("time", ascending=False))
            if not standing_df.empty:
                sheets["standing"].append(standing_df.sort_values("time", ascending=False))
            if not stairs_df.empty:
                sheets["stairs"].append(stairs_df.sort_values("time", ascending=False))

        # --- NUTRITION ---
        if val_dict.get("nutrition_valid", False):
            nutrition = clean_entries(data.get("nutrition_entries", []), required_fields=["item","time"])
            if nutrition:
                df_nutri = pd.DataFrame(nutrition).drop(columns=["category"], errors="ignore")
                df_nutri["time"] = pd.to_datetime(df_nutri["time"], errors='coerce')
                df_nutri["File"] = filename
                meals_df = df_nutri[df_nutri["item"].astype(str).str.contains("🍽️")].sort_values("time", ascending=False)
                liquids_df = df_nutri[df_nutri["item"].astype(str).str.contains("💧")].sort_values("time", ascending=False)
                general_df = df_nutri[~df_nutri["item"].astype(str).str.contains("🍽️|💧")].sort_values("time", ascending=False)
                general_df = general_df.drop(columns=["amount"], errors="ignore")
                sheets["nutrition_general"].append(general_df)
                sheets["nutrition_meals"].append(meals_df)
                sheets["nutrition_liquids"].append(liquids_df)

                if not liquids_df.empty and "amount" in liquids_df.columns:
                    liquids_df["amount_num"] = liquids_df["amount"].apply(map_amount_to_number)
                    liquids_df["date"] = liquids_df["time"].dt.date
                    daily_liquids = liquids_df.groupby("date")["amount_num"].sum().reset_index().rename(columns={"amount_num":"daily_total_liquid"})
                    sheets["daily_liquids"].append(daily_liquids)

                if not meals_df.empty and "amount" in meals_df.columns:
                    meals_df["amount_num"] = meals_df["amount"].apply(map_amount_to_number)
                    meals_df["date"] = meals_df["time"].dt.date
                    daily_meals = meals_df.groupby("date")["amount_num"].sum().reset_index().rename(columns={"amount_num":"daily_total_meals"})
                    sheets["daily_meals"].append(daily_meals)

        # --- DIGESTION ---
        if val_dict.get("digestion_valid", False):
            digestion = clean_entries(data.get("digestion_entries", []), required_fields=["item","time"])
            if digestion:
                df_dig = pd.DataFrame(digestion).drop(columns=["category"], errors="ignore")
                df_dig["time"] = pd.to_datetime(df_dig["time"], errors='coerce')
                df_dig["File"] = filename
                sheets["digestion"].append(df_dig.sort_values("time", ascending=False))

        # --- MEDS ---
        if val_dict.get("med_valid", False):
            meds = data.get("med_entries", [])
            med_list = [m for m in meds if isinstance(m, dict) and all(k in m for k in ["name","dose","time"])]
            if med_list:
                df_meds = pd.DataFrame(med_list)[["name","dose","time"]]
                df_meds["time"] = pd.to_datetime(df_meds["time"], errors='coerce')
                df_meds["File"] = filename
                sheets["meds"].append(df_meds.sort_values("time", ascending=False))

    # --- Helper: concat and sort ---
    def concat_or_empty(lst):
        if not lst:
            return pd.DataFrame()
        df = pd.concat(lst, ignore_index=True)
        if "time" in df.columns:
            return df.sort_values("time", ascending=False, na_position="last")
        elif "date" in df.columns:
            return df.sort_values("date", ascending=False, na_position="last")
        return df

    # --- Combine DataFrames ---
    hr_stats_df = pd.DataFrame(sheets["hr_stats"]).sort_values("date", ascending=False, na_position="last")
    tachy_df_all = concat_or_empty(sheets["tachy_events"])
    sleep_stats_df = pd.DataFrame(sheets["sleep"]).sort_values("date", ascending=False, na_position="last")
    weather_stats_df = pd.DataFrame(sheets["weather"])
    if not weather_stats_df.empty and "date" in weather_stats_df.columns:
        weather_stats_df = weather_stats_df.sort_values("date", ascending=False, na_position="last")
    symptoms_df_all = concat_or_empty(sheets["symptoms"])
    symptom_events_df_all = concat_or_empty(sheets["symptom_events"])
    conditions_df_all = concat_or_empty(sheets["conditions"])
    loc_activities_df_all = concat_or_empty(sheets["activities"])
    stairs_df_all = concat_or_empty(sheets["stairs"])
    standing_df_all = concat_or_empty(sheets["standing"])
    walking_df_all = concat_or_empty(sheets["walking"])
    nutrition_general_df = concat_or_empty(sheets["nutrition_general"])
    nutrition_meals_df = concat_or_empty(sheets["nutrition_meals"])
    nutrition_liquids_df = concat_or_empty(sheets["nutrition_liquids"])
    digestion_df_all = concat_or_empty(sheets["digestion"])
    daily_liquids_df = concat_or_empty(sheets["daily_liquids"])
    daily_meals_df = concat_or_empty(sheets["daily_meals"])
    validated_keys_df = pd.DataFrame(sheets["validated_flags"]).fillna(False)

    if "File" in validated_keys_df.columns:
        # Extract the yyyy-MM-dd part and parse to datetime
        validated_keys_df["date"] = (
            validated_keys_df["File"]
            .str.extract(r"health_log_(\d{4}-\d{2}-\d{2})")[0]
            .apply(pd.to_datetime, errors="coerce")
        )
        validated_keys_df = validated_keys_df.sort_values("date", ascending=False).reset_index(drop=True)

    # --------------------------
    # Replace existing meds_df_all line with this block
    # --------------------------

    # Attempt to load all_lists from Dropbox (try common names)
    all_lists = {}
    for candidate in ("all_lists.json", "all_lists.txt", "all_lists"):
        try:
            md, res = dbx.files_download(f"{dropbox_folder_path}/{candidate}")
            all_lists = json.loads(res.content.decode("utf-8"))
            break
        except Exception:
            all_lists = {}

    def parse_med_items(raw_list):
        """
        raw_list: list of JSON strings or dicts (from your all_lists file)
        returns list of dicts with keys: name, emoji, dose
        """
        out = []
        if not raw_list:
            return out
        for item in raw_list:
            try:
                if isinstance(item, str):
                    parsed = json.loads(item)
                elif isinstance(item, dict):
                    parsed = item
                else:
                    continue
                # normalize keys
                parsed_name = parsed.get("name") or parsed.get("item") or ""
                out.append({
                    "name": parsed_name,
                    "emoji": parsed.get("emoji", ""),
                    "dose_default": parsed.get("dose", "")
                })
            except Exception:
                continue
        return out

    # Build master ordered med list: morning -> night -> as_needed
    morning_list = parse_med_items(all_lists.get("morning_med", []))
    night_list = parse_med_items(all_lists.get("night_med", []))
    as_needed_list = parse_med_items(all_lists.get("as_needed_med", []) or all_lists.get("as_needed", []))

    # create unique master order but keep duplicates if present in the lists separately
    master_order = []
    order_idx = {}
    idx = 0
    for lst, cat in [(morning_list, "morning"), (night_list, "night"), (as_needed_list, "as_needed")]:
        for med in lst:
            name = med["name"]
            if name not in order_idx:
                order_idx[name] = idx
                master_order.append({
                    "name": name,
                    "emoji": med.get("emoji", ""),
                    "dose_default": med.get("dose_default", ""),
                    "category": cat,
                    "order": idx
                })
                idx += 1

    # If some meds appear in files but not in all_lists, we'll add them at the end when discovered
    # First pass: discover first-taken date per med across ALL cached logs
    first_taken_date = {}  # med_name -> first date (datetime.date)

    for date_str, payload in cache["logs"].items():
        # only consider logs where med entries were validated
        data = payload.get("data", {})
        validated = data.get("validated_entries", {})
        val_dict = validated[0] if isinstance(validated, list) and validated else validated
        med_valid = False
        if isinstance(val_dict, dict):
            med_valid = bool(str(val_dict.get("med_valid", val_dict.get("medications_valid", False))).lower() == "true")
        # fallback: check top-level field "med_valid"
        if not med_valid:
            med_valid = bool(str(data.get("med_valid", False)).lower() == "true")
        if not med_valid:
            continue

        file_date = pd.to_datetime(date_str, errors="coerce").date()
        meds_entries = data.get("med_entries", []) or []
        for m in meds_entries:
            if not isinstance(m, dict):
                continue
            name = m.get("name") or m.get("medication") or ""
            status = str(m.get("status", "")).strip().lower()
            if name and status == "taken":
                existing = first_taken_date.get(name)
                if existing is None or file_date < existing:
                    first_taken_date[name] = file_date

    # Also ensure all meds in master_order are in first_taken_date only if they were ever taken;
    # meds not yet taken will not be added until first_taken_date exists for them.

    # Now build meds rows for every validated file/date
    meds_rows = []
    for date_str, payload in cache["logs"].items():
        data = payload.get("data", {})
        filename = payload.get("filename", f"health_log_{date_str}.txt")
        validated = data.get("validated_entries", {})
        val_dict = validated[0] if isinstance(validated, list) and validated else validated
        med_valid = False
        if isinstance(val_dict, dict):
            med_valid = bool(str(val_dict.get("med_valid", val_dict.get("medications_valid", False))).lower() == "true")
        if not med_valid:
            med_valid = bool(str(data.get("med_valid", False)).lower() == "true")
        if not med_valid:
            # skip adding meds for this date per your rule
            continue

        file_date = pd.to_datetime(date_str, errors="coerce").date()
        meds_entries = data.get("med_entries", []) or []
        # normalize med entries for quick lookup (case-insensitive)
        meds_lookup = {}
        for ent in meds_entries:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name") or ""
            if not name:
                continue
            key = name.strip().lower()
            meds_lookup.setdefault(key, []).append(ent)

        # For ordering: iterate through master_order; if a med name not present in master_order
        # but seen as taken in first_taken_date, add to end with next order idx (and category "unknown")
        # Build dynamic master list that includes discovered meds
        dynamic_master = list(master_order)  # shallow copy
        # find discovered meds not in order_idx
        for med_name in sorted(first_taken_date.keys()):
            if med_name not in order_idx:
                order_idx[med_name] = idx
                dynamic_master.append({
                    "name": med_name,
                    "emoji": "",
                    "dose_default": "",
                    "category": "unknown",
                    "order": idx
                })
                idx += 1

        # Iterate master list and only include med rows if the med's first_taken_date <= current file_date
        for med_info in dynamic_master:
            med_name = med_info["name"]
            med_key = med_name.strip().lower()
            med_cat = med_info.get("category", "unknown")
            # include only if med has a recorded first-taken date and that date <= this file_date
            first_date = first_taken_date.get(med_name) or first_taken_date.get(med_key)
            if first_date is None:
                # never recorded taken anywhere yet -> do not include on any date
                continue
            if first_date and file_date < first_date:
                # this date is before med first taken -> skip
                continue

            # find entries for this med on this date (could be multiple)
            entries_for_med = meds_lookup.get(med_key, [])
            picked_entry = entries_for_med[0] if entries_for_med else None

            # Determine status per rules
            if med_cat in ("morning", "night"):
                # morning/night -> taken if present and marked taken, else skipped
                if picked_entry and str(picked_entry.get("status", "")).strip().lower() == "taken":
                    status = "taken"
                else:
                    status = "skipped"
            else:
                # as_needed / unknown -> no data unless present as taken
                if picked_entry and str(picked_entry.get("status", "")).strip().lower() == "taken":
                    status = "taken"
                else:
                    status = "no data"

            # dose: prefer entry dose, else default from all_lists
            dose = ""
            if picked_entry:
                dose = picked_entry.get("dose") or picked_entry.get("amount") or ""
            if not dose:
                dose = med_info.get("dose_default", "") or ""

            # time taken: use entry time if present
            time_taken = ""
            if picked_entry:
                t = picked_entry.get("time") or picked_entry.get("time_taken")
                if t:
                    # try to normalize to ISO-ish string; if parse fails, keep original
                    try:
                        time_taken = pd.to_datetime(t).isoformat()
                    except Exception:
                        time_taken = str(t)

            meds_rows.append({
                "File": filename,
                "date": file_date,
                "time taken": time_taken,
                "medication": med_name,
                "status": status,
                "dose": dose,
                "emoji": med_info.get("emoji", "")
            })

    # Create meds_df_all from meds_rows and sort by master order then date desc
    if meds_rows:
        meds_df_all = pd.DataFrame(meds_rows)
        # map name -> order index (some meds may not exist in order_idx; default large)
        meds_df_all["med_order"] = meds_df_all["medication"].map(lambda n: order_idx.get(n, idx+1000))
        meds_df_all = meds_df_all.sort_values(["date", "med_order"], ascending=[False, True])
        meds_df_all = meds_df_all.drop(columns=["med_order"])
    else:
        meds_df_all = pd.DataFrame(columns=["File", "date", "time taken", "medication", "status", "dose", "emoji"])
    
    # --- Export to Excel ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        hr_stats_df.to_excel(writer, sheet_name="HR Stats", index=False)
        tachy_df_all.to_excel(writer, sheet_name="Tachy Events", index=False)
        sleep_stats_df.to_excel(writer, sheet_name="Sleep Stats", index=False)
        weather_stats_df.to_excel(writer, sheet_name="Weather Stats", index=False)
        hourly_weather_df = pd.concat(sheets["hourly_weather"], ignore_index=True) if sheets["hourly_weather"] else pd.DataFrame()
        if not hourly_weather_df.empty:
            # Sort by date (newest first) and then by hour
            hourly_weather_df = hourly_weather_df.sort_values(["Date", "Hour"], ascending=[False, True])
            hourly_weather_df.to_excel(writer, sheet_name="Hourly Weather", index=False)
        symptoms_df_all.to_excel(writer, sheet_name="Symptoms", index=False)
        if not symptom_events_df_all.empty:
            startrow_events = len(symptoms_df_all) + 3
            worksheet_symp = writer.sheets["Symptoms"]
            worksheet_symp.write(startrow_events, 0, "Symptom Events")
            symptom_events_df_all.to_excel(writer, sheet_name="Symptoms", startrow=startrow_events + 1, index=False)
        conditions_df_all.to_excel(writer, sheet_name="Conditions", index=False)
        loc_activities_df_all.to_excel(writer, sheet_name="Locations", index=False)
        stairs_df_all.to_excel(writer, sheet_name="Stairs", index=False)
        standing_df_all.to_excel(writer, sheet_name="Standing", index=False)
        walking_df_all.to_excel(writer, sheet_name="Walking", index=False)
        activity_entries_df_all = (
            pd.concat(sheets["activity_entries"], ignore_index=True)
            if sheets["activity_entries"] else pd.DataFrame()
        )
        if not activity_entries_df_all.empty:
            activity_entries_df_all.to_excel(writer, sheet_name="Activity Entries", index=False)
        nutrition_general_df.to_excel(writer, sheet_name="Nutrition - General", index=False)
        nutrition_meals_df.to_excel(writer, sheet_name="Nutrition - Meals", index=False)
        if not daily_meals_df.empty:
            startrow_meals = len(nutrition_meals_df) + 3
            worksheet_meals = writer.sheets["Nutrition - Meals"]
            worksheet_meals.write(startrow_meals, 0, "Daily Meal Totals")
            daily_meals_df.to_excel(writer, sheet_name="Nutrition - Meals", startrow=startrow_meals + 1, index=False)
        nutrition_liquids_df.to_excel(writer, sheet_name="Nutrition - Liquids", index=False)
        if not daily_liquids_df.empty:
            startrow_liquids = len(nutrition_liquids_df) + 3
            worksheet_liquids = writer.sheets["Nutrition - Liquids"]
            worksheet_liquids.write(startrow_liquids, 0, "Daily Liquid Totals")
            daily_liquids_df.to_excel(writer, sheet_name="Nutrition - Liquids", startrow=startrow_liquids + 1, index=False)
        digestion_df_all.to_excel(writer, sheet_name="Digestion", index=False)
        meds_df_all.to_excel(writer, sheet_name="Meds", index=False)
        validated_keys_df.to_excel(writer, sheet_name="Validated Keys", index=False)

        for sheet_name, worksheet in writer.sheets.items():
            # Map only the sheets that are *actually created as sheets*
            df_map = {
                "HR Stats": hr_stats_df,
                "Tachy Events": tachy_df_all,
                "Sleep Stats": sleep_stats_df,
                "Weather Stats": weather_stats_df,
                "Hourly Weather": hourly_weather_df,
                "Symptoms": symptoms_df_all,
                "Conditions": conditions_df_all,
                "Locations": loc_activities_df_all,
                "Stairs": stairs_df_all,
                "Standing": standing_df_all,
                "Walking": walking_df_all,
                "Nutrition - General": nutrition_general_df,
                "Nutrition - Meals": nutrition_meals_df,
                "Nutrition - Liquids": nutrition_liquids_df,
                "Digestion": digestion_df_all,
                "Meds": meds_df_all,
                "Validated Keys": validated_keys_df,
            }

            df = df_map.get(sheet_name)
            if df is not None and not df.empty:
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(i, i, max_len)
    
    output.seek(0)
    excel_path = f"{dropbox_folder_path}/{EXCEL_FILENAME}"
    dbx.files_upload(output.read(), excel_path, mode=dropbox.files.WriteMode.overwrite)

    print("✅ Incremental parallel Excel updated in Dropbox:", excel_path)
