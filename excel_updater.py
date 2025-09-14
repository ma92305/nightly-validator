import json
import pandas as pd
import warnings
from io import BytesIO

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

def parse_tachy_events(tachy_events_input):
    events = {k: [] for k in ["event_start_epoch", "event_end_epoch", "event_start", "event_end", "duration_seconds", "max_bpm"]}
    if not tachy_events_input:
        return events
    try:
        if isinstance(tachy_events_input, dict):
            for k in events.keys():
                events[k].extend(tachy_events_input.get(k, []))
        elif isinstance(tachy_events_input, str):
            for line in tachy_events_input.strip().splitlines():
                if line.strip():
                    d = json.loads(line)
                    for k in events.keys():
                        events[k].extend(d.get(k, []))
    except Exception:
        pass
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

def update_combined_excel(dbx, dropbox_folder_path: str):
    """
    Generates a combined Excel file from all validated health logs in Dropbox
    and uploads it back to the same folder. Requires a dropbox.Dropbox client.
    """
    # --- Containers ---
    all_hr_stats, all_tachy_events, all_sleep_stats, all_weather_stats, all_hourly_weather = [], [], [], [], []
    all_symptoms, all_symptom_events, all_conditions, all_loc_activities, all_stairs, all_standing, all_walking = [], [], [], [], [], [], []
    all_nutrition_general, all_nutrition_meals, all_nutrition_liquids, all_digestion, all_meds = [], [], [], [], []
    all_daily_liquids, all_daily_meals = [], []
    all_validated_flags = []

    try:
        res = dbx.files_list_folder(dropbox_folder_path)
    except Exception as e:
        print("Dropbox folder error:", e)
        return

    for entry in res.entries:
        if not entry.name.endswith(".txt"):
            continue
        file_path = entry.path_lower
        try:
            metadata, f = dbx.files_download(file_path)
            data = json.loads(f.content)
        except Exception as e:
            print(f"Failed to read {entry.name}:", e)
            continue

        validated = data.get("validated_entries", {})
        val_dict = validated[0] if isinstance(validated, list) and validated else validated
        val_dict = {k: str(v).lower() == "true" for k, v in val_dict.items()} if isinstance(val_dict, dict) else {}

        # --- Store validated flags ---
        row_flags = {"File": entry.name}
        for key, value in val_dict.items():
            row_flags[key] = bool(value)
        all_validated_flags.append(row_flags)

        # --- Extract date ---
        file_date = pd.to_datetime(entry.name.split("_")[-1].replace(".txt",""), errors='coerce').date()

        # --- HEART RATE ---
        hr = data.get("heartrate_entries", {}) or {}
        all_hr_stats.append({
            "File": entry.name,
            "HR_max": hr.get("HR_max"),
            "HR_avg": hr.get("HR_avg"),
            "HR_min": hr.get("HR_min"),
            "tachy_percent": hr.get("tachy_percent"),
            "HRV": hr.get("HRV"),
            "date": file_date,
        })
        tachy_dict = parse_tachy_events(hr.get("tachy_events", {}))
        if any(len(v) > 0 for v in tachy_dict.values()):
            tachy_df = pd.DataFrame(tachy_dict)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                tachy_df["event_start"] = pd.to_datetime(tachy_df.get("event_start"), errors='coerce')
                tachy_df["event_end"] = pd.to_datetime(tachy_df.get("event_end"), errors='coerce')
            tachy_df["File"] = entry.name
            all_tachy_events.append(tachy_df)

        # --- SLEEP ---
        sleep = data.get("sleep_entries", {})
        sleep_date = pd.to_datetime(sleep.get("bedtime"), errors='coerce').date() if sleep.get("bedtime") else file_date
        all_sleep_stats.append({
            "File": entry.name,
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

        # --- WEATHER ---
        weather = data.get("weather_entries", {})
        if weather:
            stats = extract_weather_stats(weather)
            stats["File"] = entry.name
            stats["date"] = sleep_date
            all_weather_stats.append(stats)

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
                    df_regular["File"] = entry.name
                    all_symptoms.append(df_regular.sort_values("time", ascending=False))
                if not df_events.empty:
                    df_events = df_events[["item", "time"]].copy()
                    df_events["time"] = pd.to_datetime(df_events["time"], errors='coerce')
                    df_events["File"] = entry.name
                    all_symptom_events.append(df_events.sort_values("time", ascending=False))

        # --- CONDITIONS & ACTIVITIES ---
        if val_dict.get("conditions_valid", False):
            cond_entries = clean_entries(data.get("condition_entries", []), required_fields=["item","time"])
            if cond_entries:
                df_cond = pd.DataFrame(cond_entries)
                df_cond["time"] = pd.to_datetime(df_cond["time"], errors='coerce')
                df_cond["File"] = entry.name
                if "category" not in df_cond.columns:
                    df_cond["category"] = ""
                df_cond["category"] = df_cond["category"].astype(str).str.lower()
                df_activities = df_cond[df_cond["category"] == "activities"].copy()
                df_conditions = df_cond[df_cond["category"] == "conditions"].copy()
                if not df_activities.empty:
                    df_activities = df_activities.drop(columns=["category"], errors="ignore")
                    all_loc_activities.append(df_activities.sort_values("time", ascending=False))
                if not df_conditions.empty:
                    df_conditions = df_conditions.drop(columns=["category"], errors="ignore")
                    all_conditions.append(df_conditions.sort_values("time", ascending=False))
                all_stairs.append(df_cond[df_cond["item"].str.lower() == "stairs"].drop(columns=["status","category"], errors="ignore").sort_values("time", ascending=False))
                all_standing.append(df_cond[df_cond["item"].str.lower() == "🧍prolonged standing"].drop(columns=["status","category"], errors="ignore").sort_values("time", ascending=False))
                all_walking.append(df_cond[df_cond["item"].str.lower() == "walking"].drop(columns=["status","category"], errors="ignore").sort_values("time", ascending=False))

        # --- NUTRITION ---
        if val_dict.get("nutrition_valid", False):
            nutrition = clean_entries(data.get("nutrition_entries", []), required_fields=["item","time"])
            if nutrition:
                df_nutri = pd.DataFrame(nutrition).drop(columns=["category"], errors="ignore")
                df_nutri["time"] = pd.to_datetime(df_nutri["time"], errors='coerce')
                df_nutri["File"] = entry.name
                meals_df = df_nutri[df_nutri["item"].astype(str).str.contains("🍽️")].sort_values("time", ascending=False)
                liquids_df = df_nutri[df_nutri["item"].astype(str).str.contains("💧")].sort_values("time", ascending=False)
                general_df = df_nutri[~df_nutri["item"].astype(str).str.contains("🍽️|💧")].sort_values("time", ascending=False)
                all_nutrition_meals.append(meals_df)
                all_nutrition_liquids.append(liquids_df)
                all_nutrition_general.append(general_df)

                if not liquids_df.empty and "amount" in liquids_df.columns:
                    liquids_df["amount_num"] = liquids_df["amount"].apply(map_amount_to_number)
                    liquids_df["date"] = liquids_df["time"].dt.date
                    daily_liquids = liquids_df.groupby("date")["amount_num"].sum().reset_index().rename(columns={"amount_num":"daily_total_liquid"})
                    all_daily_liquids.append(daily_liquids)

                if not meals_df.empty and "amount" in meals_df.columns:
                    meals_df["amount_num"] = meals_df["amount"].apply(map_amount_to_number)
                    meals_df["date"] = meals_df["time"].dt.date
                    daily_meals = meals_df.groupby("date")["amount_num"].sum().reset_index().rename(columns={"amount_num":"daily_total_meals"})
                    all_daily_meals.append(daily_meals)

        # --- DIGESTION ---
        if val_dict.get("digestion_valid", False):
            digestion = clean_entries(data.get("digestion_entries", []), required_fields=["item","time"])
            if digestion:
                df_dig = pd.DataFrame(digestion).drop(columns=["category"], errors="ignore")
                df_dig["time"] = pd.to_datetime(df_dig["time"], errors='coerce')
                df_dig["File"] = entry.name
                all_digestion.append(df_dig.sort_values("time", ascending=False))

        # --- MEDS ---
        if val_dict.get("med_valid", False):
            meds = data.get("med_entries", [])
            med_list = [m for m in meds if isinstance(m, dict) and all(k in m for k in ["name","dose","time"])]
            if med_list:
                df_meds = pd.DataFrame(med_list)[["name","dose","time"]]
                df_meds["time"] = pd.to_datetime(df_meds["time"], errors='coerce')
                df_meds["File"] = entry.name
                all_meds.append(df_meds.sort_values("time", ascending=False))

    # --- Combine DataFrames ---
    def concat_or_empty(lst):
        return pd.concat(lst, ignore_index=True).sort_values("time", ascending=False, na_position="last") if lst else pd.DataFrame()

    hr_stats_df = pd.DataFrame(all_hr_stats).sort_values("date", ascending=False, na_position="last")
    tachy_df_all = concat_or_empty(all_tachy_events)
    sleep_stats_df = pd.DataFrame(all_sleep_stats).sort_values("date", ascending=False, na_position="last")
    weather_stats_df = pd.DataFrame(all_weather_stats)
    symptoms_df_all = concat_or_empty(all_symptoms)
    symptom_events_df_all = concat_or_empty(all_symptom_events)
    conditions_df_all = concat_or_empty(all_conditions)
    loc_activities_df_all = concat_or_empty(all_loc_activities)
    stairs_df_all = concat_or_empty(all_stairs)
    standing_df_all = concat_or_empty(all_standing)
    walking_df_all = concat_or_empty(all_walking)
    nutrition_general_df = concat_or_empty(all_nutrition_general)
    nutrition_meals_df = concat_or_empty(all_nutrition_meals)
    nutrition_liquids_df = concat_or_empty(all_nutrition_liquids)
    digestion_df_all = concat_or_empty(all_digestion)
    meds_df_all = concat_or_empty(all_meds)
    daily_liquids_df = pd.concat(all_daily_liquids, ignore_index=True).sort_values("date", ascending=False, na_position="last") if all_daily_liquids else pd.DataFrame()
    daily_meals_df = pd.concat(all_daily_meals, ignore_index=True).sort_values("date", ascending=False, na_position="last") if all_daily_meals else pd.DataFrame()
    validated_keys_df = pd.DataFrame(all_validated_flags).fillna(False)

    # --- Export to Dropbox Excel ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        hr_stats_df.to_excel(writer, sheet_name="HR Stats", index=False)
        tachy_df_all.to_excel(writer, sheet_name="Tachy Events", index=False)
        sleep_stats_df.to_excel(writer, sheet_name="Sleep Stats", index=False)
        weather_stats_df.to_excel(writer, sheet_name="Weather Stats", index=False)
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

    output.seek(0)
    excel_path = f"{dropbox_folder_path}/{EXCEL_FILENAME}"
    dbx.files_upload(output.read(), excel_path, mode=dropbox.files.WriteMode.overwrite)
    print("✅ Combined Excel updated in Dropbox:", excel_path)
