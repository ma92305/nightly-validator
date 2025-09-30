import streamlit as st
import pandas as pd
import io

DROPBOX_FOLDER = "/HealthLogs"
DROPBOX_EXCEL_NAME = "combined_data.xlsx"

@st.cache_data(ttl=300)
def load_excel_from_dropbox(_dbx, folder=DROPBOX_FOLDER, file_name=DROPBOX_EXCEL_NAME):
    path = f"{folder}/{file_name}"
    try:
        md, res = _dbx.files_download(path)
        excel_bytes = res.content

        # Read all sheets, force first row as header
        sheets = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None, header=0)

        # Ensure column names are strings and parse datetime columns
        for name, df in sheets.items():
            df.columns = df.columns.map(str)  # convert all column names to strings
            for col in df.columns:
                if "time" in col.lower() or "date" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors="ignore")
                    except Exception:
                        pass
            sheets[name] = df

            # --- DEBUG PRINT ---
            st.write(f"Columns in sheet '{name}': {df.columns.tolist()[:20]}")
            # Optionally print which migraine columns are missing
            MIGRAINE_SYMPTOMS = [
                "Headache","Right side headache","Left side headache","Nausea",
                "Sensory sensitivity","Vision issues","Eye pain","Base of head pain",
                "Neck pain","Fatigue","Dizziness","Fuzziness","Weakness/shakiness",
                "Brain fog","Heavy eyes","Negative mood","Overheating","Bloating",
                "Low appetite/early satiety","Stomach cramping"
            ]
            found = [c for c in MIGRAINE_SYMPTOMS if c in df.columns]
            missing = [c for c in MIGRAINE_SYMPTOMS if c not in df.columns]
            st.write(f"Migraine symptom columns found: {found}")
            st.write(f"Migraine symptom columns missing: {missing}")

        return sheets

    except Exception as e:
        st.error(f"Failed to load Excel from Dropbox: {e}")
        return {}
