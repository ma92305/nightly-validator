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
        sheets = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)

        # Try parsing date/time-like columns
        for name, df in sheets.items():
            for col in df.columns:
                if "time" in col.lower() or "date" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors="ignore")
                    except Exception:
                        pass
            sheets[name] = df

        return sheets
    except Exception as e:
        st.error(f"Failed to load Excel from Dropbox: {e}")
        return {}
