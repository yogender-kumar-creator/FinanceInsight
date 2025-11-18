import streamlit as st
import pandas as pd
import os
import json
# === Paths (Docker) ===
SEGMENT_DIR = "/app/outputs/doc_segments"
TABLE_INDEX = "/app/outputs/tables/tables_index.csv"
EVENTS = "/app/outputs/financial_events_extracted.csv"
VERIFIED = "/app/outputs/financial_events_verified.csv"
NER_ERRORS = "/app/outputs/eda/fiqa_errors.csv"


st.set_page_config(page_title="Financial Document Analyzer", layout="wide")

st.title("📘 Financial Document Analysis Dashboard")
st.write("This interface allows you to explore segmented sections, extracted tables, events, and financial insights.")

# --- Segmented Sections ---
st.header("📄 Segmented Report Sections")
json_files = [f for f in os.listdir(SEGMENT_DIR) if f.endswith("_sections.json")]

if json_files:
    selected_json = st.selectbox("Select segmented JSON:", json_files)
    json_path = os.path.join(SEGMENT_DIR, selected_json)
    data = json.load(open(json_path, "r"))

    st.subheader("Detected Sections")
    st.json(data["sections"])

    selected_section = st.selectbox("Open Section Text", list(data["sections"].keys()))
    if selected_section:
        entry = data["sections"][selected_section][0]["text_path"]

# Fix Windows path in JSON → convert to Docker path
        entry = entry.replace("A:\\Infosys\\outputs", "/app/outputs").replace("\\", "/")

        with open(entry, "r", encoding="utf-8") as f:
            text = f.read()
        st.text_area(selected_section, text, height=300)
else:
    st.info("No segmented sections found. Run 05_segment_reports.py first.")

# --- Tables ---
st.header("📊 Extracted Financial Tables")
if os.path.exists(TABLE_INDEX):
    idx_df = pd.read_csv(TABLE_INDEX)
    if len(idx_df) > 0:
        st.dataframe(idx_df)

        chosen = st.selectbox("Preview a table", idx_df["csv_path"])
        st.write(pd.read_csv(chosen))
    else:
        st.warning("No tables found in this document.")
else:
    st.info("Table index not found. Run 06_parse_tables.py.")

# --- Events ---
st.header("📌 Financial Event Extraction")
if os.path.exists(EVENTS):
    events_df = pd.read_csv(EVENTS)
    st.dataframe(events_df.head(50))
else:
    st.info("Events file not found. Run 03_event_extraction.py.")

# --- Verified Events ---
st.header("💹 Verified Events With Stock Data")
if os.path.exists(VERIFIED):
    vdf = pd.read_csv(VERIFIED)
    st.dataframe(vdf)
else:
    st.info("Verified events not found. Run 04_integrate_yfinance.py.")

# --- NER Errors ---
st.header("🚫 NER Misclassifications")
if os.path.exists(NER_ERRORS):
    err_df = pd.read_csv(NER_ERRORS)
    st.write(err_df.head(50))
else:
    st.info("NER error file missing.")

st.success("Dashboard Ready!")
