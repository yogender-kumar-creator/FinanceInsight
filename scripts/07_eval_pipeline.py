import os, time, json, glob
import pandas as pd

# === PATHS ===
SECTIONS_JSON = r"A:\Infosys\outputs\doc_segments\10K_sample_sections.json"
TABLES_INDEX = r"A:\Infosys\outputs\tables\tables_index.csv"
NER_ERRORS = r"A:\Infosys\outputs\eda\fiqa_errors.csv"  # optional
EVENTS = r"A:\Infosys\outputs\financial_events_extracted.csv"
VERIFIED = r"A:\Infosys\outputs\financial_events_verified.csv"

OUT_JSON = r"A:\Infosys\outputs\evaluation_summary.json"

def check_exists(path):
    return os.path.exists(path)

def count_rows(path):
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path))
    except:
        return 0

def main():
    t0 = time.time()

    summary = {}

    # --- 1. Segmentation Evaluation ---
    summary["segmentation"] = {
        "file_exists": check_exists(SECTIONS_JSON),
        "sections_found": 0,
        "section_names": []
    }
    if check_exists(SECTIONS_JSON):
        meta = json.load(open(SECTIONS_JSON))
        summary["segmentation"]["sections_found"] = len(meta.get("sections", {}))
        summary["segmentation"]["section_names"] = list(meta.get("sections", {}).keys())

    # --- 2. Table Parsing ---
    summary["tables"] = {
        "file_exists": check_exists(TABLES_INDEX),
        "tables_extracted": count_rows(TABLES_INDEX)
    }
    if check_exists(TABLES_INDEX):
        idx = pd.read_csv(TABLES_INDEX)
        summary["tables"]["by_type"] = idx["type"].value_counts().to_dict()

    # --- 3. Event Extraction ---
    summary["event_extraction"] = {
        "events_found": count_rows(EVENTS),
        "verified_events": count_rows(VERIFIED)
    }
    if check_exists(VERIFIED):
        v = pd.read_csv(VERIFIED)
        summary["event_extraction"]["linked_companies"] = int(v["ticker"].notna().sum())

    # --- 4. NER Errors (optional) ---
    summary["ner_error_analysis"] = {
        "file_exists": check_exists(NER_ERRORS),
        "errors": count_rows(NER_ERRORS)
    }

    summary["runtime_seconds"] = round(time.time() - t0, 2)

    # Save summary JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Evaluation Summary Generated:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
