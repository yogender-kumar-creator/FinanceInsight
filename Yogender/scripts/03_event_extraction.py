import re
import pandas as pd
from transformers import pipeline

# === CONFIG ===
DATA_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"
OUTPUT_PATH = r"A:\Infosys\outputs\financial_events_extracted.csv"

# === Load dataset ===
df = pd.read_csv(DATA_PATH)

# === Zero-Shot Classifier (for event type detection) ===
event_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# --- Define event labels ---
EVENT_LABELS = [
    "merger and acquisition",
    "initial public offering (IPO)",
    "stock split",
    "earnings report",
    "dividend announcement",
    "revenue growth",
    "profit decline",
    "market expansion",
    "partnership announcement"
]

# --- Simple Regex-based event phrase patterns ---
EVENT_PATTERNS = {
    "merger and acquisition": r"\b(merge|acquire|acquisition|buyout|takeover)\b",
    "IPO": r"\b(initial public offering|IPO|go public)\b",
    "stock split": r"\b(stock split|share split|reverse split)\b",
    "earnings report": r"\b(earnings|quarterly results|Q\d results|profit report)\b",
    "dividend announcement": r"\b(dividend|payout|shareholder return)\b",
    "revenue growth": r"\b(revenue (rise|increase|growth|jump))\b",
    "profit decline": r"\b(loss|profit (drop|decline|fall|decrease))\b",
    "market expansion": r"\b(expand|expansion|new market|global reach)\b",
    "partnership announcement": r"\b(partnership|collaboration|deal signed)\b"
}


def detect_event_type(text):
    """
    Uses both regex rules and zero-shot classification to detect event types.
    """
    text_lower = text.lower()
    detected_events = set()

    # --- 1. Rule-based matching ---
    for event, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, text_lower):
            detected_events.add(event)

    # --- 2. Model-based classification (optional, for more complex sentences) ---
    if not detected_events:
        res = event_classifier(text, EVENT_LABELS)
        if res['scores'][0] > 0.6:
            detected_events.add(res['labels'][0])

    return list(detected_events)


# --- Process dataset ---
extracted_rows = []

for idx, row in df.iterrows():
    text = row['text']
    events = detect_event_type(text)
    if events:
        for event in events:
            extracted_rows.append({
                "text": text,
                "detected_event": event,
                "sentiment": row['label']
            })

# --- Save results ---
events_df = pd.DataFrame(extracted_rows)
events_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Extracted {len(events_df)} financial events.")
print(f"💾 Saved to {OUTPUT_PATH}")

# --- Display sample output ---
print(events_df.head(10))
