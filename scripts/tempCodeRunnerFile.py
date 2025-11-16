# a:\Infosys\scripts\06_parse_tables.py
import os
import pdfplumber
import pandas as pd
import re

# === CONFIG ===
INPUT_PDF = r"A:\Infosys\sample_reports\10K_sample.pdf"
OUTPUT_DIR = r"A:\Infosys\outputs\tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_num(x):
    """Convert accounting-style numbers to floats."""
    if x is None:
        return None
    s = str(x).replace(",", "").strip()

    # Handle empty or dash
    if s in ["", "-", "—", "–"]:
        return None

    # Handle parentheses for negative values
    if s.startswith("(") and s.endswith(")"):
        try:
            return -float(s[1:-1])
        except:
            return None

    # Try normal float
    try:
        return float(s)
    except:
        return None

def classify_table(text):
    """Simple keyword-based classification."""
    t = text.lower()
    if "balance sheet" in t:
        return "Balance Sheet"
    if "income statement" in t:
        return "Income Statement"
    if "cash flow" in t:
        return "Cash Flow Statement"
    return "Other"

def extract_tables(pdf_path):
    index = []

    with pdfplumber.open(pdf_path) as pdf:
        for p_no, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()

            if not tables:
                continue

            label = classify_table(text)

            for t_no, table in enumerate(tables):
                df = pd.DataFrame(table)

                # First row is headers
                df.columns = df.iloc[0]
                df = df.drop(0).reset_index(drop=True)

                # Clean numbers
                for col in df.columns:
                    df[col] = df[col].apply(clean_num).fillna(df[col])

                out_csv = os.path.join(
                    OUTPUT_DIR,
                    f"table_p{p_no}_t{t_no}_{label.replace(' ', '_')}.csv"
                )

                df.to_csv(out_csv, index=False)
                index.append([p_no, t_no, label, out_csv])

                print(f"✅ Saved table: {out_csv}")

    index_df = pd.DataFrame(index, columns=["page", "table_no", "type", "csv_path"])
    index_df.to_csv(os.path.join(OUTPUT_DIR, "tables_index.csv"), index=False)
    print("\n📌 All tables indexed at:", os.path.join(OUTPUT_DIR, "tables_index.csv"))

if __name__ == "__main__":
    extract_tables(INPUT_PDF)
