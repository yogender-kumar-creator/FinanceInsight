# a:\Infosys\scripts\01_preprocess_fiqa.py
import os
import re
import pandas as pd
from tqdm import tqdm

# === CONFIG - change if needed ===
RAW_DIR = r"A:\Infosys\FinancialPhraseBank"   # folder containing all Sentences_*.txt files
OUTPUT_PATH = r"A:\Infosys\data\processed\preprocessed_fiqa.csv"

# small set of stopwords (we don't rely on NLTK downloads here)
STOPWORDS = {
    'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'on', 'for', 'this', 'that',
    'it', 'was', 'as', 'by', 'with', 'at', 'from', 'be', 'are', 'or'
}

def read_file_lines(path):
    """
    Robust file reader: tries utf-8, then latin-1, then fallback with errors='replace'.
    Returns list of lines.
    """
    # try utf-8
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        pass
    # try latin-1
    try:
        with open(path, "r", encoding="latin-1") as f:
            return f.readlines()
    except Exception:
        pass
    # final fallback: replace errors
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()

def collect_files(root_dir):
    """
    Collect all .txt files in folder (non-recursive by default).
    If your files are nested, set recursive=True or adjust.
    """
    files = []
    for entry in os.listdir(root_dir):
        full = os.path.join(root_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(".txt"):
            files.append(full)
    return sorted(files)

def extract_label_and_text(line):
    """
    From a line like:
      "Some sentence here ...@positive"
    returns ("Some sentence here ...", "positive")
    If no label present returns (None, None)
    """
    if not line or not line.strip():
        return None, None
    # we expect label token like @positive, @negative, @neutral at end, but be flexible
    m = re.search(r'@(?P<label>positive|negative|neutral)\s*$', line.strip(), flags=re.I)
    if not m:
        # also allow inline like "... something @positive" anywhere
        m = re.search(r'@(?P<label>positive|negative|neutral)\b', line, flags=re.I)
        if not m:
            return None, None
    label = m.group("label").lower()
    # remove the label token from text
    text = re.sub(r'@(?P<label>positive|negative|neutral)\b', '', line, flags=re.I).strip()
    # normalize spaces and strip punctuation-only lines
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) == 0:
        return None, None
    return text, label

def clean_text(text):
    # lowercase (optional)
    text = text.strip()
    # remove URLs
    text = re.sub(r'http\S+', '', text)
    # remove weird control chars
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    # remove excessive non-printable chars except punctuation we might want to keep
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  # optional: convert non-ascii to space
    # keep common punctuation, remove other symbols
    text = re.sub(r'[^A-Za-z0-9\s$%.,:\-\'()]', ' ', text)
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess():
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")

    files = collect_files(RAW_DIR)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {RAW_DIR}. Please check path and files.")

    rows = []
    total_lines = 0
    for filepath in files:
        fname = os.path.basename(filepath)
        try:
            lines = read_file_lines(filepath)
        except Exception as e:
            print(f"[WARN] Could not read {filepath}: {e}")
            continue
        for raw_line in lines:
            total_lines += 1
            text, label = extract_label_and_text(raw_line)
            if text is None or label is None:
                continue
            cleaned = clean_text(text)
            if len(cleaned) < 3:
                continue
            rows.append((cleaned, label))
    df = pd.DataFrame(rows, columns=["text", "label"])
    # drop duplicates and empty
    df.drop_duplicates(inplace=True)
    df = df[df['text'].str.strip().astype(bool)]
    print(f"Files processed: {len(files)}")
    print(f"Raw lines read: {total_lines}")
    print(f"Usable sentences collected: {len(df)}")
    # ensure output folder exists
    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✅ Saved preprocessed file to: {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
