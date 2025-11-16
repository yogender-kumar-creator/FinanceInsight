# a:\Infosys\scripts\05_segment_reports.py
import os, re, json
import fitz  # PyMuPDF
from collections import defaultdict

# === CONFIG ===
INPUT_PDF = r"A:\Infosys\sample_reports\10K_sample.pdf"   # change as needed
OUT_DIR = r"A:\Infosys\outputs\doc_segments"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Canonical SEC/Annual Report sections (regex, case-insensitive) ---
SECTION_PATTERNS = [
    (r"\bItem\s+7\.\s*Management'?s?\s+Discussion\s+and\s+Analysis\b", "MD&A"),
    (r"\bManagement'?s?\s+Discussion\s+and\s+Analysis\b", "MD&A"),
    (r"\bItem\s+1A\.\s*Risk\s+Factors\b", "Risk Factors"),
    (r"\bRisk\s+Factors\b", "Risk Factors"),
    (r"\bItem\s+8\.\s*Financial\s+Statements\b", "Financial Statements"),
    (r"\bConsolidated\s+Financial\s+Statements\b", "Financial Statements"),
    (r"\bItem\s+1\.\s*Business\b", "Business"),
    (r"\bItem\s+7A\.\s*Quantitative\s+and\s+Qualitative\b", "MD&A Supplement"),
    (r"\bItem\s+9\.\s*Changes\b", "Controls & Procedures"),
]

def find_candidates(text):
    hits = []
    for pat, label in SECTION_PATTERNS:
        for m in re.finditer(pat, text, flags=re.I):
            hits.append((m.start(), label))
    hits.sort(key=lambda x: x[0])
    return hits

def union_by_pages(spans):
    # Merge overlapping or consecutive page spans
    if not spans: return []
    spans.sort()
    merged = [spans[0]]
    for s in spans[1:]:
        if s[0] <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
        else:
            merged.append(s)
    return merged

def extract_pdf_sections(pdf_path):
    doc = fitz.open(pdf_path)
    # 1) Heading detection by font size (collect big titles)
    heading_candidates = []
    for i, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b: 
                continue
            # measure average font size
            sizes = []
            text_buf = []
            for l in b["lines"]:
                for s in l["spans"]:
                    sizes.append(s["size"])
                    text_buf.append(s["text"])
            if not sizes: 
                continue
            avg_size = sum(sizes)/len(sizes)
            text = " ".join(text_buf).strip()
            # heuristic: very short & larger font → heading candidate
            if len(text) > 0 and len(text) < 180 and avg_size >= 10.5:
                heading_candidates.append((i, avg_size, text))

    # 2) Regex anchors by whole-document text (coarse)
    full_text = []
    page_offsets = []
    acc = 0
    for i, page in enumerate(doc):
        t = page.get_text()
        full_text.append(t)
        page_offsets.append(acc)
        acc += len(t)
    joined = "".join(full_text)

    regex_hits = []
    for pat, label in SECTION_PATTERNS:
        for m in re.finditer(pat, joined, flags=re.I):
            # map char offset to page index approx
            # (binary search would be better; linear ok for typical sizes)
            char_idx = m.start()
            page_idx = 0
            for j in range(len(page_offsets)-1, -1, -1):
                if char_idx >= page_offsets[j]:
                    page_idx = j
                    break
            regex_hits.append((page_idx, label))
    regex_hits.sort()

    # 3) Combine signals → segment boundaries
    # Convert heading candidates that match common section titles into labels
    normalized_headings = []
    for (pg, size, txt) in heading_candidates:
        norm = txt.lower()
        label = None
        if re.search(r"risk\s+factors", norm): label = "Risk Factors"
        elif re.search(r"management.?s.*discussion.*analysis", norm): label = "MD&A"
        elif re.search(r"financial\s+statements", norm): label = "Financial Statements"
        elif re.search(r"\bitem\s+7a\b", norm): label = "MD&A Supplement"
        elif re.search(r"\bitem\s+7\b", norm): label = "MD&A"
        elif re.search(r"\bitem\s+8\b", norm): label = "Financial Statements"
        elif re.search(r"\bitem\s+1\b", norm): label = "Business"
        if label:
            normalized_headings.append((pg, label))

    candidates = sorted(set(regex_hits + normalized_headings))
    # If nothing found, fall back to whole doc "Body"
    if not candidates:
        return {"Body":[(0, len(doc)-1)]}, doc

    # Build ranges by next anchor - 1
    ranges = defaultdict(list)
    for idx, (pg, label) in enumerate(candidates):
        start = pg
        end = candidates[idx+1][0]-1 if idx+1 < len(candidates) else len(doc)-1
        if start <= end:
            ranges[label].append((start, end))

    # Merge overlapping spans per label
    for k, spans in list(ranges.items()):
        ranges[k] = union_by_pages(spans)

    return ranges, doc

def save_segments(pdf_path, out_dir):
    ranges, doc = extract_pdf_sections(pdf_path)
    meta = {"pdf": pdf_path, "sections": {}}
    for label, spans in ranges.items():
        meta["sections"][label] = []
        for (s, e) in spans:
            # save text
            buf = []
            for p in range(s, e+1):
                buf.append(doc[p].get_text())
            txt = "\n".join(buf)
            section_dir = os.path.join(out_dir, label.replace(" ", "_"))
            os.makedirs(section_dir, exist_ok=True)
            out_txt = os.path.join(section_dir, f"{os.path.basename(pdf_path)}_{s}-{e}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(txt)
            meta["sections"][label].append({"start_page": s, "end_page": e, "text_path": out_txt})
    json_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_sections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Segmentation saved:\n- JSON: {json_path}\n- Text per section under: {out_dir}")

if __name__ == "__main__":
    save_segments(INPUT_PDF, OUT_DIR)
