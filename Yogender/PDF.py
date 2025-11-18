from fpdf import FPDF
import os

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Add a Unicode font
pdf.add_font("DejaVu", "", r"D:\Languages\python\Lib\site-packages\fpdf\fonts\DejaVuSans.ttf", uni=True)

sections = {
    "Item 1. Business": "The company manufactures electronic products and provides related services.",
    "Item 1A. Risk Factors": "Risk factors include competition, supply chain issues, and global economic conditions.",
    "Item 7. Management's Discussion and Analysis": "Revenue increased by 12 percent due to strong product demand.",
    "Item 7A. Quantitative and Qualitative Disclosures": "Interest-rate fluctuations could impact future cash flows.",
    "Item 8. Financial Statements": """Balance Sheet:
Total Assets: $120,000
Total Liabilities: $60,000
Shareholder Equity: $60,000

Income Statement:
Revenue: $200,000
Net Income: $80,000
"""
}

for title, body in sections.items():
    pdf.add_page()
    pdf.set_font("DejaVu", "", 16)
    pdf.multi_cell(0, 10, title)
    pdf.ln(5)
    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(0, 8, body * 3)  # repeat text to make it longer

out_dir = r"A:\Infosys\sample_reports"
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "10K_sample.pdf")
pdf.output(pdf_path)

print("✅ Dummy 10-K-style PDF created at:", pdf_path)
