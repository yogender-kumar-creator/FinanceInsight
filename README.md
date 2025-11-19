<p align="center">
  <img src="logo.png" alt="FinanceInsight Logo" width="160">
</p>

<h1 align="center">📊 FinanceInsight — Financial Document Intelligence Platform</h1>

<p align="center">
A complete end-to-end AI system that reads, segments, extracts events, parses tables, and visualizes insights from financial reports using NLP, ML, and Dockerized deployment.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-ff4b4b?logo=streamlit">
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker">
  <img src="https://img.shields.io/github/stars/yogender-kumar-creator/FinanceInsight?style=social">
  <img src="https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render">
</p>

---

<p align="center">
  <img src="demo.gif" width="800" alt="Demo GIF">
</p>

---

## 🚀 Overview

**FinanceInsight** is an AI-powered financial analysis pipeline that processes Annual Reports, Investor Presentations, 10-K/10-Q reports, and corporate disclosures.  
The system automatically:

- Converts PDF reports into structured sections  
- Extracts tables and financial metrics  
- Detects company events & entities  
- Performs sentiment analysis using FinBERT  
- Visualizes everything inside a clean **Streamlit Dashboard**  
- Is fully packaged & deployable using **Docker** and **Render Cloud**  

---

## 🔥 Key Features

### **📘 1. PDF Segmentation**
Breaks the report into logical sections like:
- Management Discussion
- Risk Factors  
- Financial Performance  
- Notes & Statements  

### **📊 2. Table Extraction**
Parses financial tables using:
- `pdfplumber`  
- Automatic type classification (Balance Sheet, P&L, Cash Flow)

### **🧠 3. Entity & Event Extraction**
Uses Transformer-based NLP to detect:
- Company names  
- Products  
- Events (profit drop, acquisition, revenue growth, etc.)

### **📈 4. Interactive Dashboard**
Built in Streamlit, offering:
- Document viewer  
- Segmented content explorer  
- Tables viewer  
- Financial sentiment graph  
- Event timeline  

### **🐳 5. Docker Deployment**
One-command deployment using:

```
docker build -t financial-dashboard .
docker run -p 8501:8501 financial-dashboard
```

### **🌐 6. Cloud Deployment (Render)**
Upload code → Select Docker → Deploy → Done.

---

## 🗂️ Project Structure

```
📦 FinanceInsight
│
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── streamlit_app.py
│
├── scripts/
│   ├── 01_preprocess_fiqa.py
│   ├── 02_eda_fiqa.py
│   ├── 03_event_extraction.py
│   ├── 05_segment_reports.py
│   ├── 06_parse_tables.py
│   ├── 07_eval_pipeline.py
│
├── data/
│   └── processed/
│
├── outputs/
│   ├── doc_segments/
│   ├── tables/
│   └── events/
│
└── README.md
```

---

## 🛠️ Installation

### **Clone the repository**
```
git clone https://github.com/yogender-kumar-creator/FinanceInsight.git
cd FinanceInsight
```

### **Install environment**
```
pip install -r app/requirements.txt
```

### **Run Streamlit app**
```
streamlit run app/streamlit_app.py
```

---

## 🐳 Docker Setup

### **Build Docker Image**
```
docker build -t financial-dashboard ./app
```

### **Run the Container**
```
docker run -p 8501:8501 financial-dashboard
```

Visit:  
👉 **http://localhost:8501**

---

## 🌐 Deploy on Render

1. Push to GitHub  
2. Go to **Render.com → New Web Service**  
3. Select repo  
4. Choose **Docker**  
5. Done 🎉  

---

## 📄 Supported Inputs

- ✔ PDF (Selectable text)  
- ✔ Scanned PDFs (if OCR-enabled)  
- ✔ Financial reports (10-K, 10-Q, AR, IP)  
- ✔ Company Annual Reports  

---

## 💡 Future Improvements

- OCR integration for scanned PDFs  
- Advanced entity linking with Wikidata  
- Financial anomaly detection  
- Multi-company comparison dashboard  

---

## 🙌 Acknowledgements

- HuggingFace Transformers  
- Streamlit  
- Docker  
- pandas & pdfplumber  
- FinBERT (ProsusAI)

---

## ⭐ Give this project a star!

If this project helped you, please ⭐ **star the repository** to support development.

<p align="center">
  <img src="https://img.shields.io/github/stars/yogender-kumar-creator/FinanceInsight?style=social" />
</p>
