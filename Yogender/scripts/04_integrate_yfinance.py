# a:\Infosys\scripts\04_integrate_yfinance.py
import pandas as pd
import yfinance as yf
import re
from tqdm import tqdm
import datetime as dt

# === CONFIG ===
INPUT_PATH = r"A:\Infosys\outputs\financial_events_extracted.csv"
OUTPUT_PATH = r"A:\Infosys\outputs\financial_events_verified.csv"

# === Load extracted events ===
df = pd.read_csv(INPUT_PATH)
print(f"🔹 Loaded {len(df)} extracted events")


company_ticker_map = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "alphabet": "GOOG",
    "google": "GOOG",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "intel": "INTC",
    "ibm": "IBM",
    "adobe": "ADBE",
    "netflix": "NFLX"
}

# --- Function to detect which company a sentence refers to ---
def detect_company(text):
    text_lower = text.lower()
    for name in company_ticker_map.keys():
        if name in text_lower:
            return name
    return None

df["company"] = df["text"].apply(detect_company)
df = df.dropna(subset=["company"])
print(f"✅ {len(df)} events linked to known companies")

# --- Fetch stock data from Yahoo Finance ---
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        market_cap = info.get("marketCap")
        return current_price, prev_close, market_cap
    except Exception as e:
        return None, None, None

# Add financial metrics
df["ticker"] = df["company"].map(company_ticker_map)

data_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching financial data"):
    ticker = row["ticker"]
    current, prev, cap = fetch_stock_data(ticker)
    change = None
    if current and prev:
        try:
            change = round(((current - prev) / prev) * 100, 2)
        except ZeroDivisionError:
            change = None
    data_rows.append({
        "text": row["text"],
        "detected_event": row["detected_event"],
        "sentiment": row["sentiment"],
        "company": row["company"].title(),
        "ticker": ticker,
        "current_price": current,
        "previous_close": prev,
        "change_percent": change,
        "market_cap": cap
    })

final_df = pd.DataFrame(data_rows)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Verified events saved to: {OUTPUT_PATH}")

# --- Show top 10 verified events ---
print(final_df.head(10))
