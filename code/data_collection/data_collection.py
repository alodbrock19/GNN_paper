# data_preprocessing.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import requests
import wikipedia as wp
from ta.momentum import rsi
from ta.trend import macd

# Configuration
#START_DATE = "2020-01-01"
#END_DATE = "2023-12-31"

START_DATE = "2023-01-01"

DATA_PATH = "../data"

def get_sp100_tickers():
    """
    Scrapes the S&P 100 ticker list from Wikipedia.
    """
    print("\n" + "="*60)
    print("STEP 1: Fetching S&P 100 ticker list from Wikipedia")
    print("="*60)
    
    print("  → Accessing Wikipedia page...")
    html = wp.page("S&P 100").html().encode("UTF-8")
    print("  ✓ Wikipedia page accessed successfully")
    
    print("  → Parsing HTML table...")
    tickers = pd.read_html(html)[2].set_index("Symbol")
    print(f"  ✓ Found {len(tickers)} tickers in initial list")
    
    print("  → Cleaning ticker list...")
    print("    - Dropping GOOG (keeping GOOGL)...")
    tickers = tickers.drop("GOOG")
    print("    - Updating GOOGL name to 'Alphabet'...")
    tickers.loc["GOOGL", "Name"] = "Alphabet"
    print("    - Renaming BRK.B to BRK-B...")
    tickers = tickers.rename(index={"BRK.B": "BRK-B"})
    tickers.reset_index(inplace=True)
    
    print(f"  → Saving ticker list to {DATA_PATH}/tickers_list.csv...")
    tickers.to_csv(f"{DATA_PATH}/tickers_list.csv", index=False)
    print(f"  ✓ Saved {len(tickers)} tickers to CSV")
    print(f"  ✓ Ticker list preparation complete!\n")

    return tickers

def download_stock_data(tickers, start_date):
    """
    Downloads historical data and returns it in Long (Tidy) Format.
    
    """
    
    print("\n" + "="*60)
    print("STEP 2: Downloading and processing stock data")
    print("="*60)
    print(f"  → Date range: {start_date}")
    print(f"  → Number of tickers to process: {len(tickers)}")
    print(f"  → Starting data download...")
    
    values = [
	yf.Ticker(stock).history(actions=False,start=START_DATE) for stock in tickers.Symbol]
    # List to store processed dataframes
    processed_list = []

    for idx, stock_values in enumerate(values):
        # work on a copy to avoid SettingWithCopy warnings
        df = stock_values.copy()

        # 1. Log Returns (The base feature)
        # We use Log Returns because they are additive and "more" stationary than prices
        df["SimpleReturn"] = df["Close"].pct_change()
        df["LogReturn"] = np.log(1 + df["Close"].pct_change())

        # 2. Rolling Cumulative Returns (Features)
        # Summing returns over a window is standard (Total return over 1 week, etc.)
        # Note: Do NOT multiply by the window size (e.g., *5) unless you specifically want the sum magnified.
        df["Ret_1W"] = df["LogReturn"].rolling(window=5).sum()
        df["Ret_2W"] = df["LogReturn"].rolling(window=10).sum()
        df["Ret_1M"] = df["LogReturn"].rolling(window=21).sum()
        df["Ret_2M"] = df["LogReturn"].rolling(window=42).sum()

        # 3. Rolling Volatility (The Normalizer)
        # We calculate std dev over a trailing window (e.g., 60 days) to capture RECENT volatility
        # This ensures NO future data is used.
        rolling_std = df["LogReturn"].rolling(window=60).std()

        # Handle division by zero just in case
        rolling_std = rolling_std.replace(0, np.nan)

        # 4. Apply Rolling Normalization (Z-Score)
        # This transforms the return into "How many sigmas was this move?"
        df["Norm_Ret"]    = df["LogReturn"] / rolling_std
        df["Ret_1W"] = df["Ret_1W"]    / rolling_std
        df["Ret_2W"] = df["Ret_2W"]    / rolling_std
        df["Ret_1M"] = df["Ret_1M"]    / rolling_std
        df["Ret_2M"] = df["Ret_2M"]    / rolling_std

        # 5. Other Indicators (RSI/MACD)
        # Ensure your rsi/macd functions are also strictly looking backwards
        # Normalizing RSI (0-100) to (0-1) is good practice
        if "RSI" not in df.columns: # Assuming you have the function
             df["RSI"] = rsi(df["Close"]) / 100.0
        
        if "MACD" not in df.columns:
            df["MACD"] = macd(df["Close"])
    
        # 6. Cleanup
        # Drop the first 60 rows because they will be NaN (due to the rolling window)
        df.dropna(inplace=True)

        # Drop raw price columns (GNNs usually struggle with raw prices)
        df.drop(columns=["Open", "High", "Low", "Volume"], errors='ignore', inplace=True)

        processed_list.append(df)

    # Recombine into one MultiIndex DataFrame (Symbol, Date)
    print("\n  → Combining all processed dataframes...")
    if processed_list:
        # keys=stocks.Symbol ensures the first index level is the Ticker
        final_values = pd.concat(processed_list, keys=tickers.Symbol)
        # Rename index levels for clarity
        final_values.index.names = ['Symbol', 'Date']

    print(f"\n  → Saving final data to {DATA_PATH}/values.csv...")
    final_values.to_csv(f"{DATA_PATH}/values.csv", index=True)
    print(f"  ✓ Data saved successfully!")
    print(f"  ✓ File size: {os.path.getsize(f'{DATA_PATH}/values.csv') / (1024*1024):.2f} MB")
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60 + "\n")
    

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING DATA COLLECTION PIPELINE")
    print("="*60)
    print(f"Configuration:")
    print(f"  → Start Date: {START_DATE}")
    print(f"  → Data Path: {DATA_PATH}")
    print("="*60)
    
    tickers = get_sp100_tickers()
    download_stock_data(tickers, START_DATE)