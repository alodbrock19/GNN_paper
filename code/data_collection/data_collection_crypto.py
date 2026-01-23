import yfinance as yf
import pandas as pd
import numpy as np
import os
from ta.momentum import rsi
from ta.trend import macd

# Configuration
START_DATE = "2023-01-01"
DATA_PATH = "../data/raw" # Changed folder name to separate from stocks

# Ensure directory exists
os.makedirs(DATA_PATH, exist_ok=True)

def get_top_crypto_tickers():
    """
    Returns a DataFrame of top cryptocurrency tickers formatted for Yahoo Finance.
    We use a hardcoded list of liquid assets to ensure data quality and avoid scraping errors.
    """
    print("\n" + "="*60)
    print("STEP 1: Generating Top 50 Crypto Ticker List")
    print("="*60)

    # Top 50 Liquid Cryptocurrencies (Yahoo Finance Format: SYMBOL-USD)
    # Excluded Stablecoins (USDT, USDC) as their returns are near zero and irrelevant for prediction
    crypto_symbols = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "DOGE-USD", "AVAX-USD", "TRX-USD", "LINK-USD",
        "MATIC-USD", "DOT-USD", "LTC-USD", "BCH-USD", "SHIB-USD",
        "UNI7083-USD", "ATOM-USD", "XLM-USD", "OKB-USD", "ETC-USD",
        "FIL-USD", "HBAR-USD", "LDO-USD", "APT21794-USD", "ARB11841-USD",
        "NEAR-USD", "VET-USD", "QNT-USD", "MKR-USD", "GRT6719-USD",
        "AAVE-USD", "ALGO-USD", "EGLD-USD", "SAND-USD", "STX4847-USD",
        "EOS-USD", "XTZ-USD", "THETA-USD", "IMX10603-USD", "FTM-USD",
        "MANA-USD", "APE18876-USD", "AXS-USD", "FLOW-USD", "SNX-USD",
        "KCS-USD", "NEO-USD", "CHZ-USD", "CRV-USD", "BAT-USD"
    ]

    # Create DataFrame to match the structure expected by the pipeline
    tickers = pd.DataFrame({"Symbol": crypto_symbols})
    tickers["Name"] = tickers["Symbol"].str.replace("-USD", "", regex=False) # Clean name for reference

    print(f"  → Saving ticker list to {DATA_PATH}/tickers_list_crypto.csv...")
    tickers.to_csv(f"{DATA_PATH}/tickers_list_crypto.csv", index=False)
    print(f"  ✓ Saved {len(tickers)} crypto tickers to CSV")

    return tickers

def download_crypto_data(tickers, start_date):
    """
    Downloads historical crypto data and returns it in Long (Tidy) Format.
    """

    print("\n" + "="*60)
    print("STEP 2: Downloading and processing cryptocurrency data")
    print("="*60)
    print(f"  → Date range: {start_date} to Present")
    print(f"  → Number of assets: {len(tickers)}")
    print(f"  → Starting data download...")

    # Download all data at once (Faster than loop for yfinance)
    # Group_by='ticker' ensures we get a MultiIndex if we downloaded individually,
    # but for stability with the list comprehension below, we stick to the iterative approach
    # or handle the mass download carefully. Let's stick to your iterative approach for robustness.

    processed_list = []

    for idx, row in tickers.iterrows():
        symbol = row["Symbol"]

        try:
            # Download data
            df = yf.Ticker(symbol).history(start=start_date, interval="1d")

            if df.empty:
                print(f"  ⚠️ No data found for {symbol}, skipping...")
                continue

            # 1. Log Returns
            df["SimpleReturn"] = df["Close"].pct_change()
            df["LogReturn"] = np.log(1 + df["Close"].pct_change())

            # 2. Rolling Cumulative Returns
            # Note: Window=5 in crypto is 5 days (not 1 business week), but we keep
            # consistent lag steps for model compatibility.
            df["Ret_1W"] = df["LogReturn"].rolling(window=5).sum()
            df["Ret_2W"] = df["LogReturn"].rolling(window=10).sum()
            df["Ret_1M"] = df["LogReturn"].rolling(window=21).sum()
            df["Ret_2M"] = df["LogReturn"].rolling(window=42).sum()

            # 3. Rolling Volatility
            rolling_std = df["LogReturn"].rolling(window=60).std()
            rolling_std = rolling_std.replace(0, np.nan)

            # 4. Apply Rolling Normalization (Z-Score)
            df["Norm_Ret"]    = df["LogReturn"] / rolling_std
            df["Ret_1W"] = df["Ret_1W"]    / rolling_std
            df["Ret_2W"] = df["Ret_2W"]    / rolling_std
            df["Ret_1M"] = df["Ret_1M"]    / rolling_std
            df["Ret_2M"] = df["Ret_2M"]    / rolling_std

            # 5. Other Indicators (RSI/MACD)
            if "RSI" not in df.columns:
                 df["RSI"] = rsi(df["Close"]) / 100.0

            if "MACD" not in df.columns:
                df["MACD"] = macd(df["Close"])

            # 6. Intraday Features
            df["Intraday_Volatility"] = (df["High"] - df["Low"]) / df["Close"]
            df["Intraday_Momentum"] = (df["Close"] - df["Open"]) / df["Open"]

            # Shadow Ratios
            # Guard against NaN/Zero if Open/Close are identical
            df["Upper_Shadow_Ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
            df["Lower_Shadow_Ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

            # 7. Cleanup
            # Drop initial NaN rows due to rolling windows
            df.dropna(inplace=True)

            # NOTE: Dropping Volume might be detrimental for Crypto models,
            # but we do it to match the Stock script exactly.
            # Uncomment the next line if you want to KEEP Volume.
            df.drop(columns=["Open", "High", "Low", "Volume","Dividends","Stock Splits"], errors='ignore', inplace=True)

            # Save symbol reference
            df["Symbol"] = symbol

            processed_list.append(df)

        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")

    # Recombine into one MultiIndex DataFrame (Symbol, Date)
    print("\n  → Combining all processed dataframes...")
    if processed_list:
        final_values = pd.concat(processed_list)

        # Structure the index to match the Stock logic: (Symbol, Date)
        final_values = final_values.reset_index().set_index(['Symbol', 'Date'])

        # Sort index to ensure efficiency
        final_values.sort_index(inplace=True)

        print(f"\n  → Saving final data to {DATA_PATH}/values_crypto.csv...")
        final_values.to_csv(f"{DATA_PATH}/values_crypto.csv", index=True)
        print(f"  ✓ Data saved successfully!")
        print(f"  ✓ Total Rows: {len(final_values)}")
        print(f"  ✓ File size: {os.path.getsize(f'{DATA_PATH}/values_crypto.csv') / (1024*1024):.2f} MB")
    else:
        print("  ❌ No data was processed.")

    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING CRYPTO DATA COLLECTION PIPELINE")
    print("="*60)
    print(f"Configuration:")
    print(f"  → Start Date: {START_DATE}")
    print(f"  → Data Path: {DATA_PATH}")
    print("="*60)

    tickers = get_top_crypto_tickers()
    download_crypto_data(tickers, START_DATE)