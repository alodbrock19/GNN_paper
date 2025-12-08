# data_preprocessing.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import requests

# Configuration
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"

#START_DATE = "2023-01-01"
#END_DATE = "2025-10-31"

DATA_PATH = "./data"
OUTPUT_FILE = f"{DATA_PATH}/sp100_data.csv"

def get_sp100_tickers():
    """
    Scrapes the S&P 100 ticker list from Wikipedia.
    """
    print("Fetching S&P 100 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    try:
        # Add user-agent header to avoid 403 Forbidden error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        tables = pd.read_html(response.text)
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch data from Wikipedia (HTTP error): {e}")
    except Exception as e:
        raise Exception(f"Failed to parse data from Wikipedia: {e}")
    
    # Find the table with 'Symbol' column (more robust than hardcoding index)
    df = None
    for table in tables:
        if 'Symbol' in table.columns:
            df = table
            break
    
    if df is None:
        raise ValueError("Could not find table with 'Symbol' column in Wikipedia page")
    
    tickers = df['Symbol'].tolist()
    
    # Clean tickers (replace dots with dashes for Yahoo Finance, e.g., BRK.B -> BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]
    print(f"Found {len(tickers)} tickers.")
    return tickers

def download_stock_data(tickers):
    """
    Downloads historical data for the given tickers using yfinance.
    Returns a pivot table with Date as index and (Ticker, Feature) as columns.
    """
    print(f"Downloading data from {START_DATE} to {END_DATE}...")
    
    # Download all data at once (faster than loop)
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker', progress=False)
    except Exception as e:
        raise Exception(f"Failed to download data from yfinance: {e}")
    
    # If multi-level columns (Ticker, Feature), we want to stack them nicely
    # We want a DataFrame where columns are like 'AAPL_Close', 'AAPL_Volume', etc.
    df_list = []
    
    valid_tickers = []
    
    for ticker in tqdm(tickers):
        try:
            # Extract data for this ticker
            df_ticker = data[ticker].copy()
            
            # Check if DataFrame is empty
            if df_ticker.empty:
                print(f"Skipping {ticker} - no data available.")
                continue
            
            # Drop if mostly empty
            if df_ticker.isnull().mean().max() > 0.5:
                print(f"Skipping {ticker} due to missing data.")
                continue
                
            # Rename columns to {Ticker}_{Feature}
            df_ticker.columns = [f"{ticker}_{col}" for col in df_ticker.columns]
            df_list.append(df_ticker)
            valid_tickers.append(ticker)
            
        except KeyError:
            print(f"Could not find data for {ticker}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    # Check if we have any valid data
    if len(df_list) == 0:
        raise ValueError("No valid stock data was downloaded. All tickers failed.")
    
    # Combine all stocks into one large dataframe aligned by Date
    full_df = pd.concat(df_list, axis=1)
    
    # Forward fill missing data (standard finance practice)
    # Note: Only forward fill to avoid data leakage (bfill uses future values)
    full_df = full_df.ffill()
    # Only backward fill at the beginning if needed (first valid value forward)
    full_df = full_df.bfill()
    
    return full_df, valid_tickers

def main():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        
    try:
        tickers = get_sp100_tickers()
        df, valid_tickers = download_stock_data(tickers)
        
        # Validate that we have data before saving
        if df.empty:
            raise ValueError("Downloaded DataFrame is empty. Cannot save.")
        
        print("Saving processed data...")
        df.to_csv(OUTPUT_FILE)
        
        # Save list of tickers for the Graph Generator to use
        if len(valid_tickers) > 0:
            pd.Series(valid_tickers).to_csv(f"{DATA_PATH}/tickers_list.csv", index=False)
        else:
            print("Warning: No valid tickers to save.")
        
        print(f"Data saved to {OUTPUT_FILE}")
        print(f"Shape: {df.shape}")
        print(f"Successfully processed {len(valid_tickers)} out of {len(tickers)} tickers")
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()