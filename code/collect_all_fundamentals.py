"""
Script to collect fundamental data for all tickers in tickers_list.csv
and save the results to a CSV file.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from get_fundamentals import get_rich_fundamentals

# Configuration
DATA_PATH = "./data"
TICKERS_FILE = f"{DATA_PATH}/tickers_list.csv"
OUTPUT_FILE = f"{DATA_PATH}/fundamentals_data.csv"
START_DATE = "2020-01-01"

def load_tickers():
    """Load ticker symbols from CSV file, handling the '0' issue."""
    try:
        # Read without header to avoid issues
        df = pd.read_csv(TICKERS_FILE, header=None)
        # Filter out any "0" values and get unique tickers
        tickers = df[0].astype(str).str.strip()
        tickers = tickers[tickers != '0'].unique().tolist()
        tickers = [t for t in tickers if t and len(t) > 0]  # Remove empty strings
        print(f"Loaded {len(tickers)} tickers from {TICKERS_FILE}")
        return tickers
    except Exception as e:
        print(f"Error loading tickers: {e}")
        return []

def collect_all_fundamentals(tickers, start_date=START_DATE):
    """
    Collect fundamental data for all tickers.
    
    Returns:
        pd.DataFrame: Combined DataFrame with all fundamental data
    """
    all_data = []
    successful_tickers = []
    failed_tickers = []
    
    print(f"\nCollecting fundamentals for {len(tickers)} tickers...")
    print(f"Start date: {start_date}\n")
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        try:
            # Get fundamentals for this ticker
            df = get_rich_fundamentals(ticker, start_date=start_date)
            
            if df is not None and not df.empty:
                # Add ticker column to identify which stock each row belongs to
                df['Ticker'] = ticker
                all_data.append(df)
                successful_tickers.append(ticker)
            else:
                print(f"  ⚠️  No data returned for {ticker}")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"  ❌ Error processing {ticker}: {e}")
            failed_tickers.append(ticker)
            continue
    
    # Combine all DataFrames
    if all_data:
        print(f"\nCombining data from {len(all_data)} successful tickers...")
        combined_df = pd.concat(all_data, ignore_index=False)
        
        # Reorder columns to have Ticker first
        cols = ['Ticker'] + [c for c in combined_df.columns if c != 'Ticker']
        combined_df = combined_df[cols]
        
        # Sort by date (index) and ticker
        # The index should be a DatetimeIndex from get_rich_fundamentals
        combined_df = combined_df.reset_index()
        date_col = combined_df.columns[0]  # First column is the date index
        combined_df = combined_df.sort_values([date_col, 'Ticker'])
        combined_df = combined_df.set_index(date_col)
        
        print(f"\n✅ Successfully collected data for {len(successful_tickers)} tickers")
        print(f"❌ Failed to collect data for {len(failed_tickers)} tickers")
        
        if failed_tickers:
            print(f"\nFailed tickers: {', '.join(failed_tickers)}")
        
        return combined_df
    else:
        print("\n❌ No data collected for any tickers!")
        return pd.DataFrame()

def main():
    """Main execution function."""
    # Ensure data directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory: {DATA_PATH}")
    
    # Load tickers
    tickers = load_tickers()
    if not tickers:
        print("No tickers to process. Exiting.")
        return
    
    # Collect fundamentals
    fundamentals_df = collect_all_fundamentals(tickers, START_DATE)
    
    if not fundamentals_df.empty:
        # Save to CSV
        print(f"\nSaving fundamentals data to {OUTPUT_FILE}...")
        fundamentals_df.to_csv(OUTPUT_FILE)
        
        print(f"\n✅ Successfully saved fundamentals data!")
        print(f"   File: {OUTPUT_FILE}")
        print(f"   Shape: {fundamentals_df.shape}")
        print(f"   Date range: {fundamentals_df.index.min()} to {fundamentals_df.index.max()}")
        print(f"   Unique tickers: {fundamentals_df['Ticker'].nunique()}")
        print(f"\nColumns: {', '.join(fundamentals_df.columns)}")
    else:
        print("\n❌ No data to save!")

if __name__ == "__main__":
    main()

