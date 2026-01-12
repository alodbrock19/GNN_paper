import pandas as pd
import yfinance as yf

TICKERS_PATH = "../../data/tickers_list.csv"
OUTPUT_FUNDAMENTALS_PATH = "../../data/fundamentals.csv"

def get_fundamental_values(tickers_path: str):
    """
    Retrieve fundamental values for a list of stock tickers.

    Parameters:
    - tickers_path: str
        Path to a file containing a list of stock ticker symbols.
    - fundamentals: pd.DataFrame
        A DataFrame containing fundamental data with a 'Symbol' column.

    Returns:
    - pd.DataFrame
        A DataFrame containing the fundamental values for the specified tickers.
    """
    print(f"Step 1: Loading tickers from {TICKERS_PATH}...")
    stocks = pd.read_csv(TICKERS_PATH)
    print(f"  ✓ Loaded {len(stocks)} tickers")
    
    print(f"\nStep 2: Fetching fundamental data for {len(stocks)} tickers...")
    fundamentals = []
    for i, stock in enumerate(stocks.Symbol, 1):
        print(f"  [{i}/{len(stocks)}] Fetching data for {stock}...", end="\r")
        fundamentals.append(yf.Ticker(stock).info)
    print(f"  ✓ Fetched fundamental data for all {len(stocks)} tickers")
    
    print(f"\nStep 3: Creating DataFrame and setting index...")
    fundamentals = pd.DataFrame(fundamentals).set_index("symbol")
    fundamentals.index = fundamentals.index.rename("Symbol")  # for consistency with the stocks dataframe
    print(f"  ✓ DataFrame created with shape {fundamentals.shape}")
    
    print(f"\nStep 4: Selecting fundamental columns and filling NaN values...")
    fundamentals = fundamentals[["marketCap", "trailingPE", "forwardPE", "priceToBook", "trailingEps", 
                             "forwardEps", "bookValue", "payoutRatio", "beta", "fiveYearAvgDividendYield", 
                             "52WeekChange", "averageVolume", "enterpriseToRevenue", "profitMargins"]].fillna(0)
    print(f"  ✓ Selected {len(fundamentals.columns)} columns")
    print(f"  ✓ Filled NaN values with 0")
    
    print(f"\nStep 5: Normalizing all columns using z-score normalization (standardization)...")
    for i, column in enumerate(fundamentals.columns, 1):
        col_mean = fundamentals[column].mean()
        col_std = fundamentals[column].std()
        if col_std != 0:  # Avoid division by zero for constant columns
            fundamentals[column] = (fundamentals[column] - col_mean) / col_std
            print(f"  [{i}/{len(fundamentals.columns)}] Normalized '{column}' (mean={col_mean:.4f}, std={col_std:.4f})")
        # If col_std == 0, the column is constant, so we set it to 0 (mean-centered)
        else:
            fundamentals[column] = 0
            print(f"  [{i}/{len(fundamentals.columns)}] '{column}' is constant (std=0), set to 0")
    print(f"  ✓ Normalized all {len(fundamentals.columns)} columns")
    
    print(f"\nStep 6: Saving fundamentals to {OUTPUT_FUNDAMENTALS_PATH}...")
    fundamentals.to_csv(OUTPUT_FUNDAMENTALS_PATH)
    print(f"  ✓ Saved fundamentals data successfully!")
    print(f"\n✓ Process completed! Final DataFrame shape: {fundamentals.shape}")
    
if __name__ == "__main__":
    get_fundamental_values(TICKERS_PATH)