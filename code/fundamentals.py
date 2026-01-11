import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    # --- STEP 1: DEFINE PARAMETERS & TICKERS ---
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    correlation_threshold = 0.5  # Only link stocks with correlation > 0.5

    # A representative subset of S&P 100 (Sector diversified)
    tickers = [
        "AAPL", "MSFT", "GOOG", "NVDA", "AMZN", "META", "TSLA",  # Tech
        "JPM", "BAC", "V", "MA",                                 # Finance
        "JNJ", "PFE", "UNH", "LLY",                              # Healthcare
        "XOM", "CVX",                                            # Energy
        "WMT", "HD", "PG", "KO"                                  # Consumer
    ]

    print(f"Downloading data for {len(tickers)} companies...")

    # --- STEP 2: DOWNLOAD & PROCESS DATA ---
    # We use 'Adj Close' to account for dividends and splits
    # When downloading multiple tickers, yfinance returns MultiIndex columns (ticker, field)
    downloaded = yf.download(tickers, start=start_date, end=end_date, progress=False)
    data = downloaded.xs('Adj Close', level=1, axis=1)

    # CRITICAL STEP: Calculate Daily Returns
    # We cannot correlate raw prices (non-stationary). We must correlate % change.
    returns = data.pct_change().dropna()

    print(f"\nData Shape: {returns.shape[0]} days x {returns.shape[1]} stocks")

    # --- STEP 3: COMPUTE DYNAMIC CORRELATION ---
    # This computes the Pearson correlation matrix (-1 to 1)
    corr_matrix = returns.corr()

    # --- STEP 4: BUILD ADJACENCY MATRIX ---
    # Create a binary matrix: 1 if connected, 0 if not
    # We use absolute value because strong negative correlation is also a valid link
    adj_matrix = (np.abs(corr_matrix) > correlation_threshold).astype(int)

    # Remove self-loops (Connection to self is always 1, but useless for GNN)
    np.fill_diagonal(adj_matrix.values, 0)

    # --- STEP 5: VISUALIZE THE GRAPH ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(adj_matrix, cmap="Blues", linewidths=0.5, cbar=False)
    plt.title(f"Adjacency Matrix (Threshold > {correlation_threshold})")
    plt.xlabel("Target Node")
    plt.ylabel("Source Node")
    plt.savefig("adjacency_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- STEP 6: NETWORK STATISTICS ---
    num_edges = adj_matrix.sum().sum()
    density = num_edges / (len(tickers) * (len(tickers) - 1))
    print(f"\nGraph Statistics:")
    print(f"Total Edges: {num_edges}")
    print(f"Graph Density: {density:.2%} (Target is usually 5-20%)")

    # Example: Who is connected to Apple?
    aapl_neighbors = adj_matrix['AAPL'][adj_matrix['AAPL'] == 1].index.tolist()
    print(f"\nStocks connected to AAPL: {aapl_neighbors}")

if __name__ == "__main__":
    main()
