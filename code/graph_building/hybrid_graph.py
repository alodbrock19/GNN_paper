import pandas as pd
import numpy as np

TICKERS_PATH = "../data/tickers_list.csv"
FUNDAMENTALS_PATH = "../data/fundamentals.csv"
OUTPUT_PATH = "../data/hybrid_adj.npy"

print("=" * 60)
print("Loading stocks data...")
stocks = pd.read_csv(TICKERS_PATH, index_col=0)
print(f"Loaded {len(stocks)} stocks")
print(f"Stocks shape: {stocks.shape}")
print(f"Columns: {list(stocks.columns)}")
print()

def get_fundamental_graph():
    print("=" * 60)
    print("Building fundamental graph...")
    print("Loading fundamentals data...")
    fundamentals = pd.read_csv(FUNDAMENTALS_PATH, index_col=0)
    print(f"Fundamentals shape: {fundamentals.shape}")
    print(f"Number of features: {len(fundamentals.columns)}")
    print(f"Number of stocks: {len(fundamentals)}")
    print()
    
    print("Normalizing fundamentals data...")
    normalized_count = 0
    zero_std_count = 0
    for i, column in enumerate(fundamentals.columns, 1):
        col_mean = fundamentals[column].mean()
        col_std = fundamentals[column].std()
        if col_std != 0:  # Avoid division by zero for constant columns
            fundamentals[column] = (fundamentals[column] - col_mean) / col_std
            normalized_count += 1
        else:
            fundamentals[column] = 0
            zero_std_count += 1
        if i % 10 == 0 or i == len(fundamentals.columns):
            print(f"  Processed {i}/{len(fundamentals.columns)} columns...")
    
    print(f"Normalized {normalized_count} columns")
    print(f"Columns with zero std (set to 0): {zero_std_count}")
    print()
    
    print("Computing correlation matrix...")
    fundamentals_corr = fundamentals.transpose().corr(method="pearson")
    print(f"Correlation matrix shape: {fundamentals_corr.shape}")
    print(f"Correlation stats before removing self-correlation:")
    print(f"  Min: {fundamentals_corr.min().min():.4f}")
    print(f"  Max: {fundamentals_corr.max().max():.4f}")
    print(f"  Mean: {fundamentals_corr.values[np.triu_indices_from(fundamentals_corr.values, k=1)].mean():.4f}")
    print()
    
    print("Removing self-correlation (diagonal)...")
    fundamentals_corr = (fundamentals_corr - (fundamentals_corr == 1))  # Remove self-correlation
    print(f"Correlation stats after removing self-correlation:")
    print(f"  Min: {fundamentals_corr.min().min():.4f}")
    print(f"  Max: {fundamentals_corr.max().max():.4f}")
    print(f"  Mean: {fundamentals_corr.values[np.triu_indices_from(fundamentals_corr.values, k=1)].mean():.4f}")
    print()
    
    fundamentals_corr_np = fundamentals_corr.to_numpy()
    print(f"Converted to numpy array: shape {fundamentals_corr_np.shape}")
    print("=" * 60)
    print()
    return fundamentals_corr_np
    
def get_sector_graph():
    print("=" * 60)
    print("Building sector graph...")
    print(f"Number of stocks: {len(stocks)}")
    print(f"Unique sectors: {stocks['Sector'].nunique()}")
    print(f"Sector distribution:")
    print(stocks['Sector'].value_counts())
    print()
    
    print("Creating one-hot encoded sector matrix...")
    sector_dummies = pd.get_dummies(stocks[["Sector"]])
    print(f"Sector dummies shape: {sector_dummies.shape}")
    print()
    
    print("Computing sector correlation...")
    share_sector = sector_dummies.transpose().corr().to_numpy().astype(int) - np.eye(len(stocks), dtype=int)
    print(f"Sector graph shape: {share_sector.shape}")
    print(f"Sector connections: {share_sector.sum()} total connections")
    print(f"Average connections per stock: {share_sector.sum() / len(stocks):.2f}")
    print("=" * 60)
    print()
    return share_sector


def merge_graphs(fundamentals_corr_np, share_sector,corr_threshold=.7, sector_bonus=.05):
    print("=" * 60)
    print("Merging graphs...")
    print(f"Parameters:")
    print(f"  Correlation threshold: {corr_threshold}")
    print(f"  Sector bonus: {sector_bonus}")
    print()
    
    print("Step 1: Combining absolute fundamentals correlation with sector graph...")
    adj = abs(fundamentals_corr_np) + share_sector * sector_bonus
    print(f"  Combined adjacency matrix shape: {adj.shape}")
    print(f"  Stats before thresholding:")
    print(f"    Min: {adj.min():.4f}")
    print(f"    Max: {adj.max():.4f}")
    print(f"    Mean: {adj.mean():.4f}")
    print(f"    Non-zero elements: {np.count_nonzero(adj)}")
    print()
    
    print(f"Step 2: Applying threshold (>{corr_threshold})...")
    adj_before_threshold = adj.copy()
    adj = adj * (abs(adj) > corr_threshold)
    print(f"  Stats after thresholding:")
    print(f"    Min: {adj.min():.4f}")
    print(f"    Max: {adj.max():.4f}")
    print(f"    Mean: {adj.mean():.4f}")
    print(f"    Non-zero elements: {np.count_nonzero(adj)}")
    print(f"    Elements removed: {np.count_nonzero(adj_before_threshold) - np.count_nonzero(adj)}")
    print()
    
    print("Step 3: Normalizing by maximum value...")
    max_val = adj.max()
    print(f"  Maximum value before normalization: {max_val:.4f}")
    if max_val > 0:
        adj = adj / adj.max()
        print(f"  Stats after normalization:")
        print(f"    Min: {adj.min():.4f}")
        print(f"    Max: {adj.max():.4f}")
        print(f"    Mean: {adj.mean():.4f}")
    else:
        print("  Warning: Maximum value is 0, skipping normalization")
    print()
    
    print(f"Final adjacency matrix:")
    print(f"  Shape: {adj.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(adj)}")
    print(f"  Sparsity: {(1 - np.count_nonzero(adj) / adj.size) * 100:.2f}%")
    print("=" * 60)
    print()
    return adj

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HYBRID GRAPH CONSTRUCTION")
    print("=" * 60)
    print()
    
    fundamentals_corr_np = get_fundamental_graph()
    share_sector = get_sector_graph()
    adj = merge_graphs(fundamentals_corr_np, share_sector)
    
    print("=" * 60)
    print("Saving hybrid adjacency matrix...")
    np.save(OUTPUT_PATH, adj)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"File size: {adj.nbytes / 1024 / 1024:.2f} MB")
    print("=" * 60)
    print("\nGraph construction complete!")