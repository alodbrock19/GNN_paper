import pandas as pd
import numpy as np
import networkx as nx

DATA_PATH = "../data/raw/values_crypto.csv"
OUTPUT_GRAPH_PATH = "../data/pearson_adj_crypto.npy"
THRESHOLD = 0.4

def build_pearson_correlation_graph(values: pd.DataFrame, threshold: float = 0.5):
    """
    Build a Pearson Correlation Graph from stock return data.

    Parameters:
    - values: pd.DataFrame
        A MultiIndex DataFrame with stock symbols as the first index level and dates as the second.
    - threshold: float
        The correlation threshold to create edges between nodes.
    Returns:
        Saves the adjacency matrix of the correlation graph as a .npy file.
    """
    print("=" * 60)
    print("Building Pearson Correlation Graph")
    print("=" * 60)
    print(f"Input DataFrame shape: {values.shape}")
    print(f"Threshold: {threshold}")
    
    print("\n[Step 1] Filtering columns: Symbol, Date, LogReturn")
    filtered_values = values[['Symbol', 'Date', 'LogReturn']]
    print(f"  Filtered DataFrame shape: {filtered_values.shape}")
    filtered_values = filtered_values.copy()  # To avoid SettingWithCopyWarning
    print("\n[Step 2] Converting Date to datetime")
    filtered_values['Date'] = pd.to_datetime(filtered_values['Date'], utc=True)
    print(f"  Date conversion completed")

    print("\n[Step 3] Normalizing dates (removing time and timezone info)")
    filtered_values['Date'] = filtered_values['Date'].dt.date
    print(f"  Date normalization completed")

    print("\n[Step 4] Dropping duplicates")
    initial_count = len(filtered_values)
    filtered_values = filtered_values.drop_duplicates(subset=['Date', 'Symbol'])
    final_count = len(filtered_values)
    duplicates_removed = initial_count - final_count
    print(f"  Initial rows: {initial_count}")
    print(f"  Final rows: {final_count}")
    print(f"  Duplicates removed: {duplicates_removed}")
    
    print("\n[Step 5] Pivoting DataFrame")
    pivot_df = filtered_values.pivot(index='Date', columns='Symbol', values='LogReturn')
    print(f"  Pivot DataFrame shape: {pivot_df.shape}")
    print(f"  Number of unique dates: {pivot_df.shape[0]}")
    print(f"  Number of unique symbols: {pivot_df.shape[1]}")

    print("\n[Step 6] Computing correlation matrix (Pearson method)")
    corr_matrix = pivot_df.corr(method='pearson')
    print(f"  Correlation matrix shape: {corr_matrix.shape}")
    
    print("\n[Step 7] Removing self-correlation (diagonal)")
    corr_matrix = (corr_matrix - (corr_matrix == 1))  # Remove self-correlation
    print(f"  Self-correlation removed")
    
    print("\n[Step 8] Converting correlation matrix to numpy array")
    corr_np = corr_matrix.to_numpy()
    print(f"  Numpy array shape: {corr_np.shape}")
    
    print(f"\n[Step 9] Creating adjacency matrix with threshold {threshold}")
    adj_fundamentals_corr = (corr_np * (abs(corr_np) > threshold).astype(int))
    num_edges = np.sum(adj_fundamentals_corr != 0) // 2  # Divide by 2 for undirected graph
    print(f"  Adjacency matrix shape: {adj_fundamentals_corr.shape}")
    print(f"  Number of edges: {num_edges}")

    print("\n[Step 10] Creating NetworkX graph from adjacency matrix")
    corr_graph = nx.from_numpy_array(adj_fundamentals_corr)
    print(f"  Graph created with {corr_graph.number_of_nodes()} nodes and {corr_graph.number_of_edges()} edges")
    
    print("\n[Step 11] Relabeling nodes with stock symbols")
    corr_graph = nx.relabel_nodes(corr_graph, dict(enumerate(corr_matrix.index)))
    print(f"  Nodes relabeled successfully")
    
    print(f"\n[Step 12] Saving adjacency matrix to {OUTPUT_GRAPH_PATH}")
    np.save(OUTPUT_GRAPH_PATH, adj_fundamentals_corr)
    print(f"  Adjacency matrix saved successfully!")
    print(f"  Saved array shape: {adj_fundamentals_corr.shape}")
    print("=" * 60)



if __name__ == "__main__":
    print(f"Loading data from {DATA_PATH}")
    values = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully. Shape: {values.shape}\n")
    
    build_pearson_correlation_graph(values, THRESHOLD)
    
    print("\nProcess completed!")