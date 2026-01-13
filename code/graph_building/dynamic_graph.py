import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import torch

def build_dynamic_mscmg(close_prices, window_size=20, return_threshold=0.01, max_dist_threshold=5):
    """
    Constructs a sequence of dynamic graphs based on the MSCMG method.
    
    Parameters:
    -----------
    close_prices : pd.DataFrame
        DataFrame where index is Date and columns are Stock Symbols. Values are Close prices.
    window_size : int (Delta t)
        The rolling observation window length.
    return_threshold : float
        The percentage change required to label a movement as +1 or -1.
        (e.g., 0.01 means 1% change).
    max_dist_threshold : float
        The maximum Manhattan distance allowed to form an edge between two stocks.
        
    Returns:
    --------
    list of dict
        A list where each item represents a graph at time t containing:
        - 'date': The timestamp for this graph
        - 'edge_index': PyTorch geometric edge index (2, E)
        - 'adj_matrix': Dense adjacency matrix (N, N)
    """
    
    # ---------------------------------------------------------
    # 1. Daily Movement Labeling
    # ---------------------------------------------------------
    # Formula: return = p_t / p_{t-1} - 1
    returns = close_prices.pct_change()
    
    # Initialize all labels as 0 (Neutral)
    labels = pd.DataFrame(0, index=returns.index, columns=returns.columns)
    
    # Apply thresholding logic:
    # Upward trend (+1): return > positive threshold
    labels[returns > return_threshold] = 1
    
    # Downward trend (-1): return < negative threshold
    labels[returns < -return_threshold] = -1
    
    # Drop the first row (NaN from pct_change)
    labels = labels.iloc[1:]
    
    # ---------------------------------------------------------
    # 2. Dynamic Graph Construction Loop
    # ---------------------------------------------------------
    dynamic_graphs = []
    timestamps = labels.index
    num_stocks = len(labels.columns)
    
    print(f"Building dynamic graphs for {len(timestamps) - window_size} time steps...")
    
    # We start from 'window_size' because we need looking back history
    for t in range(window_size, len(timestamps)):
        current_date = timestamps[t]
        
        # Extract the sequence for the window [t - Delta_t + 1 : t]
        # Shape: (Window_Size, Num_Stocks)
        window_data = labels.iloc[t-window_size : t]
        
        # Transpose to (Num_Stocks, Window_Size) so each row is a stock's sequence
        stock_sequences = window_data.values.T 
        
        # -----------------------------------------------------
        # 3. Similarity Measurement (Manhattan Distance)
        # -----------------------------------------------------
        # We use scipy's cdist with 'cityblock' (Manhattan) metric.
        # This calculates Equation (6) efficiently for all pairs.
        # d_ij = sum(|l_s^i - l_s^j|)
        dist_matrix = cdist(stock_sequences, stock_sequences, metric='cityblock')
        
        # -----------------------------------------------------
        # 4. Edge Construction
        # -----------------------------------------------------
        # Edge exists if distance falls within range (distance <= threshold)
        # Note: Smaller distance = Higher similarity
        adj_matrix = np.where(dist_matrix <= max_dist_threshold, 1, 0)
        
        # Remove self-loops (diagonal)
        np.fill_diagonal(adj_matrix, 0)
        
        # Convert to PyTorch Geometric Edge Index (COO format)
        rows, cols = np.where(adj_matrix == 1)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        
        dynamic_graphs.append({
            'timestamp': current_date,
            'edge_index': edge_index,
            'adj_matrix': adj_matrix,
            # Optional: You might want the distances as edge weights later
            'distances': dist_matrix 
        })
        
    print(f"Completed! Generated {len(dynamic_graphs)} graphs.")
    return dynamic_graphs

# ==========================================
# Example Usage with your data
# ==========================================
if __name__ == "__main__":
    # Create dummy data for demonstration
    # In your case, use: df = pd.read_csv("data/values.csv").pivot(...)
    dates = pd.date_range(start="2023-01-01", periods=100)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    dummy_prices = pd.DataFrame(
        np.random.uniform(100, 200, size=(100, 5)), 
        index=dates, 
        columns=tickers
    )

    # Configuration based on the text
    DELTA_T = 10        # Window size (e.g., 10 days)
    THRESHOLD_PCT = 0.01 # 1% change threshold
    MAX_DIST = 4        # Allow at most 4 divergent days in the window

    graphs = build_dynamic_mscmg(
        dummy_prices, 
        window_size=DELTA_T, 
        return_threshold=THRESHOLD_PCT, 
        max_dist_threshold=MAX_DIST
    )

    # Check the first graph
    if graphs:
        g0 = graphs[0]
        print(f"\nGraph at {g0['timestamp']}:")
        print(f"Edges: {g0['edge_index'].shape[1]}")
        print("Adjacency Matrix Sample:\n", g0['adj_matrix'])