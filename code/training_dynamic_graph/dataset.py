import os.path as osp
from typing import Callable
from functools import partial

import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse

# --- 1. The MSCMG Graph Generator ---
def mscmg_adjacency_calculator(window_data_numpy, graph_return_threshold=0.01, max_dist_threshold=5):
    """
    Calculates the MSCMG Adjacency Matrix for a SINGLE time window.
    window_data_numpy: (Num_Nodes, Window_Size, Features)
    graph_return_threshold: Threshold for discretizing returns during graph construction
    """
    # 1. Extract Close Prices (Assuming Feature 0 is Close Price)
    prices = window_data_numpy[:, :, 0] 
    
    # 2. Compute Returns
    # Avoid division by zero with epsilon
    price_diffs = prices[:, 1:] - prices[:, :-1]
    returns = price_diffs / (prices[:, :-1] + 1e-9)
    
    # 3. Apply Movement Labels (+1, -1, 0)
    labels = np.zeros_like(returns)
    labels[returns > graph_return_threshold] = 1
    labels[returns < -graph_return_threshold] = -1
    
    # 4. Calculate Manhattan Similarity (Cityblock)
    # Shape: (Num_Stocks, Num_Stocks)
    dist_matrix = cdist(labels, labels, metric='cityblock')
    
    # 5. Thresholding (Create Edges)
    adj_matrix = np.where(dist_matrix <= max_dist_threshold, 1, 0)
    
    # 6. Remove Self-Loops
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix.astype(int)

# --- 2. The Dataset Class ---
class InMemoryDynamicSP100(InMemoryDataset):
    def __init__(
        self, 
        root: str = "./data/", 
        values_file_name: str = "values.csv", 
        # We don't need adj_file_name anymore, but keeping arg for compatibility
        adj_file_name: str = None, 
        past_window: int = 25, 
        future_window: int = 1, 
        graph_return_threshold: float = 0.01,
        label_return_threshold: float = 0.01,
        max_dist_threshold: float = 5,
        force_process: bool = False,
        transform: Callable = None,
        pre_transform: Callable = None
    ):
        self.values_file_name = values_file_name
        self.past_window = past_window
        self.future_window = future_window
        self.label_return_threshold = label_return_threshold
        
        # Bake hyperparameters into the calculator
        self.adj_calculator = partial(
            mscmg_adjacency_calculator, 
            graph_return_threshold=graph_return_threshold, 
            max_dist_threshold=max_dist_threshold
        )
        
        # Handle Force Reload
        processed_path = osp.join(root, "processed", "sp100_dynamic_mscmg.pt")
        if force_process and osp.exists(processed_path):
            import os
            os.remove(processed_path)
            print("Force reload: Deleted old processed file.")

        super().__init__(root, transform, pre_transform)
        
        # Load data from RAM
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        # We only strictly require the CSV now
        return [self.values_file_name]

    @property
    def processed_file_names(self) -> list[str]:
        return ['sp100_dynamic_mscmg.pt']

    def download(self) -> None:
        pass

    def _load_data_from_csv(self):
        """
        Internal helper to load CSV and reshape to (Nodes, Time, Features).
        FIXED: Uses pivot + reindex to guarantee perfect shape (Nodes, Time, Feat).
        """
        path = self.raw_paths[0]
        print(f"Loading raw data from: {path}")
        
        # 1. Read CSV
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 2. Extract Dimensions & Sorting
        # We need unique sorted lists to enforce the grid structure
        unique_symbols = sorted(df['Symbol'].unique())
        unique_dates = sorted(df['Date'].unique())
        
        num_nodes = len(unique_symbols)
        num_times = len(unique_dates)
        
        # Identify feature columns (everything that isn't Symbol or Date)
        # Note: We preserve the column order from the CSV
        feature_cols = [c for c in df.columns if c not in ['Symbol', 'Date']]
        num_features = len(feature_cols)
        
        print(f"Detected: {num_nodes} Stocks, {num_times} Days, {num_features} Features")
        
        # 3. Pivot and Reindex (The Robust Fix)
        # We process each feature separately to handle the 2D matrix (Time x Stocks)
        feature_matrices = []
        
        for col in feature_cols:
            # Step A: Pivot to create a grid (Index=Date, Columns=Symbol)
            # This automatically aligns data. Missing data becomes NaN.
            pivot = df.pivot(index='Date', columns='Symbol', values=col)
            
            # Step B: Reindex to ensure NO dates or symbols are missing
            # If a stock is missing entirely for a date, reindex inserts a NaN row/col
            pivot = pivot.reindex(index=unique_dates, columns=unique_symbols)
            
            # Step C: Smart Filling (Forward Fill -> Backward Fill -> Zero)
            # ffill: Uses yesterday's price for today (standard for stocks)
            # bfill: Handles missing start data
            pivot = pivot.ffill().bfill().fillna(0.0)
            
            # Step D: Transpose to (Nodes, Time)
            # Current shape is (Time, Nodes), we want (Nodes, Time)
            # because we want [Stock_i, Time_0...T]
            feature_matrices.append(pivot.values.T)
            
        # 4. Stack into Final Tensor
        # We have a list of [ (Nodes, Time), (Nodes, Time), ... ]
        # Stacking along axis 2 gives (Nodes, Time, Features)
        x_all = np.stack(feature_matrices, axis=-1)
        
        print(f"Successfully reshaped data to: {x_all.shape}")
        
        return torch.tensor(x_all, dtype=torch.float)

    def process(self) -> None:
        print("Processing Dynamic Graphs (MSCMG)...")
        
        # 1. Load Raw Data directly from CSV
        # Shape: (Num_Nodes, Total_Time, Features)
        x_all = self._load_data_from_csv()
        
        data_list = []
        num_samples = x_all.shape[1] - self.past_window - self.future_window
        
        for idx in range(num_samples):
            # --- A. Prepare Window Data ---
            # Shape: (Nodes, Window_Size, Features)
            x_window = x_all[:, idx : idx + self.past_window, :]
            
            # --- B. Compute Dynamic Graph ---
            window_numpy = x_window.numpy() 
            adj_matrix = self.adj_calculator(window_numpy)
            
            # Convert to PyTorch Sparse Format
            adj_tensor = torch.from_numpy(adj_matrix).float()
            edge_index, edge_weight = dense_to_sparse(adj_tensor)
            
            # --- C. Prepare Label ---
            # Get Future Return and convert to 3-class labels (Downward, Neutral, Upward)
            # Feature 0 is usually Close Price (or whatever you used for return calc)
            # We want the return at (t + future_window) vs (t)
            # Or simplified: The feature vector at the target step
            
            # Here: We grab the Return of the NEXT day relative to the window end
            # We calculate return on the fly or grab a specific target column if it exists.
            # Assuming we predict movement based on Close Price (Index 0):
            current_close = x_all[:, idx + self.past_window - 1, 0]
            future_close = x_all[:, idx + self.past_window + self.future_window - 1, 0]
            
            # Continuous Return: (Future - Current) / Current
            y_return = (future_close - current_close) / (current_close + 1e-9)
            
            # Convert to 3-class labels using the independent label_return_threshold
            # 0: Downward (return < -threshold), 1: Neutral (-threshold <= return <= threshold), 2: Upward (return > threshold)
            y_labels = np.ones_like(y_return, dtype=np.int64)  # Default to neutral (class 1)
            y_labels[y_return > self.label_return_threshold] = 2  # Upward
            y_labels[y_return < -self.label_return_threshold] = 0  # Downward
            y_labels_tensor = torch.tensor(y_labels, dtype=torch.long)
            
            # --- D. Create Data Object ---
            data = Data(
                x=x_window,                 
                edge_index=edge_index,      
                edge_attr=edge_weight,      
                y=y_labels_tensor  # 3-class labels as tensor (0=Down, 1=Neutral, 2=Up)
            )
            
            data_list.append(data)

            if idx % 100 == 0:
                print(f"Processed {idx}/{num_samples} time steps...")

        # 2. Collate and Save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Done! Saved processed data to {self.processed_paths[0]}")