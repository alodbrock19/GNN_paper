import numpy as np
import pandas as pd
import torch


def get_graph_in_pyg_format(values_path: str, adj_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Creates the PyTorch Geometric graph data from the stock price data and adjacency matrix.
	:param values_path: Path of the CSV file containing the stock price data
	:param adj_path: Path of the NumPy file containing the adjacency matrix
	:return: The graph data in PyTorch Geometric format
		x: Node features (nodes_nb, timestamps_nb, features_nb)
		close_prices: Close prices (nodes_nb, timestamps_nb)
		edge_index: Edge index (2, edge_nb)
		edge_weight: Edge weight (edge_nb,)
	"""
	print("=" * 60)
	print("Loading and processing data for PyG format...")
	
	# Load data
	values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
	adj = np.load(adj_path)
	nodes_nb, edge_nb = len(adj), np.count_nonzero(adj)
	
	print(f"Values DataFrame shape: {values.shape}")
	print(f"Values index levels: {values.index.names}")
	print(f"Unique symbols in values: {values.index.get_level_values('Symbol').nunique()}")
	print(f"Adjacency matrix shape: {adj.shape}")
	print(f"Number of nodes (from adj): {nodes_nb}")
	print(f"Number of edges: {edge_nb}")
	
	# Get unique symbols and dates
	unique_symbols = sorted(values.index.get_level_values('Symbol').unique())
	unique_dates = sorted(values.index.get_level_values('Date').unique())
	
	print(f"Unique symbols count: {len(unique_symbols)}")
	print(f"Unique dates count: {len(unique_dates)}")
	
	# Check if number of symbols matches adjacency matrix
	if len(unique_symbols) != nodes_nb:
		raise ValueError(
			f"Mismatch between number of stocks in values.csv ({len(unique_symbols)}) "
			f"and adjacency matrix ({nodes_nb}). "
			f"Please ensure the adjacency matrix corresponds to the stocks in values.csv."
		)
	
	# Pivot the data: Date as index, Symbol as columns, features as values
	# This ensures all stocks have the same timestamps
	print("\nPivoting data to align stocks and timestamps...")
	
	# Get feature columns (excluding Close)
	feature_cols = [col for col in values.columns if col != "Close"]
	print(f"Feature columns (excluding Close): {len(feature_cols)}")
	print(f"Features: {feature_cols}")
	
	# Reset index to have Symbol and Date as columns for pivoting
	values_reset = values.reset_index()
	
	# Pivot for each feature separately, ensuring consistent symbol and date ordering
	pivoted_data = {}
	for col in feature_cols + ["Close"]:
		pivot = values_reset.pivot(index='Date', columns='Symbol', values=col)
		# Reindex to ensure all symbols and dates are present in sorted order
		pivot = pivot.reindex(index=unique_dates, columns=unique_symbols)
		# Forward fill missing values (if a stock is missing a date, use previous value)
		# Then backward fill for any remaining NaNs at the beginning
		pivot = pivot.ffill().bfill().fillna(0)
		pivoted_data[col] = pivot
	
	print(f"Pivoted shape (Date x Symbol): {pivoted_data[feature_cols[0]].shape}")
	
	# Stack the pivoted data: (Symbol, Date, Features)
	# We want: (nodes_nb, timestamps_nb, features_nb)
	# So we need to transpose and organize
	
	# Get the feature arrays (Date x Symbol)
	feature_arrays = [pivoted_data[col].values.T for col in feature_cols]  # (Symbol, Date)
	close_array = pivoted_data["Close"].values.T  # (Symbol, Date)
	
	# Stack features: (Symbol, Date, Features)
	x_array = np.stack(feature_arrays, axis=-1)  # (nodes_nb, timestamps_nb, features_nb)
	
	print(f"\nFinal array shapes:")
	print(f"  x (features): {x_array.shape}")
	print(f"  close_prices: {close_array.shape}")
	print(f"  Expected x shape: (nodes_nb={nodes_nb}, timestamps_nb={x_array.shape[1]}, features_nb={len(feature_cols)})")
	
	# Convert to tensors
	x = torch.tensor(x_array, dtype=torch.float32)
	close_prices = torch.tensor(close_array, dtype=torch.float32)
	
	# Transpose x to match expected format: (nodes_nb, features_nb, timestamps_nb)
	# Actually, the docstring says (nodes_nb, timestamps_nb, features_nb), so we keep it as is
	# But the original code had transpose(1, 2), so let's check the expected format
	# Looking at SP100Stock.py line 45: x[:, :, idx:idx + self.past_window]
	# This suggests x is (nodes_nb, features_nb, timestamps_nb)
	# So we need to transpose
	x = x.transpose(1, 2)  # (nodes_nb, timestamps_nb, features_nb) -> (nodes_nb, features_nb, timestamps_nb)
	
	print(f"  After transpose: x shape = {x.shape}")
	print(f"  Expected: (nodes_nb={nodes_nb}, features_nb={len(feature_cols)}, timestamps_nb={x_array.shape[1]})")
	
	# Build edge index and edge weights
	print("\nBuilding edge index and edge weights...")
	edge_index, edge_weight = torch.zeros((2, edge_nb), dtype=torch.long), torch.zeros((edge_nb,), dtype=torch.float32)
	count = 0
	for i in range(nodes_nb):
		for j in range(nodes_nb):
			if (weight := adj[i, j]) != 0:
				edge_index[0, count], edge_index[1, count] = i, j
				edge_weight[count] = weight
				count += 1
	
	print(f"  Created {count} edges")
	print("=" * 60)
	print()
	
	return x, close_prices, edge_index, edge_weight


def get_stocks_labels() -> list[str]:
	"""
	Retrieves the labels (symbols) of the dataset stocks
	:return: The list of stock labels
	"""
	return pd.read_csv("../data/tickers_list.csv")["Ticker"].unique().tolist()