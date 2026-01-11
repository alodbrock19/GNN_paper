# dynamic_graph_construction.py
import pandas as pd
import numpy as np
import torch
from apyori import apriori

class DynamicGraphGenerator:
    def __init__(self, data_path='./data/sp100_data.csv', ticker_path='./data/tickers_list.csv'):
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.tickers = pd.read_csv(ticker_path, header=None)[0].tolist()
        self.num_nodes = len(self.tickers)
        
        # Hyperparameters from RNNmeetGNN
        self.corr_threshold = 0.75
        self.association_params = {
            'min_support': 0.1,
            'min_confidence': 0.1,
            'min_lift': 1.7,
            'price_change_threshold': 0.01 # 1% change to be considered "UP"
        }

    def get_pearson_edges(self, window_data):
        """
        Method A: Pearson Correlation Graph
        """
        # Extract only 'Close' prices for correlation (or 'Adj Close')
        # Assuming columns are named like 'AAPL_Close'
        price_cols = [f"{t}_Close" for t in self.tickers]
        # Filter to available columns (in case some were dropped)
        price_cols = [c for c in price_cols if c in window_data.columns]
        
        sub_df = window_data[price_cols]
        
        # Calculate Correlation Matrix
        corr_matrix = sub_df.corr(method='pearson').values
        
        # Find indices where correlation > threshold
        # We use np.where to get (row, col) pairs
        rows, cols = np.where(corr_matrix > self.corr_threshold)
        
        start_edges = []
        end_edges = []
        
        for r, c in zip(rows, cols):
            if r != c: # Remove self-loops
                start_edges.append(r)
                end_edges.append(c)
                
        return start_edges, end_edges

    def get_association_edges(self, window_data):
        """
        Method B: Association Analysis (Apriori)
        """
        # 1. Discretize Data: Convert continuous returns to "Transactions"
        # A "transaction" is a list of stocks that went UP significantly on a specific day
        transactions = []
        
        # Calculate daily returns for this window
        price_cols = [c for c in window_data.columns if 'Close' in c]
        returns = window_data[price_cols].pct_change().dropna()
        
        threshold = self.association_params['price_change_threshold']
        
        for date_idx in range(len(returns)):
            daily_row = returns.iloc[date_idx]
            # Get tickers that went up > 1%
            winners = daily_row[daily_row > threshold].index.tolist()
            # Clean column names to get just tickers (e.g., 'AAPL_Close' -> 'AAPL')
            winners = [w.split('_')[0] for w in winners]
            
            if len(winners) > 0:
                transactions.append(winners)
        
        if not transactions:
            return [], []

        # 2. Run Apriori Algorithm
        rules = apriori(
            transactions, 
            min_support=self.association_params['min_support'],
            min_confidence=self.association_params['min_confidence'],
            min_lift=self.association_params['min_lift']
        )
        
        start_edges = []
        end_edges = []
        
        # 3. Parse Rules into Edges
        for rule in rules:
            items = list(rule.items)
            if len(items) == 2: # We only care about Pairwise rules (Stock A <-> Stock B)
                stock_a = items[0]
                stock_b = items[1]
                
                # Find their indices in our ticker list
                if stock_a in self.tickers and stock_b in self.tickers:
                    idx_a = self.tickers.index(stock_a)
                    idx_b = self.tickers.index(stock_b)
                    
                    # Add bidirectional edges
                    start_edges.extend([idx_a, idx_b])
                    end_edges.extend([idx_b, idx_a])
                    
        return start_edges, end_edges

    def build_dynamic_graph(self, current_date, window_size=30):
        """
        Main function to call during training loop.
        Returns edge_index for the specific time window ending at current_date.
        """
        # Slice the data for the lookback window
        end_idx = self.df.index.get_loc(current_date)
        start_idx = max(0, end_idx - window_size)
        
        window_data = self.df.iloc[start_idx:end_idx]
        
        if len(window_data) < 5: # Not enough data for correlation
            return torch.empty((2, 0), dtype=torch.long)

        # 1. Get Edges from Correlation
        p_start, p_end = self.get_pearson_edges(window_data)
        
        # 2. Get Edges from Association Rules
        a_start, a_end = self.get_association_edges(window_data)
        
        # 3. Combine
        final_start = p_start + a_start
        final_end = p_end + a_end
        
        # Convert to PyTorch Tensor [2, Num_Edges]
        edge_index = torch.tensor([final_start, final_end], dtype=torch.long)
        
        # Remove duplicates (if an edge is found by both methods)
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index

# --- Usage Example ---
if __name__ == "__main__":
    # Initialize
    graph_gen = DynamicGraphGenerator()
    
    # Simulate a training step
    sample_date = graph_gen.df.index[100] 
    print(f"Building graph for window ending on: {sample_date}")
    
    edge_index = graph_gen.build_dynamic_graph(sample_date, window_size=60)
    
    print(f"Graph Created!")
    print(f"Num Nodes: {graph_gen.num_nodes}")
    print(f"Num Edges: {edge_index.shape[1]}")
    print("Edge Index Sample:\n", edge_index[:, :5])