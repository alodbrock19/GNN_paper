# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Import your custom modules
from MST_sections.layers import AttentiveLSTM
from MST_sections.fusion import FeatureFusion

class MST_GNN(nn.Module):
    """
    The Full MST-GNN Architecture.
    """
    def __init__(self, in_features, hidden_size, num_graph_layers=2, num_cross_layers=2):
        super(MST_GNN, self).__init__()
        
        # 1. Temporal Encoding
        self.att_lstm = AttentiveLSTM(in_features, hidden_size)
        
        # 2. Spatial-Temporal Aggregation (The "Multilayer" part)
        # We use a ModuleList to stack SageConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_graph_layers):
            # SAGEConv aggregates neighbor info
            self.convs.append(SAGEConv(hidden_size, hidden_size))
            
        # 3. Feature Fusion
        self.fusion = FeatureFusion(
            hidden_size=hidden_size, 
            num_graph_layers=num_graph_layers,
            num_cross_layers=num_cross_layers,
            mlp_hidden_dims=[64]
        )
        
        # 4. Final Prediction
        # Projects the complex fused vector to 3 classes (Down, Neutral, Up)
        self.predictor = nn.Linear(self.fusion.final_output_dim, 3)

    def forward(self, x, edge_index):
        """
        x: (Batch_Size, Time_Steps, Features)
        edge_index: (2, Num_Edges)
        """
        
        # --- Step 1: Encode History ---
        # Output: (Batch, Hidden)
        lstm_out = self.att_lstm(x) 
        
        # --- Step 2: Aggregate Neighbors ---
        layer_outputs = []
        h = lstm_out
        
        for conv in self.convs:
            # Apply SageConv + Activation
            # SAGEConv handles the "Spatial" aggregation.
            # Passing 'h' (which contains LSTM history) handles the "Temporal" aspect.
            h = F.relu(conv(h, edge_index))
            
            # Store this layer's output for the fusion block
            layer_outputs.append(h)
            
        # --- Step 3: Fuse Everything ---
        fused_vector = self.fusion(lstm_out, layer_outputs)
        
        # --- Step 4: Predict ---
        return self.predictor(fused_vector)