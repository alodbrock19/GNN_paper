# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops 

from .MST_sections.layers import AttentiveLSTM
from .MST_sections.fusion import FeatureFusion

class MST_GNN(nn.Module):
    def __init__(self, in_features, hidden_size, num_graph_layers=2, num_cross_layers=2):
        super(MST_GNN, self).__init__()
        
        # 1. Temporal Encoding
        self.att_lstm = AttentiveLSTM(in_features, hidden_size)
        
        # 2. Spatial-Temporal Aggregation
        self.convs = nn.ModuleList()
        for _ in range(num_graph_layers):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
            
        # 3. Feature Fusion
        self.fusion = FeatureFusion(
            hidden_size=hidden_size, 
            num_graph_layers=num_graph_layers,
            num_cross_layers=num_cross_layers,
            mlp_hidden_dims=[64]
        )
        
        # 4. Final Prediction (Single Head)
        # Output size 1 for Binary Classification (Up/Down)
        self.predictor = nn.Linear(self.fusion.final_output_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        # ... (Step 0, 1, 2, 3 are identical to previous code) ...
        
        # Step 0: Graph Prep
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Initialize edge_weight as zero tensor if not provided
        #if edge_weight is None:
        #    edge_weight = torch.zeros(edge_index_with_loops.size(1), device=x.device)

        # Step 1: LSTM
        lstm_out = self.att_lstm(x) 
        
        # Step 2: SAGE Loop
        layer_outputs = []
        h = lstm_out
        for conv in self.convs:
            h = F.relu(conv(h, edge_index_with_loops))
            layer_outputs.append(h)
            
        # Step 3: Fusion
        fused_vector = self.fusion(lstm_out, layer_outputs)
        
        # Step 4: Single Classification Output
        # We return logits (raw scores). 
        # The Sigmoid/Softmax will happen in the Loss Function (BCEWithLogitsLoss).
        return self.predictor(fused_vector)