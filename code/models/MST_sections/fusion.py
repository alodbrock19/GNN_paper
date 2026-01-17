# fusion.py
import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    """
    The explicit feature interaction module.
    Formula: x_{l+1} = x_0 * (x_l^T * w) + b + x_l
    """
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        
        # Learnable parameters for each layer
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x_0):
        x_prev = x_0
        
        for i in range(self.num_layers):
            # Optimized calculation: (Batch, Dim) dot (Dim, 1) -> (Batch, 1) scalar
            # This represents the interaction strength for this layer
            interaction_scalar = torch.mm(x_prev, self.w[i])
            
            # Apply formula: x_0 * scalar + b + x_prev
            x_next = x_0 * interaction_scalar + self.b[i] + x_prev
            x_prev = x_next
            
        return x_prev

class FeatureFusion(nn.Module):
    """
    Module 3: Cross-Layer High-Order Feature Fusion.
    Combines outputs from LSTM and all Graph Layers using DCN + MLP.
    """
    def __init__(self, hidden_size, num_graph_layers, num_cross_layers=2, mlp_hidden_dims=[64]):
        super(FeatureFusion, self).__init__()
        
        # Calculate the size of the "Super Vector" p_0
        # Input = LSTM_Out + (Layer1 + ... + LayerM)
        self.total_input_dim = hidden_size * (1 + num_graph_layers)
        
        # Path A: Cross Network (Explicit Interactions)
        self.cross_net = CrossNetwork(self.total_input_dim, num_cross_layers)
        
        # Path B: MLP (Deep Non-linear patterns)
        mlp_layers = []
        input_dim = self.total_input_dim
        for h_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h_dim))
            mlp_layers.append(nn.ReLU())
            input_dim = h_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final Dimension = CrossNet Output (Size matches input) + MLP Output
        self.final_output_dim = self.total_input_dim + mlp_hidden_dims[-1]

    def forward(self, lstm_out, graph_outputs):
        # 1. Create p_0 by concatenating everything side-by-side
        # All inputs have shape (Batch, Hidden) -> Result (Batch, Hidden * (M+1))
        all_features = [lstm_out] + graph_outputs
        p_0 = torch.cat(all_features, dim=1)
        
        # 2. Run Parallel Paths
        cross_out = self.cross_net(p_0)
        mlp_out = self.mlp(p_0)
        
        # 3. Final Concatenation
        return torch.cat([cross_out, mlp_out], dim=1)