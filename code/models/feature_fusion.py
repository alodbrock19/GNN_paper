import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossNetwork(nn.Module):
    """
    Implements the Cross Network (Left path in your diagram).
    Formula: p_c = p_0 * (p_{c-1}^T * w) + b + p_{c-1}
    """
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        
        # We create a list of weights and biases for each layer
        # w has shape (dim, 1) and b has shape (dim, 1) as per the paper
        self.w = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1)) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)
        ])

    def forward(self, x_0):
        # x_0 shape: (Batch_Size, Input_Dim)
        x_prev = x_0
        
        for i in range(self.num_layers):
            # 1. Compute the "Interaction Scalar" (x_prev^T * w)
            # We want dot product per sample: (Batch, Dim) * (Dim, 1) -> (Batch, 1)
            # This represents the "Interaction" strength for this layer
            interaction_scalar = torch.mm(x_prev, self.w[i])
            
            # 2. Apply formula: x_0 * scalar + b + x_prev
            # Broadcasting: (Batch, Dim) * (Batch, 1) -> (Batch, Dim)
            x_next = x_0 * interaction_scalar + self.b[i] + x_prev
            
            x_prev = x_next
            
        return x_prev

class FeatureFusion(nn.Module):
    """
    Combines Cross Network (explicit interactions) and MLP (implicit non-linearities).
    """
    def __init__(self, hidden_size, num_graph_layers, num_cross_layers=2, mlp_hidden_dims=[64]):
        super().__init__()
        
        # 1. Calculate Input Dimension (The "Super Vector" Size)
        # Input = LSTM_Out + (Layer1 + ... + LayerM)
        # Total tensors = 1 (LSTM) + num_graph_layers
        self.total_input_dim = hidden_size * (1 + num_graph_layers)
        
        # --- Path A: Cross Network ---
        self.cross_net = CrossNetwork(self.total_input_dim, num_cross_layers)
        
        # --- Path B: Deep Network (MLP) ---
        # Standard MLP: Linear -> ReLU -> Linear
        mlp_layers = []
        input_dim = self.total_input_dim
        
        for h_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, h_dim))
            mlp_layers.append(nn.ReLU())
            input_dim = h_dim # Next layer input is current output
            
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output dimension for the next stage (Predictor)
        # It is the CrossNet output (same as input) + MLP output (last hidden dim)
        self.final_output_dim = self.total_input_dim + mlp_hidden_dims[-1]

    def forward(self, lstm_out, graph_outputs):
        """
        lstm_out: Tensor (Batch, Hidden_Size)
        graph_outputs: List of Tensors [(Batch, Hidden), (Batch, Hidden)...]
        """
        # 1. Construct p_0 (The Super Vector)
        # Concatenate all inputs along feature dimension
        # Shape: (Batch, Hidden * (1 + M))
        all_features = [lstm_out] + graph_outputs
        p_0 = torch.cat(all_features, dim=1)
        
        # 2. Run Paths in Parallel
        # Path A: Cross Network (Explicit Interactions)
        cross_out = self.cross_net(p_0)
        
        # Path B: MLP (Deep Non-linear patterns)
        mlp_out = self.mlp(p_0)
        
        # 3. Final Concatenation
        # Shape: (Batch, Total_Input + MLP_Hidden)
        final_fusion = torch.cat([cross_out, mlp_out], dim=1)
        
        return final_fusion