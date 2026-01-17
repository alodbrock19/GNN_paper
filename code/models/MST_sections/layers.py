# layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveLSTM(nn.Module):
    """
    Module 1: Stock Feature Encoding.
    Encodes a time sequence (e.g., 20 days) into a single vector using 
    LSTM + Temporal Attention.
    """
    def __init__(self, in_channels, hidden_size):
        super(AttentiveLSTM, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        
        # Attention Mechanism: Determines which days in the sequence matter most
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        Input: (Batch_Size, Time_Steps, Features)
        Output: (Batch_Size, Hidden_Size) -> The "h_{i,t}" vector
        """
        # 1. LSTM Encoding
        # outputs shape: (Batch, Time, Hidden)
        outputs, _ = self.lstm(x)
        
        # 2. Calculate Attention Weights
        # We project the LSTM outputs to a scalar score, then softmax over Time dim
        # attn_weights shape: (Batch, Time, 1)
        attn_weights = F.softmax(self.attention(outputs), dim=1)
        
        # 3. Weighted Sum (Context Vector)
        # Sum over the Time dimension
        context = torch.sum(attn_weights * outputs, dim=1)
        
        return context