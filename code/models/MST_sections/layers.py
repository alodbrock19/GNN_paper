import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentiveLSTM(nn.Module):
    """
    Module 1: Stock Feature Encoding.
    Updated to match Equation 7: Attention includes current context x_{i,t}
    """
    def __init__(self, in_channels, hidden_size):
        super(AttentiveLSTM, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        
        # FIXED: Input to attention is Hidden (History) + In_Channels (Current Context)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size + in_channels, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (Batch, Time_Steps, Features)
        outputs, _ = self.lstm(x)
        
        # --- Context Injection (Equation 7) ---
        # Get the features of the *last* time step (Today's Context)
        # Shape: (Batch, 1, Features)
        current_x = x[:, -1, :].unsqueeze(1)
        
        # Expand to match sequence length: (Batch, Time, Features)
        current_x_expanded = current_x.expand(-1, outputs.size(1), -1)
        
        # Concatenate: [History; Context]
        attn_input = torch.cat([outputs, current_x_expanded], dim=2)
        
        # Calculate weights using the combined info
        attn_weights = F.softmax(self.attention(attn_input), dim=1)
        
        # Weighted Sum (Equation 5)
        context = torch.sum(attn_weights * outputs, dim=1)
        
        return context