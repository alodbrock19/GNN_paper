"""
TGCN with Hierarchical Multi-Task Learning Heads

Implements a dual-head TGCN architecture for Phase 2 hierarchical multi-task learning:
1. Signal Detection Head: Predicts if movement is signal (significant) or noise
2. Direction Head: Predicts Down vs Up (only meaningful for detected signals)

The shared TGCN backbone learns features useful for both tasks simultaneously,
reducing ambiguity in the neutral zone and improving overall accuracy.

Reference: López de Prado (2018) - "Advances in Financial Machine Learning"
"""

import torch
from torch import nn
from torch.nn import functional as F

from .TGCN import TGCN


class TGCN_HierarchicalMT(nn.Module):
    """
    Hierarchical Multi-Task TGCN with dual output heads.
    
    Architecture:
    ```
    Input Features [N, T, F]
        ↓
    TGCN Backbone (shared feature extraction)
        ↓ Output: [N, hidden_size]
        ├─→ Signal Head: Linear(hidden_size, 2)
        │   Output: [N, 2] (Noise=0, Signal=1)
        │
        └─→ Direction Head: Linear(hidden_size, 2)
            Output: [N, 2] (Down=0, Up=1)
    ```
    
    Args:
        in_channels (int): Number of input features (13 for stock data)
        hidden_size (int): TGCN hidden dimension (default: 16)
        layers_nb (int): Number of TGCN layers (default: 2)
        use_gat (bool): Use GAT instead of GCN (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 16,
        layers_nb: int = 2,
        use_gat: bool = True
    ):
        super().__init__()
        
        # ============================================
        # Shared Backbone: TGCN Feature Extractor
        # ============================================
        # Modified to output hidden_size instead of fixed out_channels
        # This allows flexibility for multi-task heads
        self.tgcn_backbone = TGCN(
            in_channels=in_channels,
            out_channels=hidden_size,  # ← Changed from 3 (Down/Neutral/Up)
            hidden_size=hidden_size,
            layers_nb=layers_nb,
            use_gat=use_gat
        )
        
        # ============================================
        # Task 1: Signal Detection Head
        # ============================================
        # Binary classification: Noise (0) vs Signal (1)
        self.signal_head = nn.Linear(hidden_size, 2)
        
        # ============================================
        # Task 2: Direction Prediction Head
        # ============================================
        # Binary classification: Down (0) vs Up (1)
        # Only trained/evaluated on samples where signal_head predicts Signal
        self.direction_head = nn.Linear(hidden_size, 2)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass through both task heads.
        
        Args:
            x: Node feature matrix of shape [N, T, F]
               where N=num_nodes, T=sequence_length, F=num_features
            edge_index: Graph edges of shape [2, E]
            edge_weight: Edge weights of shape [E] (optional)
        
        Returns:
            Tuple of (signal_logits, direction_logits):
                signal_logits: [N, 2] - Signal detection logits
                direction_logits: [N, 2] - Direction prediction logits
        
        Example:
            >>> model = TGCN_HierarchicalMT(in_channels=13, hidden_size=16)
            >>> signal_logits, direction_logits = model(x, edge_index, edge_weight)
            >>> signal_logits.shape
            torch.Size([100, 2])  # 100 stocks, 2 classes (Noise, Signal)
            >>> direction_logits.shape
            torch.Size([100, 2])  # 100 stocks, 2 classes (Down, Up)
        """
        
        # ============================================
        # Shared Feature Extraction
        # ============================================
        # All temporal and graph information processed here
        backbone_output = self.tgcn_backbone(x, edge_index, edge_weight)
        # backbone_output shape: [N, hidden_size]
        
        # ============================================
        # Task 1: Signal Detection
        # ============================================
        signal_logits = self.signal_head(backbone_output)  # [N, 2]
        
        # ============================================
        # Task 2: Direction Prediction
        # ============================================
        direction_logits = self.direction_head(backbone_output)  # [N, 2]
        
        return signal_logits, direction_logits


def hierarchical_inference(
    signal_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    confidence_scores: bool = False
) -> dict:
    """
    Convert dual-head outputs to final 3-class predictions using hierarchical logic.
    
    Inference Strategy:
    1. Get signal detection prediction from signal_logits
    2. If signal detected (signal_pred == 1):
       - Use direction prediction (0→Down, 1→Up)
    3. If noise detected (signal_pred == 0):
       - Predict Neutral (no direction)
    
    Output Mapping:
    - signal_pred=0 (Noise) → final_pred=1 (Neutral)
    - signal_pred=1, direction_pred=0 → final_pred=0 (Down)
    - signal_pred=1, direction_pred=1 → final_pred=2 (Up)
    
    Args:
        signal_logits: [N, 2] logits from signal head
        direction_logits: [N, 2] logits from direction head
        confidence_scores: Whether to return confidence scores
    
    Returns:
        Dictionary with keys:
            - 'final_predictions': [N] final class predictions (0=Down, 1=Neutral, 2=Up)
            - 'signal_predictions': [N] signal head predictions (0=Noise, 1=Signal)
            - 'direction_predictions': [N] direction head predictions (0=Down, 1=Up)
            - 'signal_confidence': [N] confidence for signal prediction (if confidence_scores=True)
            - 'direction_confidence': [N] confidence for direction prediction (if confidence_scores=True)
            - 'final_confidence': [N] confidence for final prediction (if confidence_scores=True)
    
    Example:
        >>> signal_logits = torch.randn(100, 2)
        >>> direction_logits = torch.randn(100, 2)
        >>> result = hierarchical_inference(signal_logits, direction_logits, confidence_scores=True)
        >>> result['final_predictions'].shape
        torch.Size([100])
        >>> result['final_confidence'].mean()
        tensor(0.55)  # Average confidence
    """
    
    # ============================================
    # Task 1: Signal Detection
    # ============================================
    signal_probs = F.softmax(signal_logits, dim=-1)  # [N, 2]
    signal_preds = signal_probs.argmax(dim=-1)       # [N] → 0 or 1
    signal_conf = signal_probs.max(dim=-1)[0]        # [N] → confidence
    
    # ============================================
    # Task 2: Direction Prediction
    # ============================================
    direction_probs = F.softmax(direction_logits, dim=-1)  # [N, 2]
    direction_preds = direction_probs.argmax(dim=-1)       # [N] → 0 or 1
    direction_conf = direction_probs.max(dim=-1)[0]        # [N] → confidence
    
    # ============================================
    # Hierarchical Decision
    # ============================================
    # Default: all predictions are Neutral (1)
    final_preds = torch.ones_like(signal_preds)  # [N] initialized to 1
    final_conf = signal_conf.clone()              # [N] inherit signal confidence
    
    # Where signal is detected (signal_preds == 1)
    signal_detected = (signal_preds == 1)
    
    if signal_detected.sum() > 0:
        # For detected signals: map direction predictions to Down (0) or Up (2)
        direction_on_signal = direction_preds[signal_detected]  # [M] where M = number of signals
        
        # Map: 0→0 (Down stays Down), 1→2 (Up becomes 2)
        direction_mapped = direction_on_signal.clone()
        direction_mapped[direction_on_signal == 1] = 2
        
        # Update final predictions and confidences
        final_preds[signal_detected] = direction_mapped
        final_conf[signal_detected] = direction_conf[signal_detected]
    
    # ============================================
    # Build result dictionary
    # ============================================
    result = {
        'final_predictions': final_preds,           # [N] 0=Down, 1=Neutral, 2=Up
        'signal_predictions': signal_preds,         # [N] 0=Noise, 1=Signal
        'direction_predictions': direction_preds,   # [N] 0=Down, 1=Up
    }
    
    if confidence_scores:
        result['signal_confidence'] = signal_conf
        result['direction_confidence'] = direction_conf
        result['final_confidence'] = final_conf
    
    return result
