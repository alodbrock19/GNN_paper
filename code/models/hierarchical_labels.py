"""
Hierarchical Label Generation for Phase 2 Multi-Task Learning

Transforms standard 3-class labels (Down/Neutral/Up) into hierarchical task labels:
- Task 1 (Signal Detection): Noise (0) vs Signal (1)
- Task 2 (Direction Prediction): Down (0) vs Up (1)

This enables the dual-head architecture to separately learn signal detection
and directional prediction, reducing ambiguity in borderline cases.
"""

import torch
from torch_geometric.data import Data


def preprocess_returns(sample: Data) -> Data:
    """
    Preprocess data by converting 'y' field to 'returns' field.
    
    The SP100Stocks dataset loads data with a 'y' field that contains returns.
    This function aliases 'y' to 'returns' so it can be used by downstream
    label generation functions.
    
    Args:
        sample: Graph data sample
        
    Returns:
        sample with 'returns' field set to 'y' if 'y' exists
    """
    if hasattr(sample, 'y') and not hasattr(sample, 'returns'):
        sample.returns = sample.y
    return sample


def create_hierarchical_labels(sample: Data, threshold: float = 0.0055) -> Data:
    """
    Transform stock returns into hierarchical task labels for multi-task learning.
    
    Converts the 3-class problem (Down/Neutral/Up) into two complementary tasks:
    1. Signal Detection: Distinguish real signals (significant moves) from noise
    2. Direction Prediction: Predict Down vs Up (only for detected signals)
    
    Args:
        sample: Graph data sample from dataset with 'returns' field
        threshold: Threshold for neutral zone (default: ±0.55%)
    
    Returns:
        sample with added fields:
            - signal_labels: [num_stocks] binary labels (0=Noise, 1=Signal)
            - direction_labels: [num_stocks] binary labels (0=Down, 1=Up)
                                Only meaningful where signal_labels=1
    
    Example:
        For returns = [-1.2%, -0.3%, 0.2%, 0.8%, 1.5%] and threshold=0.55%:
        
        signal_labels:    [1,      0,      0,      1,      1]      # signal vs noise
        direction_labels: [0,      0,      0,      1,      1]      # down vs up
    """
    
    if not hasattr(sample, 'returns'):
        raise ValueError("Sample must have 'returns' field. Ensure data was loaded with create_labels_3class transform.")
    
    returns = sample.returns.squeeze()  # [num_stocks]
    
    # ============================================
    # Task 1: Signal Detection (Noise vs Signal)
    # ============================================
    # 0 = Noise (within ±threshold, ambiguous)
    # 1 = Signal (outside ±threshold, clear direction)
    sample.signal_labels = (torch.abs(returns) > threshold).long()
    
    # ============================================
    # Task 2: Direction Prediction (Down vs Up)
    # ============================================
    # 0 = Down (return < -threshold)
    # 1 = Up (return > +threshold)
    # Note: Only meaningful for samples where signal_labels=1
    sample.direction_labels = (returns > threshold).long()
    
    return sample


def analyze_hierarchical_distribution(dataset, threshold: float = 0.0055) -> dict:
    """
    Analyze the distribution of hierarchical labels in a dataset.
    
    Useful for understanding class imbalance and task difficulty.
    
    Args:
        dataset: PyTorch Geometric dataset with hierarchical labels
        threshold: Neutral zone threshold
    
    Returns:
        Dictionary with statistics:
            {
                'total_samples': int,
                'total_nodes': int,
                'signal_stats': {'noise': int, 'signal': int, 'signal_pct': float},
                'direction_stats': {'down': int, 'up': int, 'balance': float},
                'direction_on_signal_stats': {'down': int, 'up': int, 'balance': float}
            }
    """
    
    all_signal_labels = []
    all_direction_labels = []
    all_returns = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        all_signal_labels.extend(sample.signal_labels.cpu().numpy())
        all_direction_labels.extend(sample.direction_labels.cpu().numpy())
        all_returns.extend(sample.returns.squeeze().cpu().numpy())
    
    all_signal_labels = torch.tensor(all_signal_labels)
    all_direction_labels = torch.tensor(all_direction_labels)
    all_returns = torch.tensor(all_returns)
    
    # Overall statistics
    total_nodes = len(all_signal_labels)
    
    # Signal detection statistics
    signal_mask = all_signal_labels == 1
    noise_count = (~signal_mask).sum().item()
    signal_count = signal_mask.sum().item()
    
    # Direction statistics (all samples)
    down_all = (all_direction_labels == 0).sum().item()
    up_all = (all_direction_labels == 1).sum().item()
    
    # Direction statistics (only signals)
    if signal_count > 0:
        down_signal = ((all_direction_labels[signal_mask] == 0).sum()).item()
        up_signal = ((all_direction_labels[signal_mask] == 1).sum()).item()
    else:
        down_signal = 0
        up_signal = 0
    
    return {
        'total_samples': len(dataset),
        'total_nodes': total_nodes,
        'signal_stats': {
            'noise': noise_count,
            'signal': signal_count,
            'signal_pct': signal_count / total_nodes * 100 if total_nodes > 0 else 0,
            'noise_pct': noise_count / total_nodes * 100 if total_nodes > 0 else 0,
        },
        'direction_stats': {
            'down': down_all,
            'up': up_all,
            'down_pct': down_all / total_nodes * 100 if total_nodes > 0 else 0,
            'up_pct': up_all / total_nodes * 100 if total_nodes > 0 else 0,
        },
        'direction_on_signal_stats': {
            'down': down_signal,
            'up': up_signal,
            'down_pct': down_signal / signal_count * 100 if signal_count > 0 else 0,
            'up_pct': up_signal / signal_count * 100 if signal_count > 0 else 0,
        }
    }


def print_hierarchical_distribution(stats: dict) -> None:
    """
    Pretty-print hierarchical label distribution statistics.
    
    Args:
        stats: Dictionary returned by analyze_hierarchical_distribution()
    """
    print("\n" + "="*70)
    print("HIERARCHICAL LABEL DISTRIBUTION ANALYSIS")
    print("="*70)
    
    print(f"\nOverall Dataset:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total nodes: {stats['total_nodes']}")
    
    print(f"\nTask 1: Signal Detection (Noise vs Signal)")
    print(f"  Noise:  {stats['signal_stats']['noise']:5d} ({stats['signal_stats']['noise_pct']:5.1f}%)")
    print(f"  Signal: {stats['signal_stats']['signal']:5d} ({stats['signal_stats']['signal_pct']:5.1f}%)")
    print(f"  → Signal class balance: {'✓ Good' if 30 < stats['signal_stats']['signal_pct'] < 70 else '⚠ Imbalanced'}")
    
    print(f"\nTask 2: Direction Prediction (All Samples)")
    print(f"  Down: {stats['direction_stats']['down']:5d} ({stats['direction_stats']['down_pct']:5.1f}%)")
    print(f"  Up:   {stats['direction_stats']['up']:5d} ({stats['direction_stats']['up_pct']:5.1f}%)")
    
    print(f"\nTask 2: Direction Prediction (On Signal Samples Only)")
    total_signal = stats['direction_on_signal_stats']['down'] + stats['direction_on_signal_stats']['up']
    if total_signal > 0:
        print(f"  Down: {stats['direction_on_signal_stats']['down']:5d} ({stats['direction_on_signal_stats']['down_pct']:5.1f}%)")
        print(f"  Up:   {stats['direction_on_signal_stats']['up']:5d} ({stats['direction_on_signal_stats']['up_pct']:5.1f}%)")
        balance = abs(stats['direction_on_signal_stats']['down_pct'] - stats['direction_on_signal_stats']['up_pct'])
        print(f"  → Direction balance: {'✓ Good' if balance < 20 else '⚠ Imbalanced'}")
    else:
        print(f"  No signal samples in dataset!")
    
    print("="*70 + "\n")
