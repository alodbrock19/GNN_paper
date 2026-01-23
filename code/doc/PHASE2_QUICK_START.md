# Phase 2 Quick Reference - Using the New Components

## Import Statement (for Phase 2 notebooks)

```python
from models import (
    TGCN_HierarchicalMT,           # Dual-head model
    MultiTaskLoss,                  # Multi-task loss function
    create_hierarchical_labels,     # Label transformation
    hierarchical_inference,         # Inference utility
)
```

---

## 1. Data Preparation with Hierarchical Labels

```python
from functools import partial

# Create transform function
transform = partial(create_hierarchical_labels, threshold=0.0055)

# Load dataset with hierarchical labels
dataset = SP100Stocks(
    root="data/",
    adj_file_name="hybrid_adj.npy",
    future_window=1,
    transform=transform,
    force_reload=True
)

# Check that labels were created
sample = dataset[0]
assert hasattr(sample, 'signal_labels'), "Missing signal_labels"
assert hasattr(sample, 'direction_labels'), "Missing direction_labels"
```

---

## 2. Analyze Label Distribution

```python
from models import analyze_hierarchical_distribution, print_hierarchical_distribution

# Get statistics
stats = analyze_hierarchical_distribution(dataset, threshold=0.0055)

# Print nicely formatted distribution
print_hierarchical_distribution(stats)

# Access individual statistics
num_signals = stats['signal_stats']['signal']
signal_pct = stats['signal_stats']['signal_pct']
direction_balance = stats['direction_on_signal_stats']
```

**Example Output:**
```
======================================================================
HIERARCHICAL LABEL DISTRIBUTION ANALYSIS
======================================================================

Overall Dataset:
  Total samples: 100
  Total nodes: 10000

Task 1: Signal Detection (Noise vs Signal)
  Noise:  5400 ( 54.0%)
  Signal: 4600 ( 46.0%)
  → Signal class balance: ✓ Good

Task 2: Direction Prediction (All Samples)
  Down: 4800 ( 48.0%)
  Up:   5200 ( 52.0%)

Task 2: Direction Prediction (On Signal Samples Only)
  Down: 2300 ( 50.0%)
  Up:   2300 ( 50.0%)
  → Direction balance: ✓ Good

======================================================================
```

---

## 3. Create and Initialize Model

```python
# Initialize dual-head model
model = TGCN_HierarchicalMT(
    in_channels=13,          # Stock features
    hidden_size=16,          # Hidden dimension
    layers_nb=2,             # Number of TGCN layers
    use_gat=True             # Use GAT attention
)

# Move to device
model = model.to(device)

# Check output shapes
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 4. Setup Multi-Task Loss

```python
# Initialize multi-task loss (equal weighting)
criterion = MultiTaskLoss(lambda_direction=1.0)

# Move to device
criterion = criterion.to(device)

# Alternative weightings
criterion_signal_heavy = MultiTaskLoss(lambda_direction=0.5)  # Emphasize signal
criterion_direction_heavy = MultiTaskLoss(lambda_direction=2.0)  # Emphasize direction
```

---

## 5. Forward Pass and Loss Computation

```python
# Get batch
batch = next(iter(train_dataloader))
batch = batch.to(device)

# Forward pass (returns both heads)
signal_logits, direction_logits = model(
    batch.x,
    batch.edge_index,
    batch.edge_weight
)

# Prepare targets
signal_targets = batch.signal_labels.squeeze()
direction_targets = batch.direction_labels.squeeze()

# Create mask for valid direction samples (where signal=1)
signal_mask = (signal_targets == 1)

# Compute multi-task loss
total_loss, loss_dict = criterion(
    signal_logits, direction_logits,
    signal_targets, direction_targets,
    signal_mask=signal_mask
)

# Access individual losses
print(f"Signal loss: {loss_dict['signal_loss']:.4f}")
print(f"Direction loss: {loss_dict['direction_loss']:.4f}")
print(f"Total loss: {loss_dict['total_loss']:.4f}")

# Backward pass (same as Phase 1)
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

---

## 6. Evaluation with Hierarchical Inference

```python
model.eval()

# Get test batch
test_batch = next(iter(test_dataloader))
test_batch = test_batch.to(device)

with torch.no_grad():
    # Forward pass
    signal_logits, direction_logits = model(
        test_batch.x,
        test_batch.edge_index,
        test_batch.edge_weight
    )
    
    # Hierarchical inference
    result = hierarchical_inference(
        signal_logits,
        direction_logits,
        confidence_scores=True  # Get confidences
    )
    
    # Access predictions
    final_predictions = result['final_predictions']      # [N] 0=Down, 1=Neutral, 2=Up
    signal_predictions = result['signal_predictions']    # [N] 0=Noise, 1=Signal
    direction_predictions = result['direction_predictions'] # [N] 0=Down, 1=Up
    
    # Access confidences (if requested)
    final_confidence = result['final_confidence']        # [N] confidence for final pred
    signal_confidence = result['signal_confidence']      # [N] confidence for signal
    direction_confidence = result['direction_confidence'] # [N] confidence for direction

# Compute metrics
from sklearn.metrics import confusion_matrix, classification_report

final_preds_flat = final_predictions.cpu().numpy()
actual_flat = test_batch.y.cpu().numpy()

accuracy = (final_preds_flat == actual_flat).mean()
report = classification_report(
    actual_flat, final_preds_flat,
    target_names=['Down', 'Neutral', 'Up'],
    output_dict=True
)

print(f"Overall Accuracy: {accuracy:.2%}")
print(f"Down  - Precision: {report['Down']['precision']:.3f}, Recall: {report['Down']['recall']:.3f}")
print(f"Up    - Precision: {report['Up']['precision']:.3f}, Recall: {report['Up']['recall']:.3f}")
print(f"Neutral - Precision: {report['Neutral']['precision']:.3f}, Recall: {report['Neutral']['recall']:.3f}")
```

---

## 7. Task-Specific Evaluation

```python
# Evaluate signal detection task
signal_targets = test_batch.signal_labels.squeeze()
signal_preds = result['signal_predictions']

signal_accuracy = (signal_preds == signal_targets).mean()
signal_report = classification_report(
    signal_targets.cpu().numpy(),
    signal_preds.cpu().numpy(),
    target_names=['Noise', 'Signal'],
    output_dict=True
)

print("\n=== SIGNAL DETECTION TASK ===")
print(f"Accuracy: {signal_accuracy:.2%}")
print(f"Noise  - Precision: {signal_report['Noise']['precision']:.3f}, Recall: {signal_report['Noise']['recall']:.3f}")
print(f"Signal - Precision: {signal_report['Signal']['precision']:.3f}, Recall: {signal_report['Signal']['recall']:.3f}")

# Evaluate direction task (on signal samples only)
direction_targets = test_batch.direction_labels.squeeze()
direction_preds = result['direction_predictions']
signal_mask = (signal_targets == 1)

if signal_mask.sum() > 0:
    direction_accuracy = (
        direction_preds[signal_mask] == direction_targets[signal_mask]
    ).mean()
    
    direction_report = classification_report(
        direction_targets[signal_mask].cpu().numpy(),
        direction_preds[signal_mask].cpu().numpy(),
        target_names=['Down', 'Up'],
        output_dict=True
    )
    
    print("\n=== DIRECTION PREDICTION TASK (On Detected Signals) ===")
    print(f"Samples evaluated: {signal_mask.sum()}")
    print(f"Accuracy: {direction_accuracy:.2%}")
    print(f"Down - Precision: {direction_report['Down']['precision']:.3f}, Recall: {direction_report['Down']['recall']:.3f}")
    print(f"Up   - Precision: {direction_report['Up']['precision']:.3f}, Recall: {direction_report['Up']['recall']:.3f}")
```

---

## 8. Complete Training Loop Skeleton

```python
for epoch in range(num_epochs):
    # =====================
    # TRAINING
    # =====================
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        
        # Forward pass
        signal_logits, direction_logits = model(
            batch.x, batch.edge_index, batch.edge_weight
        )
        
        # Prepare targets
        signal_targets = batch.signal_labels.squeeze()
        direction_targets = batch.direction_labels.squeeze()
        signal_mask = (signal_targets == 1)
        
        # Loss computation
        loss, loss_dict = criterion(
            signal_logits, direction_logits,
            signal_targets, direction_targets,
            signal_mask=signal_mask
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_dataloader)
    
    # =====================
    # EVALUATION
    # =====================
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            
            signal_logits, direction_logits = model(
                batch.x, batch.edge_index, batch.edge_weight
            )
            
            signal_targets = batch.signal_labels.squeeze()
            direction_targets = batch.direction_labels.squeeze()
            signal_mask = (signal_targets == 1)
            
            loss, _ = criterion(
                signal_logits, direction_logits,
                signal_targets, direction_targets,
                signal_mask=signal_mask
            )
            
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_dataloader)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f} | Test Loss={avg_test_loss:.4f}")
```

---

## Parameter Tuning Reference

**If accuracy is poor, try:**

| Metric | Try This | Rationale |
|--------|----------|-----------|
| Low overall accuracy | λ = 0.5 | Emphasize signal detection |
| Poor on Down/Up | λ = 2.0 | Emphasize direction prediction |
| Slow convergence | lr = 0.001 | Multi-task may need lower LR |
| Overfitting | Add dropout to heads | Regularization |
| Underfitting | hidden_size = 32 | More capacity |

---

## Troubleshooting

**Q: "Direction loss = 0 in first epoch?"**  
A: Normal if few signals in batch. Check signal_mask.sum() > 0

**Q: "Shape mismatch errors?"**  
A: Ensure batch labels have shapes [N] not [N, 1]. Use .squeeze()

**Q: "Model outputs same prediction for both tasks?"**  
A: Check that both heads are being used. Verify forward() returns tuple

**Q: "Phase 1 notebooks broken?"**  
A: They shouldn't be. Phase 2 components are separate. Verify imports.
