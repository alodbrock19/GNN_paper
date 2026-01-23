# Phase 2: Hierarchical Multi-Task Learning - Implementation Plan

## Overview
This document provides a step-by-step implementation plan for Phase 2 hierarchical multi-task learning, organized into three phases: Setup, Implementation, and Analysis.

---

## Phase 2a: Setup & Preparation

### Step 1: Create Hierarchical Label Generator

**File:** `models/hierarchical_labels.py` (NEW)

**Purpose:** Transform Phase 1 3-class labels into Phase 2 hierarchical labels

**Implementation:**
```python
def create_hierarchical_labels(sample: Data, threshold: float = 0.0055):
    """
    Transform 3-class labels into 2 hierarchical task labels:
    - Task 1 (Signal): Noise (0) vs Signal (1)
    - Task 2 (Direction): Down (0) vs Up (1)
    
    Args:
        sample: Data object with returns field
        threshold: Threshold for neutral zone
    
    Returns:
        sample with signal_labels and direction_labels added
    """
    returns = sample.returns.squeeze()
    
    # Task 1: Signal Detection
    # 0 = Noise (within ±threshold)
    # 1 = Signal (outside ±threshold)
    sample.signal_labels = (torch.abs(returns) > threshold).long()
    
    # Task 2: Direction (only defined for signals)
    # 0 = Down (return < -threshold)
    # 1 = Up (return > +threshold)
    # Neutral samples will have undefined direction (handled via masking)
    sample.direction_labels = (returns > threshold).long()
    
    return sample
```

**Test:** Verify label distribution
- Signal class ratio: should be ~40-60% signals, 40-60% noise
- Direction class ratio (on signals): should be ~50-50% Down/Up

### Step 2: Design Dual-Head Model Architecture

**File:** `models/TGCN_HierarchicalMT.py` (NEW)

**Purpose:** Extend existing TGCN to support dual output heads

**Key Points:**
- Reuse existing `TGCN` backbone from `models/TGCN.py`
- Modify output layer to `hidden_size` instead of 3
- Add two linear heads: signal_head and direction_head
- Implement `forward()` to return both logits

**Structure:**
```python
class TGCN_HierarchicalMT(nn.Module):
    def __init__(self, in_channels, hidden_size, layers_nb, use_gat=True):
        self.tgcn_backbone = TGCN(...)
        self.signal_head = nn.Linear(hidden_size, 2)
        self.direction_head = nn.Linear(hidden_size, 2)
    
    def forward(self, x, edge_index, edge_weight=None):
        features = self.tgcn_backbone(x, edge_index, edge_weight)
        signal_logits = self.signal_head(features)
        direction_logits = self.direction_head(features)
        return signal_logits, direction_logits
```

**Verification:**
- Check model output shapes: should be (batch_size*num_stocks, 2) for each head
- Verify parameter count: ~5-10% increase over Phase 1
- Test on dummy batch

### Step 3: Implement Multi-Task Loss

**File:** `models/train.py` (MODIFY - add MultiTaskLoss class)

**Purpose:** Loss function combining both task losses

**Implementation:**
```python
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_direction=1.0):
        super().__init__()
        self.lambda_direction = lambda_direction
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, signal_logits, direction_logits, 
                signal_targets, direction_targets, signal_mask):
        loss_signal = self.ce_loss(signal_logits, signal_targets)
        
        if signal_mask.sum() > 0:
            loss_direction = self.ce_loss(
                direction_logits[signal_mask],
                direction_targets[signal_mask]
            )
        else:
            loss_direction = torch.tensor(0.0, device=signal_logits.device)
        
        total_loss = loss_signal + self.lambda_direction * loss_direction
        return total_loss
```

**Parameters to test:**
- `lambda_direction = 1.0` (equal weighting)
- Alternative values: 0.5, 0.75, 1.5, 2.0 (if needed for tuning)

---

## Phase 2b: Training Implementation

### Step 4: Modify Dataset to Include Hierarchical Labels

**Location:** `phase2_hierarchical.ipynb` (NEW NOTEBOOK - Section 2)

**Actions:**
```python
# In data loading section:
from functools import partial
from models.hierarchical_labels import create_hierarchical_labels

transform = partial(create_hierarchical_labels, threshold=0.0055)
dataset = SP100Stocks(
    root="data/",
    adj_file_name="hybrid_adj.npy",
    future_window=1,
    transform=transform,
    force_reload=True
)

# Verify labels were created
sample = dataset[0]
assert hasattr(sample, 'signal_labels'), "Missing signal_labels"
assert hasattr(sample, 'direction_labels'), "Missing direction_labels"
print(f"Signal distribution: {sample.signal_labels.bincount()}")
```

### Step 5: Create Modified Training Loop

**Location:** `phase2_hierarchical.ipynb` (NEW NOTEBOOK - Section 4)

**Purpose:** Training loop for multi-task learning

**Pseudo-code:**
```python
for epoch in range(num_epochs):
    model.train()
    
    for batch in train_dataloader:
        batch = batch.to(device)
        
        # Forward pass
        signal_logits, direction_logits = model(
            batch.x, batch.edge_index, batch.edge_weight
        )
        
        # Prepare targets
        signal_targets = batch.signal_labels.squeeze()
        direction_targets = batch.direction_labels.squeeze()
        signal_mask = (signal_targets == 1)
        
        # Compute loss
        loss = criterion(
            signal_logits, direction_logits,
            signal_targets, direction_targets,
            signal_mask
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Test evaluation (same structure as training)
```

**Key Implementation Details:**
- Use `signal_mask` to compute direction loss only on valid samples
- Track both task losses separately for monitoring
- Save model checkpoints every 25 epochs

### Step 6: Create New Training Notebook

**File:** `phase2_hierarchical.ipynb` (NEW)

**Structure (8 sections like Phase 1):**

1. **Section 1:** Imports & Configuration
   - Load libraries, set device, set seeds
   - Display configuration summary

2. **Section 2:** Data Preprocessing
   - Load dataset with hierarchical labels
   - Analyze signal vs noise distribution
   - Analyze direction distribution (on signals only)

3. **Section 3:** Train/Test Split
   - Same 90/10 split as Phase 1
   - Verify label preservation through split

4. **Section 4:** Model & Loss Setup
   - Initialize `TGCN_HierarchicalMT`
   - Initialize `MultiTaskLoss` with λ=1.0
   - Initialize optimizer

5. **Section 5:** Training Loop
   - Implement full training with both tasks
   - Track separate task losses
   - Save best model

6. **Section 6:** Evaluation
   - Signal detection metrics (accuracy, precision, recall, F1)
   - Direction prediction metrics (on detected signals)
   - Overall accuracy with hierarchical inference

7. **Section 7:** Visualization
   - Task-specific confusion matrices
   - Per-task performance plots
   - Signal vs Noise separation quality
   - Direction accuracy on detected signals

8. **Section 8:** Results Export
   - Save metrics to JSON
   - Export predictions with confidence scores
   - Compare Phase 1 vs Phase 2 performance

---

## Phase 2c: Analysis & Optimization

### Step 7: Implement Hierarchical Inference

**Location:** `phase2_hierarchical.ipynb` (Section 6 - Evaluation)

**Logic:**
```python
def hierarchical_inference(signal_logits, direction_logits):
    """
    Convert dual-head outputs to final 3-class predictions
    
    Output mapping:
    - Signal=0 (Noise) → Final_Pred=1 (Neutral)
    - Signal=1, Direction=0 → Final_Pred=0 (Down)
    - Signal=1, Direction=1 → Final_Pred=2 (Up)
    """
    signal_preds = signal_logits.argmax(dim=-1)
    direction_preds = direction_logits.argmax(dim=-1)
    
    final_preds = torch.ones_like(signal_preds)  # Default Neutral
    
    signal_detected = (signal_preds == 1)
    final_preds[signal_detected & (direction_preds == 0)] = 0  # Down
    final_preds[signal_detected & (direction_preds == 1)] = 2  # Up
    
    return final_preds  # 0=Down, 1=Neutral, 2=Up
```

### Step 8: Detailed Performance Analysis

**Location:** `phase2_hierarchical.ipynb` (Section 7 - Visualization)

**Metrics to Track:**

**Task 1 (Signal Detection):**
- Accuracy: % correctly classified as signal/noise
- Precision (Signal): % of predicted signals that are real
- Recall (Signal): % of real signals correctly detected
- F1-Score: Harmonic mean

**Task 2 (Direction on Signals):**
- Accuracy: % correct direction on detected signals only
- Precision (Down): % of predicted downs that actually went down
- Precision (Up): % of predicted ups that actually went up
- Recall (Down/Up): Coverage of each direction
- F1-Score: Per-class

**Overall (Phase 1 Compatible):**
- 3-class accuracy (mapping back to Down/Neutral/Up)
- Comparison with Phase 1 baseline
- Improvement percentage

**Plots to Generate:**
1. Signal vs Noise confusion matrix
2. Direction confusion matrix (on signals only)
3. 3-class confusion matrix (final predictions)
4. Loss curves (task 1, task 2, total)
5. Accuracy progression per task per epoch
6. Signal detection confidence distribution
7. Direction confidence distribution (conditional on signal)

### Step 9: Hyperparameter Tuning (If Needed)

**If results < 45% accuracy:**

**Tunable Parameters:**
1. **Lambda (λ) - Task Weight Balance**
   - Current: 1.0 (equal weighting)
   - Try: 0.5 (emphasize signal), 1.5 (emphasize direction), 2.0
   - Test: Run 3 quick experiments, pick best

2. **Learning Rate**
   - Current: 0.005
   - Try: 0.001, 0.002, 0.005, 0.01
   - Impact: Lower might help with multi-task learning

3. **Backbone Hidden Size**
   - Current: 16
   - Try: 16, 32 (if underfitting)
   - Impact: More capacity for feature sharing

4. **Dropout (if needed)**
   - Could add to linear heads for regularization
   - Only if overfitting observed

**Tuning Procedure:**
- Hold everything constant except λ, run 3 experiments
- Pick best λ, fix it
- Then try learning rate variations
- Then try hidden size (only if still needed)

### Step 10: Validation Against Success Criteria

**Check Each Criterion:**

✅ **Criterion 1: Overall Accuracy ≥ 45%**
```python
phase2_accuracy = (final_preds == actuals).mean()
assert phase2_accuracy >= 0.45, f"Failed: {phase2_accuracy:.2%}"
```

✅ **Criterion 2: Signal Detection ≥ 70%**
```python
signal_accuracy = (signal_preds == signal_targets).mean()
assert signal_accuracy >= 0.70, f"Failed: {signal_accuracy:.2%}"
```

✅ **Criterion 3: Direction on Signals ≥ 50%**
```python
signal_mask = (signal_targets == 1)
direction_accuracy = (direction_preds[signal_mask] == direction_targets[signal_mask]).mean()
assert direction_accuracy >= 0.50, f"Failed: {direction_accuracy:.2%}"
```

✅ **Criterion 4: F1-Score Improvement**
```python
phase2_f1_macro = report['macro avg']['f1-score']
assert phase2_f1_macro > phase1_f1_macro, "F1 did not improve"
```

✅ **Criterion 5: No Task Collapse**
```python
# Check that both tasks are learning
assert signal_loss > 0.1, "Signal loss too small (possible collapse)"
assert direction_loss > 0.1, "Direction loss too small (possible collapse)"
```

---

## Implementation Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| Week 1 | 2a | Steps 1-3 | Label generator, model class, loss function |
| Week 1 | 2b | Steps 4-6 | Modified notebook with full pipeline |
| Week 1 | 2c | Steps 7-8 | Inference logic, analysis notebooks |
| Week 2 | 2c | Steps 9-10 | Hyperparameter tuning (if needed), validation |

---

## Files Checklist

**New Files to Create:**
- [ ] `models/hierarchical_labels.py`
- [ ] `models/TGCN_HierarchicalMT.py`
- [ ] `phase2_hierarchical.ipynb`

**Files to Modify:**
- [ ] `models/train.py` (add MultiTaskLoss class)

**Documentation to Update:**
- [ ] `doc/PHASE2_OVERVIEW.md` ✅ (Done)
- [ ] `doc/PHASE2_ARCHITECTURE.md` ✅ (Done)
- [ ] `doc/PHASE2_IMPLEMENTATION_PLAN.md` ✅ (Done)

---

## Expected Outputs

**Upon Completion:**

1. **Model Files:**
   - Trained `TGCN_HierarchicalMT` checkpoint
   - Best validation accuracy model

2. **Results Files:**
   - `runs/TGCN_Hierarchical_[date_time]/`
     - `evaluation_results.png` (comprehensive visualization)
     - `metrics.json` (per-task metrics)
     - `predictions.csv` (final predictions)

3. **Documentation:**
   - Phase 2 completion report
   - Performance comparison Phase 1 vs Phase 2
   - Lessons learned

4. **Decision Point:**
   - If Accuracy ≥ 45%: Phase 2 SUCCESS → consider further optimization
   - If Accuracy 40-45%: Marginal success → may need tuning
   - If Accuracy < 40%: Needs investigation → debug or try alternative

---

## Troubleshooting Guide

**Problem: Direction loss = 0 after first epoch**
- Cause: No samples with signal=1 early on
- Solution: Check label distribution, ensure data is correct

**Problem: Total accuracy < Phase 1**
- Cause: Multi-task learning might need tuning
- Solution: Try λ = 0.5 (emphasize signal detection)

**Problem: Signal detection working but direction poor**
- Cause: Shared backbone might not have capacity
- Solution: Try hidden_size=32 or λ=2.0 (emphasize direction)

**Problem: Both tasks have high loss, not converging**
- Cause: Learning rate might be too high for multi-task
- Solution: Try lr=0.001 or 0.002

**Problem: Model trains fast, overfits**
- Cause: Multi-task not providing enough regularization
- Solution: Add dropout to linear heads, increase num_epochs

