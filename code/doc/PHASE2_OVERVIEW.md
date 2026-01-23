# Phase 2: Hierarchical Multi-Task Learning - Overview

## Objective
Improve stock trend classification accuracy from **36.25% (Phase 1)** to **45-50% (Phase 2 target)** using hierarchical multi-task learning with dual-head architecture.

## Problem Statement (Phase 1 Analysis)
- Phase 1 achieved 36.25% accuracy with correct class balance
- Remaining bottleneck: **inherent noise in ±0.55% neutral zone**
- Model struggles to distinguish signal from noise in borderline cases
- Root cause: Forcing model to simultaneously learn noise detection AND direction prediction

## Phase 2 Solution: Hierarchical Decomposition
Split the problem into two sequential, dependent tasks with dedicated objectives:

### Task 1: Signal Detection (Binary Classification)
**Question:** Is this movement a *real signal* or just *noise*?
- **Input:** Stock features and graph structure
- **Output:** Binary classification (2 classes)
  - Class 0: **Noise** (neutral zone: returns between ±0.55%)
  - Class 1: **Signal** (significant move: returns outside ±0.55%)
- **Purpose:** Filter out ambiguous movements before direction prediction
- **Expected accuracy:** ~70-80% (easier than direction on signal)

### Task 2: Direction Prediction (Binary Classification)
**Question:** Will the stock go *down* or *up* (given it's a significant signal)?
- **Input:** Stock features and graph structure (SAME as Task 1)
- **Output:** Binary classification (2 classes)
  - Class 0: **Down** (return < -0.55%)
  - Class 1: **Up** (return > +0.55%)
- **Constraint:** Only evaluated/trained on samples where Task 1 predicts "Signal"
- **Purpose:** Focus learning on clear directional patterns, ignore neutral zone
- **Expected accuracy:** ~50-60% on signal samples

## Architecture: Dual-Head TGCN

```
Input Features (13 dimensions)
    ↓
TGCN Backbone (Shared Feature Extraction)
    - TGCN Layer 1
    - TGCN Layer 2
    ↓
[Shared Hidden Representation]
    ├────────────────────────────┐
    ↓                            ↓
Head 1: Signal Detection    Head 2: Direction Prediction
Linear(16 → 2)              Linear(16 → 2)
  ↓                            ↓
Output 1: [Noise, Signal]   Output 2: [Down, Up]
```

## Training Strategy: Multi-Task Loss

**Combined Loss Function:**
```
Total Loss = Loss₁(signal) + λ × Loss₂(direction)

Where:
- Loss₁ = CrossEntropyLoss(logits₁, signal_targets)
- Loss₂ = CrossEntropyLoss(logits₂, direction_targets)
- λ = task weight balancing parameter (default: 1.0)
```

**Key Points:**
- Both tasks trained simultaneously on all samples
- Shared backbone learns features useful for BOTH tasks
- Gradient flows through both heads to optimize backbone
- Acts as implicit regularization (multi-task learning effect)

## Label Structure

### Training Data Transformation
For each sample with return R:

**Original 3-class labels (Phase 1):**
```
if R < -0.55%  → Label = 0 (Down)
if -0.55% ≤ R ≤ 0.55% → Label = 1 (Neutral)
if R > 0.55%   → Label = 2 (Up)
```

**New hierarchical labels (Phase 2):**
```
Task 1 (Signal Detection):
  if |R| ≤ 0.55% → Signal_Label = 0 (Noise)
  if |R| > 0.55% → Signal_Label = 1 (Signal)

Task 2 (Direction - only for |R| > 0.55%):
  if R < -0.55% → Direction_Label = 0 (Down)
  if R > 0.55%  → Direction_Label = 1 (Up)
  if |R| ≤ 0.55% → Direction_Label = IGNORE (not trained)
```

## Inference Strategy

**At Test Time:**
```
1. Forward pass through TGCN backbone + 2 heads
2. Get signal_logits and direction_logits
3. signal_pred = argmax(signal_logits)

if signal_pred == 1 (Signal predicted):
    direction_pred = argmax(direction_logits)
    final_pred = [Down or Up]
else (Noise predicted):
    final_pred = Neutral  (or abstain from prediction)
```

## Expected Benefits

1. **Reduced Ambiguity:** Each task has clearer objective
2. **Better Gradient Flow:** Easier optimization with simpler sub-problems
3. **Natural Regularization:** Multi-task learning reduces overfitting
4. **Academic Grounding:** Follows López de Prado's signal detection framework
5. **Interpretability:** Can analyze signal vs noise separately

## Expected Performance Improvements

| Metric | Phase 1 | Phase 2 Target | Improvement |
|--------|---------|----------------|-------------|
| Overall Accuracy | 36.25% | 45-50% | +8-14% |
| Signal Detection Acc | N/A | 70-80% | - |
| Direction Acc (on signal) | N/A | 50-60% | - |
| Down precision | 0.319 | 0.40+ | +0.08+ |
| Up precision | 0.351 | 0.45+ | +0.10+ |
| Neutral handling | 0.789 recall | Better filtering | - |

## Implementation Roadmap

**Phase 2a: Setup & Preparation**
- [ ] Create hierarchical label generator function
- [ ] Design dual-head TGCN architecture model
- [ ] Create Phase 2 training notebook

**Phase 2b: Implementation & Training**
- [ ] Implement training loop with multi-task loss
- [ ] Train hierarchical model on Phase 1 dataset
- [ ] Evaluate per-task and overall metrics

**Phase 2c: Analysis & Optimization**
- [ ] Analyze signal vs noise separation quality
- [ ] Evaluate direction prediction on detected signals
- [ ] Visualize task-specific performance
- [ ] Tune λ (task weight) if needed

## Literature Grounding

This approach is inspired by:
- **López de Prado (2018):** "Advances in Financial Machine Learning" - Signal Detection Framework
- **Caruana (1997):** Multi-Task Learning (shared representations improve generalization)
- **Standard MTL Practice:** Dual-head architectures for related tasks

## Files to Create/Modify

**New Files:**
- `models/TGCN_HierarchicalMT.py` - Dual-head TGCN model
- `phase2_hierarchical.ipynb` - Training notebook for Phase 2

**Modified Files:**
- `models/train.py` - Add multi-task training function
- `doc/PHASE2_ARCHITECTURE.md` - Technical details
- `doc/PHASE2_IMPLEMENTATION_PLAN.md` - Step-by-step guide

## Success Criteria

✅ **Phase 2 is successful if:**
1. Overall accuracy reaches **≥ 45%**
2. Signal detection accuracy ≥ 70%
3. Direction accuracy on signals ≥ 50%
4. Both tasks show improved F1-scores vs Phase 1
5. Confusion matrix shows reduced neutral zone errors

❌ **Phase 2 needs revision if:**
1. Overall accuracy < 40% (worse than Phase 1)
2. Signal detection performs poorly (< 65%)
3. Model collapses to one task (ignores other)
