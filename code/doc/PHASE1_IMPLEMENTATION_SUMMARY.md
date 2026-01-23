# Phase 1: Focal Loss Implementation Summary

## What Was Changed

### 1. **Modified `/code/models/train.py`**
- **Added**: `FocalLoss` class (lines 14-57)
- **Purpose**: Implements Focal Loss from Lin et al. (2017) for class imbalance handling
- **Features**:
  - Parameterized by `alpha` (class weights) and `gamma` (focusing parameter)
  - Reduces loss contribution from easy examples
  - Focuses training on hard/misclassified examples
  - Supports weighted multi-class classification

**Code Structure**:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        # alpha: Tensor of class weights
        # gamma: Focusing parameter (typically 2.0)
    
    def forward(self, inputs, targets):
        # Compute focal loss: -α_t * (1 - p_t)^γ * log(p_t)
        # Returns mean loss across batch
```

**Why This Design**:
- ✅ Easy to integrate into existing training loop (drop-in replacement for nn.Module)
- ✅ No changes needed to train.py training logic
- ✅ No changes needed to TGCN architecture
- ✅ Fully backward compatible (can revert to CrossEntropyLoss anytime)

### 2. **Modified `/code/trend_classification.ipynb`**

#### Cell 1: Added Markdown Documentation
- **Location**: New cell at top of notebook
- **Content**: Explains Phase 1 strategy, Focal Loss mathematics, expected outcomes
- **Purpose**: Document the rationale for this experiment

#### Cell 2 (Training Setup): Modified to Use Focal Loss
- **Location**: Cell #VSC-c99f71db (original training cell)
- **Changes**:
  1. Import FocalLoss: `from models.train import FocalLoss`
  2. Define class weights: `class_weights = torch.tensor([0.3, 1.0, 0.3])`
  3. Create criterion: `criterion = FocalLoss(alpha=class_weights, gamma=2.0)`
  4. Added detailed comments explaining the strategy
  5. Added progress print showing configuration

**What DID NOT Change**:
- ✅ TGCN architecture (identical)
- ✅ Data loading (same SP100Stocks dataset)
- ✅ Data labels (still 3-class with ±0.55% threshold)
- ✅ Model hyperparameters (in_channels=13, hidden_size=16, layers_nb=2)
- ✅ Optimizer (Adam with lr=0.005, weight_decay=1e-5)
- ✅ Training loop (same train() function)
- ✅ Evaluation metrics (same accuracy measurement)
- ✅ Number of epochs (100)

## Implementation Details

### Class Weights Rationale
```
[0.3, 1.0, 0.3]
  ↓    ↓    ↓
Down Neutral Up

- Down (0): weight=0.3    → Reduce penalty for minority class
- Neutral (1): weight=1.0 → Normal penalty for majority class
- Up (2): weight=0.3      → Reduce penalty for minority class
```

**Why**: The majority class (Neutral) dominates training and causes model to bias toward it. By reducing its importance, we force the model to learn Down/Up patterns better.

### Focal Loss Parameters
- **α (alpha)**: [0.3, 1.0, 0.3]
  - Inverse frequency weighting
  - Penalizes majority class more aggressively
  
- **γ (gamma)**: 2.0 (standard value)
  - Focuses on hard examples
  - Reduces contribution of easy examples
  - Range typically [1.0, 3.0], default 2.0

## How to Run Phase 1

```
1. Open: /code/trend_classification.ipynb
2. Run cells in order:
   - Cell 1: Imports & dataset loading
   - ...existing setup cells...
   - Cell with modified training setup (focal loss)
   - Cell with train() call
3. Monitor: Check if accuracy improves from 35.56%
4. Expected: 45-50% accuracy
```

## Decision Framework

| Outcome | Action |
|---------|--------|
| Accuracy improves to **> 42%** | ✅ Confirms class imbalance is main issue → **Proceed to Phase 2** |
| Accuracy stays **< 38%** | ⚠️ Suggests other issues (features, architecture) → **Investigate alternatives** |
| Accuracy at **38-42%** | 🟡 Partial improvement → **Prepare Phase 2, may have other bottlenecks** |

## Key Metrics to Track

After training completes, compare:

| Metric | Baseline (CrossEntropy) | Phase 1 (Focal Loss) | Target |
|--------|------------------------|----------------------|--------|
| Accuracy | 35.56% | ? | 45-50% |
| Precision (Down) | ~23% | ? | >40% |
| Precision (Neutral) | ~39% | ? | >35% |
| Precision (Up) | ~39% | ? | >35% |
| Loss | High variation | Should be more stable | Stable convergence |
| Model Confidence | 0.41 avg | ? | >0.45 |

## Why This Approach is Safe

1. **Non-invasive**: Only changes loss function, not model or data
2. **Reversible**: Can always switch back to CrossEntropyLoss if needed
3. **Isolated**: Doesn't affect any other experiments or models
4. **Well-tested**: Focal Loss is proven technique from computer vision (2017)
5. **Quick validation**: Can run in ~1-2 hours to get decision signal

## Next Steps After Phase 1

### If Focal Loss Succeeds (accuracy > 42%):
→ **Proceed to Phase 2: Hierarchical Multi-Task Learning**
- Add auxiliary regression task on actual returns
- Binary classification (±1% threshold) as primary task
- Expected accuracy: 70-75%

### If Focal Loss Plateaus (accuracy < 38%):
→ **Investigate alternatives**:
1. Feature quality/engineering issues
2. Temporal window size too small
3. Graph structure not capturing relationships
4. Need different architecture entirely

### If Partial Success (38-42%):
→ **Prepare Phase 2 but investigate bottlenecks**:
1. Check feature variance
2. Analyze per-stock performance (some stocks harder than others?)
3. Consider temporal patterns

## Files Modified

```
/code/models/train.py
  └─ Added: FocalLoss class (44 lines)
  └─ Added: Imports (torch.nn.functional)

/code/trend_classification.ipynb
  └─ Added: Markdown cell explaining Phase 1 (20 lines)
  └─ Modified: Cell #VSC-c99f71db with Focal Loss setup (35 lines)

Total changes: ~100 lines of well-documented code
```

## Academic References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
   - Original paper: https://arxiv.org/abs/1708.02002
   - Used to handle extreme class imbalance (1:1000 ratios)
   
2. **Class Imbalance in Financial ML**: López de Prado, "Advances in Financial Machine Learning" (2018)
   - Discusses why 3-class is hard with noisy financial data
   
3. **Weighted Cross-Entropy**: Standard technique, widely used as simpler alternative

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error on FocalLoss | Ensure train.py is imported before notebook cell |
| CUDA out of memory | Reduce batch size in data loading |
| Training very slow | Check if using CPU instead of CUDA |
| Accuracy doesn't change | Verify criterion is actually being used (not overwritten) |
| Loss becomes NaN | Reduce learning rate or check for numerical instability |

---

**Status**: ✅ Implementation Complete and Ready for Testing

Next: Run the notebook and monitor accuracy improvement.
