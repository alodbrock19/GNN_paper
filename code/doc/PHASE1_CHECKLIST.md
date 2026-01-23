# ✅ Phase 1 Implementation Checklist

## Overview
Phase 1 (Focal Loss Strategy) has been **successfully implemented** to address poor 3-class model performance (35.56% accuracy).

---

## Implementation Checklist

### ✅ Code Changes
- [x] Added `FocalLoss` class to `/code/models/train.py` (lines 14-57)
  - Fully documented with docstrings
  - Implements Lin et al. (2017) Focal Loss formula
  - Supports alpha weighting and gamma focusing parameter
  
- [x] Modified `/code/trend_classification.ipynb` - Cell 11 (Training Setup)
  - Imports FocalLoss from models.train
  - Defines class weights: [0.3, 1.0, 0.3]
  - Creates FocalLoss criterion with gamma=2.0
  - Added detailed comments and configuration logging
  
- [x] Added markdown documentation cell in notebook
  - Explains problem and solution
  - Shows mathematical formula
  - Lists expected outcomes

### ✅ Architecture & Data
- [x] TGCN model architecture **UNCHANGED**
- [x] Data loading **UNCHANGED** (SP100Stocks dataset)
- [x] Data labels **UNCHANGED** (3-class, ±0.55% threshold)
- [x] Training hyperparameters **UNCHANGED** (lr=0.005, epochs=100)
- [x] Model parameters **UNCHANGED** (in_channels=13, hidden_size=16)
- [x] Training loop **UNCHANGED** (same train() function)
- [x] Evaluation metrics **UNCHANGED**

### ✅ Documentation
- [x] Created `/code/PHASE1_IMPLEMENTATION_SUMMARY.md`
  - Detailed explanation of changes
  - Rationale for design choices
  - How to run Phase 1
  - Decision framework for outcomes
  - Troubleshooting guide
  
- [x] Created `/code/analysis/labeling_strategy_recommendations.md`
  - Comprehensive strategy hierarchy
  - Academic references and justifications
  - Phased implementation plan
  - Expected improvements at each phase

### ✅ Reversibility & Safety
- [x] No breaking changes to existing code
- [x] Can revert to CrossEntropyLoss anytime (one line change)
- [x] Doesn't affect other models or experiments
- [x] Fully backward compatible

---

## What Phase 1 Does

| Component | Effect |
|-----------|--------|
| **Model** | None - same TGCN architecture |
| **Data** | None - same 3-class labels, all 100% of data |
| **Loss Function** | **Changed** - CrossEntropyLoss → FocalLoss |
| **Training** | Slightly different - focuses on hard examples |
| **Evaluation** | None - same metrics computed |
| **Result** | Should improve accuracy from 35.56% |

---

## Phase 1 Decision Tree

```
RUN PHASE 1
    ↓
[Train with Focal Loss for 100 epochs]
    ↓
    ├─→ Accuracy improves to > 42%?
    │   ├─ YES → ✅ Confirms class imbalance is bottleneck
    │   │        → PROCEED TO PHASE 2 (Multi-task learning)
    │   │        → Target: 70-75% accuracy with binary + regression
    │   │
    │   └─ NO  → ⚠️ Class imbalance is NOT main issue
    │            → INVESTIGATE alternatives:
    │              • Feature quality/engineering
    │              • Temporal window size
    │              • Graph structure
    │              • Architecture design
    │
    └─→ Accuracy 38-42%?
        └─ PARTIAL SUCCESS → Prepare Phase 2 but
                            investigate other bottlenecks
```

---

## Files Involved in Phase 1

### Modified Files
1. **`/code/models/train.py`**
   - Added: `FocalLoss` class (44 new lines)
   - Changed: Added imports for torch and F
   - Status: ✅ Ready to use

2. **`/code/trend_classification.ipynb`**
   - Added: Markdown documentation cell
   - Changed: Training setup cell to use FocalLoss
   - Status: ✅ Ready to run

### Documentation Files
1. **`/code/PHASE1_IMPLEMENTATION_SUMMARY.md`**
   - Detailed implementation guide
   - How to run, what to expect
   
2. **`/code/analysis/labeling_strategy_recommendations.md`**
   - Full strategy hierarchy
   - Phased implementation plan

---

## How to Run Phase 1

### Quick Start
```python
# In notebook, cells execute in order:
1. Dataset loading & setup (existing)
2. Model initialization (modified with FocalLoss)
3. Train (existing train() function, now uses FocalLoss)
4. Evaluate results

Expected output:
  - Training loss should stabilize faster (Focal Loss is more stable)
  - Final accuracy should improve from baseline 35.56%
  - Target: 45-50% accuracy
```

### Monitoring Progress
```python
# Watch for:
- Loss convergence (should be smoother than before)
- Accuracy improvement per epoch
- Class predictions becoming more balanced
  - If Down predictions increase from 22.4% → closer to 31.65%
  - If Neutral predictions decrease from 38.8% → closer to 33.57%
```

### After Training
```python
# Key metrics to check:
1. Overall accuracy (vs. 35.56% baseline)
2. Per-class precision (Down, Neutral, Up)
3. Confusion matrix (class predictions now more balanced?)
4. Training stability (did loss converge smoothly?)
```

---

## Expected Outcomes

### Scenario 1: Phase 1 Succeeds (accuracy > 42%)
✅ **Interpretation**: Class imbalance was the main bottleneck
**Next Action**: Implement Phase 2 (Multi-task learning)
```
Accuracy: 35.56% → 45-50%
Confidence: Low → Higher
Class Balance: Poor → Better
Decision: PROCEED to Phase 2
```

### Scenario 2: Phase 1 Plateaus (accuracy < 38%)
⚠️ **Interpretation**: Class imbalance was NOT the main issue
**Next Action**: Investigate alternatives
```
Accuracy: 35.56% → <38%
Root Cause: Likely feature quality, temporal design, or architecture limits
Decision: STOP and investigate before Phase 2
```

### Scenario 3: Partial Success (38-42% accuracy)
🟡 **Interpretation**: Class imbalance is ONE of multiple issues
**Next Action**: Prepare Phase 2 but investigate other bottlenecks
```
Accuracy: 35.56% → 38-42%
Evidence: Some improvement but not as expected
Decision: CONTINUE to Phase 2 but monitor closely
```

---

## Key Parameters

### Focal Loss Configuration
```python
alpha = [0.3, 1.0, 0.3]  # Class weights (penalize majority)
gamma = 2.0              # Focusing parameter (focus on hard examples)

# Interpretation:
# - Down (0): alpha=0.3   → Reduce penalty for minority
# - Neutral (1): alpha=1.0 → Full penalty for majority
# - Up (2): alpha=0.3     → Reduce penalty for minority
#
# - gamma=2.0 means (1-p_t)^2.0
#   - If p_t=0.9 (easy): weight = (1-0.9)^2.0 = 0.01
#   - If p_t=0.5 (hard): weight = (1-0.5)^2.0 = 0.25
#   → Focus 25x more on hard examples
```

### Training Configuration (Unchanged)
```python
in_channels = 13         # Input features per stock
out_channels = 3         # 3-class output (Down/Neutral/Up)
hidden_size = 16         # TGCN hidden state size
layers_nb = 2            # Number of TGCN layers
use_gat = True           # Use GAT instead of GCN
lr = 0.005               # Learning rate
weight_decay = 1e-5      # L2 regularization
num_epochs = 100         # Training epochs
batch_size = 32          # (from data loader)
```

---

## Troubleshooting

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| `ImportError: cannot import FocalLoss` | FocalLoss not available in module | Run `train.py` cell first, or restart kernel |
| Accuracy doesn't improve | FocalLoss not being used | Check that criterion variable shows FocalLoss, not CrossEntropy |
| Training is very slow | Running on CPU instead of GPU | Check device = cuda, model on cuda |
| Loss becomes NaN | Numerical instability | Try reducing learning rate to 0.001 |
| Out of memory error | GPU memory exhausted | Reduce batch size in DataLoader |
| Different accuracy each run | Random seed not set | Set torch.manual_seed(42) before training |

---

## Validation Checklist Before Running

- [ ] Read PHASE1_IMPLEMENTATION_SUMMARY.md
- [ ] Read Phase 1 markdown cell in notebook
- [ ] Verify FocalLoss is imported in training cell
- [ ] Verify class_weights tensor is defined
- [ ] Verify criterion = FocalLoss(...) line exists
- [ ] Ensure model is on GPU (device = cuda)
- [ ] Ensure dataset is loaded
- [ ] Verify num_epochs = 100
- [ ] Ready to interpret results against decision tree above

---

## Success Criteria

✅ **Phase 1 is successful if:**
1. Notebook runs without errors
2. Accuracy improves to **> 42%** (vs. baseline 35.56%)
3. Model training converges smoothly
4. No NaN or overflow errors
5. Class predictions become more balanced
6. Training loss shows stable convergence pattern

---

## What Happens Next

### If Phase 1 Succeeds (> 42% accuracy)
→ **Phase 2: Multi-Task Learning**
- Binary classification on ±1% threshold (primary)
- Regression on actual returns (auxiliary)
- Expected: 70-75% accuracy

### If Phase 1 Plateaus (< 38% accuracy)
→ **Stop and Investigate**
- Feature engineering
- Temporal window analysis
- Graph structure evaluation
- Alternative architectures

### If Phase 1 Partial Success (38-42% accuracy)
→ **Prepare Phase 2 + Root Cause Analysis**
- Continue to Phase 2 as planned
- But investigate bottlenecks in parallel
- May need additional feature engineering

---

## Summary

**Phase 1 Implementation**: ✅ **COMPLETE**

- **Focal Loss class**: Added to train.py
- **Notebook integration**: Modified to use FocalLoss
- **Documentation**: Comprehensive guides created
- **Safety**: Fully reversible, no breaking changes
- **Ready to run**: All files prepared

**Next action**: Run the notebook and monitor accuracy improvement.

If you see **> 42% accuracy**, Phase 1 validated the approach and you're ready for Phase 2. ✅
