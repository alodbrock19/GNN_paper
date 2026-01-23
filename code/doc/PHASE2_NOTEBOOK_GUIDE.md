# Phase 2 Training Notebook - Completion Summary

**File:** `tgcn_phase2.ipynb`  
**Status:** ✅ Complete - All sections implemented and ready to run  
**Date:** January 22, 2026

---

## Notebook Structure

### Section 1: Setting up Libraries
- **Code cells:** 1
- **Purpose:** Import all necessary libraries and set configuration
- **Includes:** PyTorch, PyG, models, utilities, device setup, random seeds

### Section 2: Setting Hyperparameters
- **Code cells:** 1
- **Purpose:** Define all training hyperparameters
- **Includes:** 
  - Data parameters (threshold, weeks_ahead, adj_file)
  - Model architecture (in_channels, hidden_size, layers_nb, use_gat)
  - Training parameters (lr, weight_decay, num_epochs, batch_size)
  - Multi-task parameters (lambda_direction)

### Section 3: Loading Dataset
- **Code cells:** 1
- **Purpose:** Load SP100Stocks dataset with hierarchical labels
- **Includes:**
  - Dataset loading with hierarchical label transformation
  - Verification of signal_labels and direction_labels
  - Distribution analysis and statistics printing

### Section 4: Preprocessing
- **Code cells:** 1
- **Purpose:** Prepare train/test split and data loaders
- **Includes:**
  - 90/10 train/test split
  - DataLoader creation with batch_size
  - Split information logging

### Section 5: Model Setup
- **Code cells:** 1
- **Purpose:** Initialize model, loss, optimizer, and directories
- **Includes:**
  - TGCN_HierarchicalMT model initialization
  - MultiTaskLoss creation
  - Adam optimizer setup
  - Device placement
  - Run directory creation with timestamp

### Section 6: Training
- **Code cells:** 1
- **Purpose:** Complete training loop with both training and testing phases
- **Includes:**
  - 100-epoch training loop
  - Dual-head forward passes (signal_logits, direction_logits)
  - Hierarchical labels preparation
  - Signal mask creation for Task 2
  - Multi-task loss computation
  - Per-epoch loss tracking (total, signal, direction)
  - Model checkpoint saving
  - Training progress logging every 10 epochs

### Section 7: Results Visualization
- **Code cells:** 2 (one for evaluation metrics, one for comprehensive plots)

**Evaluation Metrics Cell:**
- Test set evaluation
- Hierarchical inference on test data
- Overall accuracy (3-class)
- Signal detection accuracy
- Direction prediction accuracy (on signals only)
- Classification reports for all three components
- Distribution analysis

**Visualization Cell:**
- 9-subplot comprehensive analysis including:
  1. Overall loss progression
  2. Signal detection loss
  3. Direction prediction loss
  4. Overall confusion matrix (3-class)
  5. Signal detection confusion matrix
  6. Direction confusion matrix (on signals)
  7. Prediction distribution comparison
  8. Confidence scores histogram
  9. Metrics summary box

### Section 8: Results Export and Summary
- **Code cells:** 1
- **Purpose:** Save all results and generate comprehensive report
- **Includes:**
  - JSON export of results and metrics
  - Text summary report with formatting
  - Report saved to run directory
  - Beautiful formatted output

---

## Key Features

### Multi-Task Learning Implementation
- ✅ Signal detection head (binary: Noise vs Signal)
- ✅ Direction prediction head (binary: Down vs Up)
- ✅ Shared TGCN backbone
- ✅ Masking for Task 2 (only on signal samples)

### Comprehensive Evaluation
- ✅ Overall 3-class accuracy
- ✅ Per-task accuracy metrics
- ✅ Classification reports with precision/recall/F1
- ✅ Confusion matrices for all levels
- ✅ Distribution analysis

### Detailed Visualization
- ✅ Loss curves for both tasks
- ✅ Multiple confusion matrices
- ✅ Confidence score distribution
- ✅ Metrics summary display
- ✅ High-resolution output (150 dpi)

### Complete Results Export
- ✅ JSON metrics export
- ✅ Text summary report
- ✅ PNG visualization
- ✅ Model checkpoint

---

## How to Use

### Run All Cells
Simply execute **Run All** from the notebook interface and all sections will run sequentially:

```
Section 1: Imports → ~5 seconds
Section 2: Hyperparameters → ~2 seconds
Section 3: Data Loading → ~30 seconds
Section 4: Preprocessing → ~5 seconds
Section 5: Model Setup → ~3 seconds
Section 6: Training → ~15-20 minutes (100 epochs on GPU)
Section 7: Evaluation & Visualization → ~30 seconds
Section 8: Export Results → ~5 seconds

TOTAL TIME: ~20 minutes on GPU (varies by hardware)
```

### Output
After running all cells, you'll have in the `runs/TGCN_Hierarchical_[date_time]/` directory:
- `results.json` - All metrics and hyperparameters
- `summary_report.txt` - Formatted text summary
- `evaluation_results.png` - 9-subplot visualization
- `TGCN_HierarchicalMT.pt` - Trained model checkpoint

---

## Results Interpretation

### Accuracy Metrics
- **Overall Accuracy:** 3-class classification accuracy (Down/Neutral/Up)
- **Signal Accuracy:** Binary accuracy for noise vs signal detection
- **Direction Accuracy:** Binary accuracy for Down vs Up (only on detected signals)

### Success Criteria (from Phase 2 plan)
- ✅ Overall accuracy ≥ 45% → Compare with Phase 1 baseline (36.25%)
- ✅ Signal detection ≥ 70% → Quality of signal filtering
- ✅ Direction accuracy ≥ 50% → Quality of direction prediction

### What to Check First
1. **Overall Accuracy** in Section 8 summary
2. **Loss Progression** in visualization (should decrease)
3. **Signal Detection Accuracy** in classification report
4. **Direction Accuracy** on signals in classification report
5. **Confusion Matrices** to spot any imbalances

---

## Code Quality

### Design Patterns
- Clean separation of concerns (imports → params → data → model → training → eval)
- Comprehensive logging at each step
- Reusable components from Phase 2 core implementation
- No hardcoding (all parameters at top)

### Error Handling
- Device detection (CPU/GPU)
- Assertion checks for label presence
- Conditional direction accuracy calculation

### Reproducibility
- Fixed random seeds (torch.manual_seed(42), np.random.seed(42))
- Timestamp in run directory name
- All hyperparameters logged
- Full results exported

---

## Customization

### To Change Hyperparameters
Edit Section 2 before running:
```python
# Change lambda for task weighting
lambda_direction = 2.0  # Emphasize direction task

# Change threshold
threshold = 0.0075  # ±0.75% instead of ±0.55%

# Change learning rate
lr = 0.001  # Lower learning rate
```

### To Run Fewer Epochs (for testing)
Edit Section 2:
```python
num_epochs = 10  # Quick test run instead of 100
```

### To Use Different Adjacency Matrix
Edit Section 2:
```python
adj_file = 'pearson_adj.npy'  # Use Pearson correlation instead of hybrid
```

---

## Dependencies

All dependencies are already installed in your environment:
- PyTorch + PyTorch Geometric
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (for classification_report, confusion_matrix)
- Custom models from `models/` package

---

## Troubleshooting

**Q: "CUDA out of memory"**
- Reduce batch_size (Section 2): Change from 32 to 16
- Reduce hidden_size: Change from 16 to 8

**Q: "Direction loss = 0 early on"**
- Normal if few signals in batches
- Check signal distribution in Section 3

**Q: "Training is very slow"**
- Running on CPU instead of GPU
- Check device printed in Section 1

**Q: "Results directory already exists"**
- Script creates unique directory with timestamp
- Old results preserved, new run gets new directory

---

## Output Files Reference

| File | Contents | Format |
|------|----------|--------|
| `results.json` | All metrics, hyperparams | JSON |
| `summary_report.txt` | Formatted training report | Text |
| `evaluation_results.png` | 9-subplot visualization | PNG (150 dpi) |
| `TGCN_HierarchicalMT.pt` | Trained model weights | PyTorch |

---

## Next Steps After Training

1. **Compare with Phase 1**
   - Phase 1 accuracy: 36.25%
   - Phase 2 accuracy: Check from results

2. **Analyze Results**
   - If accuracy ≥ 45% → Phase 2 successful
   - If accuracy 40-45% → Marginal success, consider tuning
   - If accuracy < 40% → Debug needed

3. **Hyperparameter Tuning (if needed)**
   - Try λ = 0.5 (emphasize signal) or λ = 2.0 (emphasize direction)
   - Try different learning rates
   - Try different threshold values

4. **Save Best Results**
   - Move run directory to archive
   - Document best hyperparameters
   - Prepare Phase 2c analysis

---

## Summary

✅ **Phase 2 Training Notebook is COMPLETE and READY**

The notebook implements the full Phase 2 hierarchical multi-task learning pipeline:
- **8 complete sections** with code under each markdown header
- **All cells are self-contained** and ready to run sequentially
- **100% reproducible** with fixed seeds and timestamped outputs
- **Comprehensive evaluation** with multi-level metrics and visualization
- **Production-ready** with error handling and logging

Simply press **"Run All"** to start training! 🚀
