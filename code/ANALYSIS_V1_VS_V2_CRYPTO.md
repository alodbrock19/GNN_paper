# V1 vs V2 Analysis: Cryptocurrency TGCN Results

## 🔴 Critical Findings

### 1. **V2 Loss DIVERGENCE (Most Serious Issue)**
```
V1: Training converges (1.2 → 0.1), but test loss increases (1.2 → 1.6) - OVERFITTING
V2: Both losses INCREASE dramatically (Train: 1.2 → 2.0, Test: 1.2 → 7.0) - DIVERGENCE!
```

**What this means:**
- V2 is NOT learning - it's getting worse with every epoch
- The model is **destabilizing** rather than converging
- This is a fundamental training failure, not just overfitting

### 2. **V2 Does Not Actually Help (Paradoxical Result)**
```
                        V1              V2          Change
Accuracy:              38.88%          31.47%      -7.41% ❌
Macro F1:              0.418           0.215       -0.203 ❌
Down Precision:        0.606           0.247       -0.359 ❌
Neutral Precision:     0.145           0.074       -0.071 ❌
Up Precision:          0.432           0.261       -0.171 ❌
```

**The Irony:**
- V2 successfully fixes the **prediction bias** (Neutral predictions: 1% → 11%)
- But it **worsens overall model performance** significantly
- Trading false positive reduction for worse overall accuracy

### 3. **Why Did V2 Fail? Root Cause Analysis**

#### **A. Hyperparameter Combinations Are Incompatible**
```python
# Changes made in V2:
threshold:        0.55% → 0.75%        # More samples labeled as Neutral
hidden_size:      16 → 32              # Doubled model capacity
learning_rate:    0.005 → 0.003        # REDUCED learning rate
lambda_direction: 1.0 → 2.0            # Doubled task weighting
epochs:           100 → 200             # More epochs
```

**The Problem:**
- **Reduced LR (0.003)** is TOO CONSERVATIVE for crypto volatility
- **Doubled hidden_size** without layer normalization → unstable gradients
- **Increased threshold (0.75%)** creates IMBALANCED label distribution in training set:
  ```
  Old threshold (0.55%): ~equal distribution
  New threshold (0.75%): Neutral class EXPLODES (likely >50% of samples)
  ```
- **Lambda_direction = 2.0** means signal detection is ignored, direction prediction dominates
- 200 epochs allows divergence to compound

#### **B. Cryptocurrency Data is More Volatile**
```
Stock data: ~1-3% daily moves (relatively stable)
Crypto data: ~5-15% daily moves (extreme volatility)
```

**Impact:**
- Thresholds that work for stocks are calibrated for that volatility
- 0.75% threshold on crypto might be TOO LOOSE (too many Neutral samples)
- Model can't distinguish between noise and real signals with unstable thresholds

#### **C. Class Weights May Be Over-Applied**
When using weighted loss + massive class imbalance:
```
Original (0.55%): Weights balance minority classes
New (0.75%):      Neutral becomes MAJORITY, weights become inverted
                  Model now heavily penalizes Down/Up predictions
```

### 4. **Signal Detection Loss is Degenerate**
```
V1 Signal Loss:  0.42 → 0.38 ✓ Good convergence
V2 Signal Loss:  0.875 → 0.880 (FLAT!)
                 Essentially not learning signal detection at all
```

**Why?**
- V2 signal loss is much **higher** (0.88 vs 0.40)
- This suggests the signal labels don't match the model's learned representation
- Noisy/unreliable signal labels in the dataset

## 🎯 Root Causes (Ranked by Impact)

### **#1: Learning Rate Too Low (0.003)**
- Crypto market volatility requires **adaptive, faster learning**
- 0.003 LR is appropriate for smooth stock trends
- For crypto, 0.005-0.01 would be better

### **#2: Threshold Mismatch**
- 0.75% threshold on crypto creates TOO MANY neutral samples
- Likely >50% neutral in training set = majority class
- Model learns "predict Neutral" as default shortcut
- Try threshold = **0.01 (1%)** for crypto instead

### **#3: Lambda_direction = 2.0 is Wrong**
- Weighting direction 2x more than signal detection
- When signal detection is broken (high loss), this makes it worse
- Should be 1.0 or even 0.5 (prioritize signal detection first)

### **#4: Doubled Hidden Size Without Stabilization**
- 16 → 32 neurons without:
  - Layer normalization
  - Batch normalization
  - Gradient clipping
- Leads to **gradient explosion** in crypto's volatile data

### **#5: Training Epochs Too High**
- 200 epochs allows divergence to compound for 100 more steps
- With diverging loss, early stopping is critical
- Stop at epoch ~30-50 when validation loss starts increasing

## ✅ Recommended Fixes

### **Option 1: Conservative V2 (Safest)**
```python
# Keep V1 as baseline
threshold_v3 = 0.006           # 0.6% (between V1 and V2)
hidden_size_v3 = 24            # Between 16 and 32
learning_rate_v3 = 0.004       # Slightly lower than V1, but not too low
lambda_direction_v3 = 1.0      # Equal weighting
num_epochs_v3 = 100            # Match V1
batch_size_v3 = 32             # Same as V1

# Add stabilization
gradient_clip_norm = 1.0       # Clip gradients
use_layer_norm = True          # Add layer normalization
early_stopping_patience = 20   # Stop if val loss increases
```

### **Option 2: Aggressive Crypto-Optimized (Riskier)**
```python
threshold_v3 = 0.008           # 0.8% 
hidden_size_v3 = 16            # Keep smaller for stability
learning_rate_v3 = 0.008       # HIGHER for crypto volatility
lambda_direction_v3 = 0.5      # Prioritize signal detection
num_epochs_v3 = 150            # More epochs, but with early stopping
batch_size_v3 = 64             # Larger batch for stability
```

### **Option 3: Hybrid Multi-Stage**
```python
# Stage 1: Train signal detection well (epochs 0-50)
lambda_direction = 0.1         # Ignore direction, focus on signal

# Stage 2: Fine-tune direction (epochs 50-100)
lambda_direction = 1.0         # Balance both tasks
freeze_signal_layers = True    # Keep signal detector fixed
```

## 🔬 Diagnostic Checks Needed

1. **Check data distribution:**
   ```python
   # In training set, how many samples per class?
   for threshold in [0.005, 0.006, 0.0075, 0.010]:
       count_down = (returns < -threshold).sum()
       count_neutral = ((returns >= -threshold) & (returns <= threshold)).sum()
       count_up = (returns > threshold).sum()
       print(f"Threshold {threshold*100:.2f}%: Down={count_down}, Neutral={count_neutral}, Up={count_up}")
   ```

2. **Check gradient norms during V2 training:**
   ```python
   # Print max gradient norm every epoch
   # Should be stable, not exploding or vanishing
   ```

3. **Check prediction confidence:**
   ```python
   # V2 confidence is lower (0.843 vs 0.835)
   # This indicates model is LESS SURE about predictions
   # Combined with worse accuracy = model is confused
   ```

4. **Compare test set statistics:**
   ```python
   # Check if test set has same distribution as training
   # Crypto data might have train/test distribution shift
   ```

## 📊 Summary Table: The Key Issue

| Aspect | V1 | V2 | Problem |
|--------|----|----|---------|
| **Test Loss Trend** | ↑ (1.2→1.6) | ↑↑ (1.2→7.0) | V2 **diverges**, V1 only **overfits** |
| **Accuracy** | 38.88% | 31.47% | V2 **much worse** |
| **Class Balance** | Fixed (Neutral bias) | Broken (now predicts Neutral) | Neither is ideal |
| **Learning Rate** | 0.005 ✓ | 0.003 ❌ | Too conservative for crypto |
| **Threshold** | 0.55% ✓ | 0.75% ❌ | Creates too much Neutral noise |
| **Lambda** | 1.0 ✓ | 2.0 ❌ | Over-weights unstable task |

## 🎓 Key Lesson: Why V2 Failed

**V2 tried to fix the symptoms (prediction bias) without understanding the disease (data characteristics).**

The real problem isn't that the model is biased toward "Up" - it's that:
1. **Crypto data is hard** - returns are noisy, thresholds are arbitrary
2. **Multi-task learning is fragile** - weight tuning is critical
3. **Hyperparameters interact** - reducing LR while increasing threshold creates divergence
4. **No validation monitoring** - V2 kept training despite diverging loss

**Correct approach:**
1. Start with V1 (stable baseline)
2. Make ONE change at a time
3. Monitor validation loss for early stopping
4. Only change next parameter if previous improvement is stable

## 🚀 Recommendation

**Do NOT use V2.** Instead:
1. Create V3 with Option 1 (Conservative)
2. If that works, incrementally try Option 2 parameters
3. Always use early stopping (patience=20)
4. Always log gradient norms for diagnostics

The cryptocurrency dataset may require a **different architecture entirely** (e.g., LSTM for temporal dependencies, attention for volatility focusing).
