# TGCN Model - Hyperparameters & Executed Tasks

## Overview
This document details the hyperparameters and tasks executed in the `trend_classification.ipynb` notebook for the TGCN (Temporal Graph Convolutional Network) model performing 3-class stock trend classification.

---

## 1. Data Configuration

### Classification Task
- **Task Type**: 3-class classification (Down, Neutral, Up)
- **Prediction Window**: 1 week ahead
- **Threshold for Neutral Zone**: 0.55% (±0.55%)
- **Class Definitions**:
  - **Class 0 (Down)**: Return < -0.55%
  - **Class 1 (Neutral)**: -0.55% ≤ Return ≤ 0.55%
  - **Class 2 (Up)**: Return > 0.55%

### Dataset Properties
- **Dataset Source**: S&P 100 Stocks
- **Adjacency Matrix**: `hybrid_adj.npy` (hybrid correlation-based graph)
- **Input Features**: 100 stocks with multiple features
- **Feature Dimensions**: Shape[0].x → 100 features per node
- **Temporal Dimension**: Sequential price data over time

### Data Split
- **Train Set**: 90% of data
- **Test Set**: 10% of data
- **Shuffle**: Enabled for training
- **Batch Size**: 32 samples
- **Test Batch Size**: Entire test set in one batch

---

## 2. Model Architecture

### TGCN Model Configuration
```
Model: TGCN (Temporal Graph Convolutional Network)
Location: models/TGCN.py
```

#### Input/Output Channels
- **Input Channels**: 100 (number of stocks/features)
- **Output Channels**: 3 (number of classes)
- **Hidden Size**: 16
- **Number of Layers**: 2

#### Model Structure
1. **Input Layer**: (Nodes=100, Features=100, TimeSteps=T)
2. **TGCN Cells** (2 layers):
   - Layer 1: TGCNCell(in_channels=100, hidden_size=16)
   - Layer 2: TGCNCell(in_channels=16, hidden_size=16)
3. **Output Layer**: 
   - Linear(hidden_size=16 → 3 classes)
   - Activation: Identity (raw logits for CrossEntropyLoss)

#### TGCNCell Operation (per timestep)
```
For each timestep t:
1. GCN Encoding: gcn_out = sigmoid(GCN(x_t, edge_index, edge_weight))
2. Update Gate: u_t = sigmoid(Linear([x_t, gcn_out, h_{t-1}]))
3. Reset Gate: r_t = sigmoid(Linear([x_t, gcn_out, h_{t-1}]))
4. Candidate: c_t = tanh(Linear([x_t, gcn_out, r_t ⊙ h_{t-1}]))
5. Hidden Update: h_t = u_t ⊙ h_{t-1} + (1 - u_t) ⊙ c_t
```

---

## 3. Training Configuration

### Optimization Hyperparameters
- **Learning Rate**: 0.005
- **Weight Decay (L2 Regularization)**: 1e-5 (0.00001)
- **Optimizer**: Adam
- **Number of Epochs**: 100
- **Device**: CUDA (GPU)

### Loss Function
- **Loss Function**: `CrossEntropyLoss()`
- **Reason**: Multi-class classification (3 classes)
- **Note**: No class weighting applied (potential issue with class imbalance)

---

## 4. Executed Training Tasks

### Training Process
1. **Model Initialization**: TGCN with specified hyperparameters
2. **Device Setup**: Move model to GPU (CUDA device)
3. **Training Loop**: 
   - Function: `train()` from models/train.py
   - Iterations: 100 epochs
   - Metric Tracking: `measure_acc=True`
   - Run Name: "UpDownTrend_hybrid_threshold"
   - Batch Processing: 32 samples per batch
4. **Validation**: Testing on held-out test set after each epoch

### Model Checkpointing
- **Saved Model Path**: `models/saved_models/UpDownTrend_hybrid_threshold_TGCN.pt`
- **Format**: PyTorch state_dict

### Evaluation Metrics Computed
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Predictions vs actuals for all 3 classes
- **Classification Report**:
  - Precision (per class)
  - Recall (per class)
  - F1-Score (per class)
  - Macro Average
- **Visualizations**:
  - Confusion matrix heatmap
  - Prediction distribution
  - Probability distribution by class
  - Returns vs prediction probabilities scatter
  - Accuracy by return range
  - Classification metrics summary

---

## 5. Additional Analysis Performed

### Prediction Analysis
- Distribution of predicted vs actual classes
- Average prediction probabilities
- Model confidence on correct vs incorrect predictions
- Performance breakdown by return magnitude

### Graph Structure
- **Graph Type**: Hybrid correlation-based
- **Nodes**: 100 (stocks)
- **Edges**: Dynamic correlation weights
- **Edge Weights**: Correlation coefficients

---

## 6. Training Runs Summary

### Hybrid Graph with Threshold (Production Model)
- **Configuration**: 3-class classification with 0.55% threshold
- **Model Name**: UpDownTrend_hybrid_threshold
- **Performance**: Evaluated on test set
- **Status**: Saved and loaded for inference

### Alternative Models Also Trained
- **Pearson Graph**: Using Pearson correlation adjacency matrix
- **Hybrid Graph (without threshold)**: Using different threshold configuration

---

## 7. Key Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden Size | 16 | Balance between model capacity and overfitting risk |
| Layers | 2 | Capture multi-level spatial-temporal patterns |
| Learning Rate | 0.005 | Conservative rate for stable convergence |
| Weight Decay | 1e-5 | Mild regularization to prevent overfitting |
| Epochs | 100 | Sufficient iterations for convergence |
| Batch Size | 32 | Standard size for GPU memory efficiency |
| Threshold | 0.55% | Create meaningful neutral zone (±0.55%) |
| Prediction Window | 1 week | Short-term trend prediction |

---

## Notes
- The model processes temporal sequences by iterating through timesteps
- Final hidden state from last TGCN layer is used for classification
- Graph structure is fixed (adjacency matrix) but edge weights may vary
- All 100 stocks' predictions are aggregated for evaluation metrics
