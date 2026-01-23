# Output Shape Design - Detailed Explanation

## The Design: Per-Stock Trend Prediction with Graph-Aware Features

### Intended Purpose ✓

The model is designed to:
```
For each stock (node) in the portfolio:
├─ Predict its future trend: Up / Neutral / Down
├─ Use TGCN to capture INTER-STOCK relationships:
│  ├─ Stock correlations (edges in graph)
│  ├─ How one stock's movement influences others
│  └─ Collective market patterns
└─ Benefit from graph structure: Better features via GCN/GAT

Output: (100, 3)
├─ 100 = one prediction per stock
└─ 3 = three classes (Down/Neutral/Up)
```

### Why This Matters Over Traditional Methods

**Traditional Methods** (statistics, standard ML):
```
Each stock analyzed independently
├─ Ignore inter-stock relationships
├─ Miss correlation patterns
└─ Treat market as 100 independent time series
```

**TGCN Approach** (graph-aware):
```
Each stock analyzed with relationship context
├─ Stock A influences Stock B's classification
├─ Sector relationships captured via graph
├─ Stock B's features include Stock A's patterns
└─ Better predictions through collective intelligence
```

---

## Data Flow with Intended Purpose

### 1. **Graph Structure Captures Stock Relationships**

```python
# Hybrid adjacency matrix
edge_index, edge_weight = hybrid_adj.npy  # Correlation-based

# Example relationships:
# Apple (AAPL) connects to Microsoft (MSFT) - tech sector
# Tesla (TSLA) connects to Ford (F) - automotive
# Energy stocks form their own cluster

# Each edge represents: "This stock's movement correlates with that one"
```

### 2. **Per-Stock Features are Enriched by Graph**

```
GCN/GAT Processing:

For Stock AAPL:
├─ Its own features (price, volume, etc.)
├─ + Features from connected stocks (MSFT, NVIDIA, etc.)
├─ + How AAPL relates to those stocks (edge weights)
└─ Result: Enriched representation capturing ecosystem

Output for AAPL: (hidden_size=16) better features

Then classify: [AAPL_features] → Linear → (3 classes)
```

### 3. **Graph-Aware Predictions**

```python
# Without graph:
AAPL trend = f(AAPL_time_series)  # Ignore other stocks

# With TGCN:
AAPL trend = f(AAPL_time_series + context_from_related_stocks)
           = f(temporal_patterns + sector_relationships + correlations)
```

---

## Why Output is (100, 3) Not (1, 3)

| Aspect | Per-Stock (Current) | Market-Level (Not Used) |
|--------|-------------------|------------------------|
| **Prediction** | 100 individual stock trends | 1 portfolio trend |
| **Use case** | Decide which stocks to buy/hold/sell | Decide overall portfolio stance |
| **Output shape** | (100, 3) per sample | (1, 3) per sample |
| **Graph importance** | HIGH - drives individual predictions | MEDIUM - aggregate signal |
| **Business value** | "Stock X will go Up" | "Market is Up" |

**Your use case**: Per-stock predictions → More actionable for portfolio management

---

## Loss Computation Makes Sense

```python
# train.py
out = model(data.x, data.edge_index, data.edge_weight)  # (100, 3)
labels = data.y.long()  # (100,)
loss = criterion(out, labels)  # CrossEntropyLoss

# When batched (batch_size=32):
out: (3200, 3)     # 32 samples × 100 stocks each
labels: (3200,)    # Label for each stock in batch

# Loss computation:
# For each stock in the batch:
#   1. Compare its prediction to its individual label
#   2. Compute cross-entropy loss
# Average all 3200 losses

# This is CORRECT because:
# - Each stock has independent label (based on its individual return)
# - GCN/GAT features leverage graph, but losses are per-stock
# - Averaging across batch balances different samples
```

---

## Data Flow Analysis

### 1. **Dataset Level**

Each sample in the dataset (`dataset[i]`) is a PyTorch Geometric `Data` object representing **one time snapshot** containing:

```python
# From trend_classification.ipynb
Data(
    x: (100, 100, T)           # 100 stocks × 100 features × T timesteps
    edge_index: (2, E)         # Graph edges
    edge_weight: (E,)          # Edge weights
    y: (100,)                  # Label for EACH stock
    returns: (100,)            # Return for EACH stock
)

# Labels created per-stock:
sample.y[sample.returns < -0.55%] = 0      # Down
sample.y[sample.returns >= -0.55% & <= 0.55%] = 1  # Neutral
sample.y[sample.returns > 0.55%] = 2       # Up
```

**Key insight**: Each of the 100 stocks gets its **own individual label** based on its individual return.

### 2. **Model Forward Pass**

```python
def forward(self, x, edge_index, edge_weight):
    # x: (100, 100, T)
    # Process all timesteps...
    # Final hidden state: (100, 16)
    return self.out(h_final)  # Linear(16 → 3) → (100, 3)
```

**Output**: `(100, 3)` - predictions for each stock across 3 classes

### 3. **DataLoader Batching**

```python
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

When PyTorch Geometric batches 32 `Data` objects:
- Node-wise concatenation: 32 samples × 100 stocks = **3200 nodes**
- Output shape becomes: `(3200, 3)`
- Target shape becomes: `(3200,)` - one label per stock

### 4. **Loss Computation**

```python
# train.py
out = model(data.x, data.edge_index, data.edge_weight)  # (3200, 3)
labels = data.y.long()  # (3200,)
loss = criterion(out, labels)  # CrossEntropyLoss
```

**Loss calculation**:
```
CrossEntropyLoss expects:
  - Input: (N, C) = (3200, 3)
  - Target: (N,) = (3200,)
  
Loss = mean(loss for each of 3200 nodes)
```

---

## Data Flow Details

### Dataset Structure

```python
Each sample (Data object):
├─ x: (100, 100, T)        # 100 stocks, 100 features each, T timesteps
├─ edge_index: (2, E)      # Graph connections (stock correlations)
├─ edge_weight: (E,)       # Strength of correlations
├─ y: (100,)               # Label for EACH stock (Down/Neutral/Up)
└─ returns: (100,)         # Individual return for EACH stock
```

### Model Processing

```python
# Forward pass
output = model(x, edge_index, edge_weight)  # → (100, 3)

# Each stock gets a prediction vector:
# Stock 0 prediction: [prob_Down, prob_Neutral, prob_Up]
# Stock 1 prediction: [prob_Down, prob_Neutral, prob_Up]
# ...
# Stock 99 prediction: [prob_Down, prob_Neutral, prob_Up]
```

### Batch Training

```python
# Batch size = 32 samples, each with 100 stocks
Batched output: (3200, 3)  # 32 × 100 stocks
Batched labels: (3200,)    # 3200 individual stock labels

# Loss computation:
loss = CrossEntropyLoss(output[3200, 3], labels[3200])
# Compares all 3200 stock predictions to their true labels
# Averages the loss across all 3200 stocks
```

---

## Why This Design is Optimal

### Portfolio-Level Perspective

---

## Conclusion ✓

The **output shape is NOT ambiguous** - it's the **intended design**:

**Model Purpose**: 
```
Predict future trend (Up/Down/Neutral) for EACH stock
using inter-stock relationships via graph structure
```

**Why TGCN solves a real problem**:
- Traditional ML: Each stock analyzed in isolation
- TGCN: Each stock's prediction enriched by related stocks' patterns
- Result: Better predictions from collective market intelligence

**Output shape (100, 3) is correct because**:
- 100 stocks, each needs individual trend prediction
- 3 classes (Down/Neutral/Up) per stock
- Graph structure enables relationship-aware predictions
- Loss averaged across all stocks captures collective learning

The model is well-designed for its intended purpose.
