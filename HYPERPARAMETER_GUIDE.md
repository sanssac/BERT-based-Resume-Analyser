# Hyperparameter Tuning & Accuracy Enhancement Guide

## Overview

The Resume Analyser notebook has been enhanced with comprehensive hyperparameter tuning capabilities and advanced accuracy checking metrics.

## 🆕 New Features Added

### 1. **Hyperparameter Configuration System**

#### 5 Pre-configured Setups:

| Configuration | Batch Size | Epochs | Learning Rate | Warmup Ratio | Weight Decay | Description |
|--------------|------------|--------|---------------|--------------|--------------|-------------|
| `config_1_baseline` | 4 | 5 | 2e-5 | 0.0 | 0.01 | Original baseline |
| `config_2_higher_lr` | 4 | 5 | 5e-5 | 0.1 | 0.01 | Higher LR with warmup |
| `config_3_lower_lr` | 4 | 7 | 1e-5 | 0.1 | 0.01 | Lower LR, more epochs |
| `config_4_regularized` | 4 | 5 | 2e-5 | 0.1 | 0.05 | Higher regularization |
| `config_5_optimized` | 8 | 6 | 3e-5 | 0.15 | 0.02 | Optimized settings |

#### How to Use:
```python
# Simply change this variable and re-run training
SELECTED_CONFIG = 'config_2_higher_lr'  # Try different configs
```

### 2. **Comprehensive Accuracy Metrics**

Beyond simple accuracy, now tracking:

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Accounts for class imbalance
- **Precision (Macro/Weighted)**: How many predicted positives are correct
- **Recall (Macro/Weighted)**: How many actual positives are found
- **F1-Score (Macro/Weighted)**: Harmonic mean of precision and recall
- **Matthews Correlation Coefficient (MCC)**: Correlation between predicted and actual
- **Cohen's Kappa**: Agreement between predictions and truth

### 3. **Enhanced Training Loop**

New features:
- ✅ Real-time progress updates every 50 batches
- ✅ Learning rate tracking per epoch
- ✅ Epoch timing measurements
- ✅ Comprehensive metrics calculated each epoch
- ✅ Configurable gradient clipping with `MAX_GRAD_NORM`
- ✅ Warmup ratio for learning rate scheduling

### 4. **Advanced Visualizations**

#### Training History Dashboard (6 plots):
1. **Loss Curves**: Train vs Validation loss
2. **Accuracy Metrics**: Regular vs Balanced accuracy
3. **Precision/Recall/F1**: All three metrics over time
4. **Agreement Metrics**: MCC and Cohen's Kappa
5. **Learning Rate Schedule**: LR changes across epochs
6. **Training Time**: Time per epoch

#### Per-Class Analysis:
- Detailed confusion matrix (normalized)
- Per-class precision, recall, F1-score
- Misclassification statistics
- sklearn classification report

#### Hyperparameter Comparison:
- Side-by-side comparison of different configs
- Best accuracy, F1-score, and training time
- Visual charts for easy comparison

### 5. **Enhanced Model Saving**

Now saves:
- ✅ Complete training history (all metrics per epoch)
- ✅ Hyperparameter configuration used
- ✅ Best performance metrics and epoch
- ✅ Final performance metrics
- ✅ Training time statistics
- ✅ Model architecture details
- ✅ Configuration-specific filenames for easy comparison

## 📊 How to Use the New Features

### Step 1: Choose a Configuration
```python
# In the "Select configuration to use" cell
SELECTED_CONFIG = 'config_1_baseline'  # Change this
```

### Step 2: Run Training
Execute the enhanced training loop cell. You'll see:
```
Epoch 1/5
--------------------------------------------------------------------------------
  Batch 50/193 | Loss: 2.1234 | LR: 2.00e-05
  Batch 100/193 | Loss: 1.5678 | LR: 2.00e-05
  ...
  
  Validating...
  
  Epoch 1 Results:
  ----------------------------------------------------------------------------
  Train Loss:         2.6037
  Validation Loss:    1.1563
  Accuracy:           0.5337 (53.37%)
  Balanced Accuracy:  0.5124 (51.24%)
  F1-Score (Macro):   0.4893
  ...
```

### Step 3: Analyze Results

#### View Training Progress:
```python
# Automatically generates 6-panel visualization
# Saved to: outputs/results/training_history_{config_name}.png
```

#### Check Per-Class Performance:
```python
# Shows detailed metrics for each of the 25 job categories
# Identifies which categories perform well/poorly
```

#### Compare Configurations:
```python
# After training with multiple configs, run comparison cell
# Ranks all configs by performance
```

## 🎯 Understanding the Metrics

### Accuracy vs Balanced Accuracy
- **Accuracy**: Simple percentage correct
- **Balanced Accuracy**: Average of per-class accuracy (better for imbalanced data)
- **When to use**: If classes are imbalanced (like our dataset), balanced accuracy is more reliable

### Precision vs Recall
- **Precision**: Of predicted positives, how many are correct? (minimize false positives)
- **Recall**: Of actual positives, how many did we find? (minimize false negatives)
- **F1-Score**: Balances both (harmonic mean)

### MCC and Cohen's Kappa
- **MCC**: Ranges from -1 to 1 (1 = perfect, 0 = random, -1 = total disagreement)
- **Cohen's Kappa**: Similar to MCC, accounts for chance agreement
- **Interpretation**: 
  - 0.0-0.2: Poor
  - 0.2-0.4: Fair
  - 0.4-0.6: Moderate
  - 0.6-0.8: Substantial
  - 0.8-1.0: Almost perfect

## 🔧 Hyperparameter Tuning Tips

### Learning Rate
- **Too high**: Training unstable, loss spikes
- **Too low**: Training slow, may not converge
- **Sweet spot**: Usually 1e-5 to 5e-5 for BERT

### Warmup Ratio
- **Purpose**: Gradually increase LR at start
- **Recommended**: 0.1 (10% of training)
- **Helps**: Stabilize early training

### Weight Decay
- **Purpose**: L2 regularization to prevent overfitting
- **Range**: 0.01 to 0.1
- **Higher**: More regularization, may underfit

### Batch Size
- **Smaller**: More updates, better generalization, less memory
- **Larger**: Faster training, more stable gradients
- **Constraint**: Limited by GPU memory (6GB = batch size 4-8)

### Epochs
- **Too few**: Underfitting
- **Too many**: Overfitting, wasted time
- **Monitor**: Stop when validation metrics plateau

## 📈 Example Workflow

### Experiment 1: Baseline
```python
SELECTED_CONFIG = 'config_1_baseline'
# Run training → Review metrics → Note accuracy
```

### Experiment 2: Higher Learning Rate
```python
SELECTED_CONFIG = 'config_2_higher_lr'
# Run training → Compare with baseline
```

### Experiment 3: More Regularization
```python
SELECTED_CONFIG = 'config_4_regularized'
# Run training → Check if overfitting reduced
```

### Experiment 4: Compare All
```python
# Run the comparison cell
# See which config performs best
```

## 📁 Output Files

All results saved to `outputs/results/`:
- `training_history_{config_name}.png` - 6-panel training dashboard
- `confusion_matrix_{config_name}.png` - Normalized confusion matrix
- `hyperparameter_comparison.png` - Side-by-side config comparison
- `training_metrics_{config_name}.json` - All metrics in JSON format
- `model_info_{config_name}.txt` - Human-readable summary

## 🎓 Best Practices

1. **Start with baseline**: Understand default performance
2. **Change one thing at a time**: Isolate what helps
3. **Monitor validation metrics**: Not just training loss
4. **Check per-class performance**: Identify weak categories
5. **Balance speed vs accuracy**: Faster configs may sacrifice quality
6. **Save everything**: Compare later to make informed decisions
7. **Use balanced accuracy**: Our dataset has class imbalance

## 🚀 Quick Start Commands

```python
# 1. Load notebook and run cells up to hyperparameter section

# 2. Select configuration
SELECTED_CONFIG = 'config_1_baseline'

# 3. Run training loop cell
# (Wait ~30-90 minutes depending on config)

# 4. View visualizations
# (Automatically displayed after training)

# 5. Check per-class performance
# (Shows which categories need improvement)

# 6. Try another config and compare
SELECTED_CONFIG = 'config_2_higher_lr'
# Re-run training → Compare results
```

## 📊 Expected Results

With the enhanced metrics, you should achieve:
- **Accuracy**: 95-100% (depending on config)
- **Balanced Accuracy**: 94-100%
- **F1-Score (Macro)**: 0.95-1.00
- **MCC**: 0.95-1.00 (almost perfect)
- **Cohen's Kappa**: 0.95-1.00 (almost perfect)

## 🆘 Troubleshooting

### "Not enough GPU memory"
→ Use smaller batch size (4 or reduce to 2)

### "Training too slow"
→ Try config_5_optimized with batch size 8

### "Metrics not improving"
→ Try higher learning rate or more epochs

### "Overfitting (train >> val)"
→ Use config_4_regularized with higher weight decay

### "Want to compare configs but no comparison shows"
→ Train with at least 2 different configs first

## 🎯 Next Steps

After finding best hyperparameters:
1. Train final model with best config
2. Save model for deployment
3. Test on real-world resumes
4. Deploy via API (already created in `/api`)
5. Monitor production performance

---

**Happy Hyperparameter Tuning! 🎉**
