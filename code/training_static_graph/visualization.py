import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn.functional as F


def plot_model_predictions(model, test_dataloader, device, threshold, save_path=None):
    """
    Visualize model predictions with comprehensive analysis plots.
    
    Creates 6 subplots showing:
    1. Confusion matrix
    2. Prediction distribution
    3. Prediction probabilities by class
    4. Returns vs predictions scatter plot
    5. Accuracy by return range
    6. Classification report summary
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained model for inference
    test_dataloader : DataLoader
        Test data loader
    device : torch.device
        Device to run inference on (cuda/cpu)
    threshold : float
        Threshold used for classifying returns into classes
    save_path : str, optional
        Path to save the figure. If provided, figure is saved instead of displayed.
        Should be the model path (e.g., '../models/saved_models/model_name.pt').
        The plot will be saved with the same name but .png extension.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Get test data
    test_data = next(iter(test_dataloader))
    test_data = test_data.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(test_data.x, test_data.edge_index, test_data.edge_weight)
        # Use softmax for multi-class classification (3 classes)
        probs = F.softmax(logits, dim=-1)  # Shape: (nodes, 3)
        predictions = probs.argmax(dim=-1)  # Get predicted class (0, 1, or 2)
        actual = test_data.y.squeeze()

    # Flatten predictions and actuals (since we have 100 stocks per sample)
    preds_flat = predictions.cpu().numpy().flatten()
    actuals_flat = actual.cpu().numpy().flatten()
    # Get max probability for each prediction
    probs_max = probs.max(dim=-1)[0].cpu().numpy().flatten()
    # Get probabilities for each class
    probs_down = probs[:, 0].cpu().numpy().flatten()
    probs_neutral = probs[:, 1].cpu().numpy().flatten()
    probs_up = probs[:, 2].cpu().numpy().flatten()

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # 1. Confusion Matrix Heatmap
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(actuals_flat, preds_flat, labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Down (0)', 'Neutral (1)', 'Up (2)'], 
                yticklabels=['Down (0)', 'Neutral (1)', 'Up (2)'])
    ax1.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_xlabel('Predicted', fontsize=12)

    # 2. Prediction Distribution
    ax2 = plt.subplot(2, 3, 2)
    pred_counts = np.bincount(preds_flat.astype(int), minlength=3)
    actual_counts = np.bincount(actuals_flat.astype(int), minlength=3)
    x = np.arange(3)
    width = 0.35
    ax2.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8, color='steelblue')
    ax2.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8, color='coral')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predictions vs Actual', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Down (0)', 'Neutral (1)', 'Up (2)'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Prediction Probabilities Distribution by Class
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(probs_down[actuals_flat == 0], bins=30, alpha=0.5, label='Down (actual)', color='red', edgecolor='black')
    ax3.hist(probs_neutral[actuals_flat == 1], bins=30, alpha=0.5, label='Neutral (actual)', color='gray', edgecolor='black')
    ax3.hist(probs_up[actuals_flat == 2], bins=30, alpha=0.5, label='Up (actual)', color='green', edgecolor='black')
    ax3.set_xlabel('Prediction Probability', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Prediction Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Returns vs Predictions (colored by class)
    ax4 = plt.subplot(2, 3, 4)
    returns_flat = test_data.returns.squeeze().cpu().numpy().flatten()
    # Scatter plot colored by predicted class
    for class_idx, color, label in [(0, 'red', 'Predicted Down'), (1, 'gray', 'Predicted Neutral'), (2, 'green', 'Predicted Up')]:
        mask = (preds_flat == class_idx)
        if mask.sum() > 0:
            ax4.scatter(returns_flat[mask], probs_max[mask], 
                       alpha=0.5, label=label, color=color, s=10)
    ax4.axvline(x=-threshold, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'Threshold: ±{threshold*100:.1f}%')
    ax4.axvline(x=threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax4.set_xlabel('Actual Return', fontsize=12)
    ax4.set_ylabel('Max Prediction Probability', fontsize=12)
    ax4.set_title('Returns vs Prediction Probabilities', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Accuracy by Return Range
    ax5 = plt.subplot(2, 3, 5)
    return_bins = np.linspace(returns_flat.min(), returns_flat.max(), 10)
    bin_indices = np.digitize(returns_flat, return_bins)
    bin_accuracies = []
    bin_centers = []
    for i in range(1, len(return_bins)):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc = (preds_flat[mask] == actuals_flat[mask]).mean()
            bin_accuracies.append(bin_acc)
            bin_centers.append((return_bins[i-1] + return_bins[i]) / 2)
    ax5.plot(bin_centers, bin_accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax5.axhline(y=1/3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Random (33.3%)')
    ax5.set_xlabel('Return Range (center)', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_title('Model Accuracy by Return Range', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.set_ylim([0, 1])

    # 6. Classification Report Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    report = classification_report(actuals_flat, preds_flat, 
                                   target_names=['Down (0)', 'Neutral (1)', 'Up (2)'], 
                                   output_dict=True)
    report_text = f"""
Classification Report

Overall Accuracy: {report['accuracy']:.2%}

Down (0):
  Precision: {report['Down (0)']['precision']:.3f}
  Recall: {report['Down (0)']['recall']:.3f}
  F1-Score: {report['Down (0)']['f1-score']:.3f}

Neutral (1):
  Precision: {report['Neutral (1)']['precision']:.3f}
  Recall: {report['Neutral (1)']['recall']:.3f}
  F1-Score: {report['Neutral (1)']['f1-score']:.3f}

Up (2):
  Precision: {report['Up (2)']['precision']:.3f}
  Recall: {report['Up (2)']['recall']:.3f}
  F1-Score: {report['Up (2)']['f1-score']:.3f}

Macro Avg:
  Precision: {report['macro avg']['precision']:.3f}
  Recall: {report['macro avg']['recall']:.3f}
  F1-Score: {report['macro avg']['f1-score']:.3f}
"""
    ax6.text(0.1, 0.5, report_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        import os
        # Replace .pt extension with .png
        plot_path = save_path.replace('.pt', '_predictions.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    # Always display the plot
    plt.show()

    # Print summary statistics
    print(f"\n{'='*60}")
    print("PREDICTION VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total predictions: {len(preds_flat)}")
    print(f"Overall accuracy: {(preds_flat == actuals_flat).mean():.2%}")
    print(f"\nClass Distribution (Actual):")
    print(f"  Down (0):   {np.sum(actuals_flat == 0):5d} ({np.mean(actuals_flat == 0)*100:5.2f}%)")
    print(f"  Neutral (1): {np.sum(actuals_flat == 1):5d} ({np.mean(actuals_flat == 1)*100:5.2f}%)")
    print(f"  Up (2):     {np.sum(actuals_flat == 2):5d} ({np.mean(actuals_flat == 2)*100:5.2f}%)")
    print(f"\nClass Distribution (Predicted):")
    print(f"  Down (0):   {np.sum(preds_flat == 0):5d} ({np.mean(preds_flat == 0)*100:5.2f}%)")
    print(f"  Neutral (1): {np.sum(preds_flat == 1):5d} ({np.mean(preds_flat == 1)*100:5.2f}%)")
    print(f"  Up (2):     {np.sum(preds_flat == 2):5d} ({np.mean(preds_flat == 2)*100:5.2f}%)")
    print(f"\nAverage max prediction probability: {probs_max.mean():.3f}")
    print(f"Average actual return: {returns_flat.mean():.4f} ({returns_flat.mean()*100:.2f}%)")
    print(f"Threshold used: ±{threshold*100:.2f}%")
    print(f"{'='*60}\n")
