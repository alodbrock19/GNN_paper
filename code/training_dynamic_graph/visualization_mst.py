import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn.functional as F


def plot_loss_curves(train_losses, test_losses, save_path=None, title="Training and Testing Loss"):
    """
    Visualize training and testing loss curves over epochs.
    
    Parameters:
    -----------
    train_losses : list or array
        Training loss values for each epoch
    test_losses : list or array
        Testing loss values for each epoch
    save_path : str, optional
        Path to save the figure. If provided, figure is saved with '_loss_curves.png' suffix.
    title : str, optional
        Title for the plot (default: "Training and Testing Loss")
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # Plot loss curves
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4, alpha=0.7)
    plt.plot(epochs, test_losses, 'r-', linewidth=2, label='Testing Loss', marker='s', markersize=4, alpha=0.7)
    
    # Formatting
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add best epoch marker
    best_test_epoch = np.argmin(test_losses) + 1
    best_test_loss = np.min(test_losses)
    plt.plot(best_test_epoch, best_test_loss, 'g*', markersize=20, label=f'Best Test Loss (Epoch {best_test_epoch})')
    plt.legend(fontsize=11, loc='best')
    
    # Add text annotation
    plt.text(best_test_epoch, best_test_loss + 0.01 * max(test_losses), 
             f'Best: {best_test_loss:.4f}', ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        import os
        # Replace .pt extension with _loss_curves.png
        plot_path = save_path.replace('.pt', '_loss_curves.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Loss curves plot saved to: {plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("LOSS CURVES SUMMARY")
    print(f"{'='*60}")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final testing loss: {test_losses[-1]:.4f}")
    print(f"Best testing loss: {best_test_loss:.4f} (Epoch {best_test_epoch})")
    print(f"Initial training loss: {train_losses[0]:.4f}")
    print(f"Initial testing loss: {test_losses[0]:.4f}")
    print(f"Training loss improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
    print(f"Testing loss improvement: {((test_losses[0] - best_test_loss) / test_losses[0] * 100):.2f}%")
    print(f"{'='*60}\n")


def plot_model_predictions(model, test_dataloader, device, save_path=None):
    """
    Visualize model predictions with comprehensive analysis plots (adapted for MST).
    
    Creates 5 subplots showing:
    1. Confusion matrix
    2. Prediction distribution
    3. Prediction probabilities by class
    4. Prediction confidence distribution
    5. Classification report summary
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained MST model for inference
    test_dataloader : DataLoader
        Test data loader
    device : torch.device
        Device to run inference on (cuda/cpu)
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

    # Flatten predictions and actuals (since we have multiple stocks per sample)
    preds_flat = predictions.cpu().numpy().flatten()
    actuals_flat = actual.cpu().numpy().flatten()
    # Get max probability for each prediction
    probs_max = probs.max(dim=-1)[0].cpu().numpy().flatten()
    # Get probabilities for each class
    probs_down = probs[:, 0].cpu().numpy().flatten()
    probs_neutral = probs[:, 1].cpu().numpy().flatten()
    probs_up = probs[:, 2].cpu().numpy().flatten()

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))

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

    # 4. Prediction Confidence Distribution
    ax4 = plt.subplot(2, 3, 4)
    correct_mask = (preds_flat == actuals_flat)
    ax4.hist(probs_max[correct_mask], bins=30, alpha=0.7, label='Correct Predictions', color='green', edgecolor='black')
    ax4.hist(probs_max[~correct_mask], bins=30, alpha=0.7, label='Incorrect Predictions', color='red', edgecolor='black')
    ax4.axvline(x=probs_max.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {probs_max.mean():.3f}')
    ax4.set_xlabel('Max Prediction Probability', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Per-Class Accuracy
    ax5 = plt.subplot(2, 3, 5)
    class_accuracies = []
    class_labels = ['Down (0)', 'Neutral (1)', 'Up (2)']
    for class_idx in range(3):
        mask = (actuals_flat == class_idx)
        if mask.sum() > 0:
            acc = (preds_flat[mask] == actuals_flat[mask]).mean()
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0)
    
    bars = ax5.bar(class_labels, class_accuracies, alpha=0.8, color=['red', 'gray', 'green'])
    ax5.axhline(y=1/3, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Random (33.3%)')
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 1])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

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
    print(f"{'='*60}\n")
