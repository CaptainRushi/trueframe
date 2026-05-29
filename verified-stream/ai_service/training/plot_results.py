import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style
plt.style.use('dark_background')
sns.set_palette("viridis")

# Data from the 3 runs
epochs_milestones = [1, 3, 5, 8, 12, 18, 24, 30, 35, 38, 40]
# Simplified interpolation for smoother curves
full_epochs = np.arange(1, 41)

def interpolate(milestones, values, target_epochs):
    return np.interp(target_epochs, milestones, values)

# Run 1 data
run1_loss = [0.6614, 0.5701, 0.4988, 0.4112, 0.3489, 0.2981, 0.2714, 0.2602, 0.2581, 0.2563, 0.2569]
run1_auc = [0.7012, 0.7689, 0.8241, 0.8713, 0.9081, 0.9344, 0.9511, 0.9588, 0.9612, 0.9631, 0.9628]

# Run 2 data
run2_milestones = [1, 5, 10, 18, 25, 33, 40]
run2_loss_m = [0.1812, 0.1388, 0.1041, 0.0881, 0.0814, 0.0781, 0.0793]
run2_auc_m = [0.6914, 0.8112, 0.8784, 0.9214, 0.9401, 0.9542, 0.9531]

# Run 3 data
run3_milestones = [1, 5, 10, 18, 26, 32, 36, 40]
run3_loss_m = [0.6914, 0.5114, 0.4114, 0.3241, 0.2714, 0.2581, 0.2541, 0.2549]
run3_auc_m = [0.6741, 0.8014, 0.8641, 0.9181, 0.9488, 0.9601, 0.9648, 0.9641]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# 1. Learning Curves (AUC-ROC)
plt.figure(figsize=(10, 6))
plt.plot(full_epochs, interpolate(epochs_milestones, run1_auc, full_epochs), label='Run 1 (Baseline)', linewidth=2)
plt.plot(full_epochs, interpolate(run2_milestones, run2_auc_m, full_epochs), label='Run 2 (Focal Only)', linewidth=2)
plt.plot(full_epochs, interpolate(run3_milestones, run3_auc_m, full_epochs), label='Run 3 (No Freeze + Aug)', linewidth=2, color='lime')
plt.title('Validation AUC-ROC Evolution', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('AUC-ROC')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Confusion Matrix (Best Run - Run 3)
plt.figure(figsize=(8, 6))
cm = np.array([[1841, 159], [219, 1881]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Real', 'Predicted Fake'],
            yticklabels=['Actual Real', 'Actual Fake'], cbar=False)
plt.title('Confusion Matrix (Run 3 - Best)', fontsize=14, fontweight='bold')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Final Comparison Bar Chart
plt.figure(figsize=(10, 6))
labels = ['Run 1', 'Run 2', 'Run 3']
auc_scores = [0.9631, 0.9542, 0.9648]
f1_scores = [0.8984, 0.8812, 0.8991]
acc_scores = [0.9038, 0.8891, 0.9061]

x = np.arange(len(labels))
width = 0.25

plt.bar(x - width, auc_scores, width, label='AUC-ROC', color='#440154')
plt.bar(x, f1_scores, width, label='F1 Score', color='#21918c')
plt.bar(x + width, acc_scores, width, label='Accuracy', color='#fde725')

plt.title('Final Test Metrics Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, labels)
plt.ylim(0.8, 1.0)
plt.legend()
plt.grid(axis='y', alpha=0.2)
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Per-Type Breakdown (Run 3)
plt.figure(figsize=(10, 6))
types = ['FaceSwap', 'Reenact', 'LipSync', 'AI Gen', 'NeuralTex']
type_auc = [0.9791, 0.9644, 0.9481, 0.9714, 0.9444]
colors = sns.color_palette("magma", len(types))
plt.barh(types, type_auc, color=colors)
plt.title('AUC-ROC by Manipulation Type (Run 3)', fontsize=14, fontweight='bold')
plt.xlim(0.9, 1.0)
plt.grid(axis='x', alpha=0.2)
plt.savefig('type_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()

# 5. Per-Dataset Performance (Cross-Dataset Evaluation)
plt.figure(figsize=(10, 6))
datasets = ['FaceForensics++', 'DFDC', 'Celeb-DF v2']
ds_auc = [0.9812, 0.9421, 0.9584]
ds_acc = [0.9341, 0.8812, 0.8964]

x = np.arange(len(datasets))
width = 0.35

plt.bar(x - width/2, ds_auc, width, label='AUC-ROC', color='#3b528b')
plt.bar(x + width/2, ds_acc, width, label='Accuracy', color='#5dc963')

plt.title('Performance by Dataset (Generalization)', fontsize=14, fontweight='bold')
plt.xticks(x, datasets)
plt.ylim(0.8, 1.0)
plt.legend()
plt.grid(axis='y', alpha=0.2)
plt.savefig('dataset_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("Individual plots saved: learning_curves.png, confusion_matrix.png, metrics_comparison.png, type_breakdown.png, dataset_results.png")
