import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    'Run': ['Run 1 (Baseline)', 'Run 2 (Focal Only)', 'Run 3 (No Freeze)'],
    'Best Epoch': [38, 33, 36],
    'AUC-ROC': [0.9631, 0.9542, 0.9648],
    'F1 Score': [0.8984, 0.8812, 0.8991],
    'Accuracy': [0.9038, 0.8891, 0.9061],
    'EER': [0.0821, 0.0914, 0.0784]
}

df = pd.DataFrame(data)

# Create figure
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
plt.style.use('dark_background')

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center',
                 colColours=['#440154']*6, cellColours=[['#2d2d2d']*6]*3)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5)

# Style headers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_text_props(color='white')

plt.title('TrueFrame Deepfake Detection — Final Training Summary', fontsize=16, fontweight='bold', color='white', pad=20)

# Save
plt.savefig('training_summary_table.png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
print("Summary table saved as training_summary_table.png")
