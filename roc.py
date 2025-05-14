import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# True Labels (1 = functional SNP, 0 = non-functional SNP)
y_true = [1, 1, 0, 0]

# Predicted Scores (higher values indicate higher confidence in functional SNP)
y_pred = [0.9, 0.8, 0.1, 0.2]

# Calculate the ROC curve values
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# Calculate AUC score
auc_score = roc_auc_score(y_true, y_pred)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Line for random classifier (AUC = 0.5)

# Add labels and title
plt.title('ROC Curve (AUC = 1)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')

# Show the plot
plt.show()
