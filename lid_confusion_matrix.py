import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

gt_times = np.loadtxt("ground_truth_switches.txt")
pred_times = np.loadtxt("predicted_switches.txt")

n = min(len(gt_times), len(pred_times))
gt_times = gt_times[:n]
pred_times = pred_times[:n]

true_labels = []
current = 0
for i in range(n):
    true_labels.append(current)
    current = 1 - current

pred_labels = []
for i in range(n):
    if abs(gt_times[i] - pred_times[i]) < 0.2:
        pred_labels.append(true_labels[i])
    else:
        pred_labels.append(1 - true_labels[i])

cm = confusion_matrix(true_labels, pred_labels)

print("Confusion Matrix:\n", cm)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix (LID Switching)")
plt.colorbar()

labels = ["Hindi", "English"]
plt.xticks([0,1], labels)
plt.yticks([0,1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

plt.savefig("results/confusion_matrix.png")
plt.show()

print("Saved as confusion_matrix.png")