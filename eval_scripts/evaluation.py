import sys
from sklearn.metrics import accuracy_score,confusion_matrix


preds_path = sys.argv[1]
gt_path = 'gymGroundTruthValues/gymValGroundTruth.txt'
gt_labels = []
pred_labels = []
with open(gt_path) as gt, open(preds_path) as pred:
		gt_lines = gt.readlines()
		pred_lines = pred.readlines()
		for i in range(len(pred_lines)):
			pred_labels.append(int(pred_lines[i].split(' ')[1]))
			gt_labels.append(int(gt_lines[i].split(' ')[1]))

print('Top-1 Accuracy = ', accuracy_score(gt_labels,pred_labels))
print('Confusion Matrix = ',confusion_matrix(gt_labels,pred_labels))

