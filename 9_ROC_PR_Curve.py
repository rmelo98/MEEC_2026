import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import numpy as np

Inference_data_path = '/mnt/D/Projeto/Ensemble_inference'
best_ensemble = ['densenet121', 'densenet201', 'efficientnet_b5', 'efficientnet_b6', 'se-resnext101', 'PNASNet5']
ClassName=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

ensemble_results_path = '/mnt/D/Projeto/Test_ensemble_voting/Mean_efficientnet_b5_efficientnet_b6.csv'
best_model_ensemble_data = pd.read_csv(ensemble_results_path)

# get average inference values -------------------------------------------------------------------------------------
data_amount = best_model_ensemble_data.shape[0]
data = pd.DataFrame(index=range(data_amount), columns=best_ensemble)

for model in best_ensemble:
    read_data_path = os.path.join(Inference_data_path, model + '.json')
    current_model = pd.read_json(read_data_path)
    data[model] = current_model['Pred']


average_inference = pd.DataFrame()
for line in range(data_amount):
    a = np.asarray(data.iat[line, 0], dtype=np.float32)
    b = np.asarray(data.iat[line, 1], dtype=np.float32)

    line_mean = pd.DataFrame(((a + b) / 2).reshape(1, -1))
    average_inference = pd.concat([average_inference, line_mean])

average_inference = average_inference.reset_index(drop=True)

# Draw Precision Recall Curve ---------------------------------------------------------------------------------------

ground_truth_bin = label_binarize(best_model_ensemble_data['True'], classes=[0, 1, 2, 3, 4, 5, 6, 7])
pred_score = np.array(average_inference)

precision = {}  # precisao
recall = {}  # Sensibilidade
average_precision = {}

for i in range(ground_truth_bin.shape[1]):
    precision[i], recall[i], _ = precision_recall_curve(ground_truth_bin[:, i], pred_score[:, i])
    average_precision[i] = average_precision_score(ground_truth_bin[:, i], pred_score[:, i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'cyan', 'brown', 'yellow', 'magenta', 'purple']

for i, color in zip(range(ground_truth_bin.shape[1]), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'{ClassName[i]} (Precisão média = {average_precision[i]:.2f})')

plt.xlabel('Sensibilidade', fontsize=15)
plt.ylabel('Precisão', fontsize=15)
# plt.title('Curva de Precisão-Sensibilidade')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower left', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# Draw ROC -------------------------------------------------------------------------------------------------------------
fpr = {}
tpr = {}
roc_auc = {}

# Calculate ROC and AUC
for i in range(ground_truth_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(ground_truth_bin[:, i], pred_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))

for i, color in zip(range(ground_truth_bin.shape[1]), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f'{ClassName[i]} (AUC = {roc_auc[i]:.2f})'
    )

# Random classifier reference
plt.plot([0, 1], [0, 1], 'k--', lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Rácio falsos positivos', fontsize=15)
plt.ylabel('Rácio de verdadeiros positivos', fontsize=15) # Sensitivity
# plt.title('Curva ROC')
plt.legend(loc='lower right', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()
