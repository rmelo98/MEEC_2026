import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


ClassName=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

Folder = '/mnt/D/Projeto/Test_ensemble_voting'
File = 'Mean_densenet121_densenet201_efficientnet_b5_efficientnet_b6_se-resnext101_PNASNet5.csv'
best_model = pd.read_csv(Folder + '/' + File)

# Confusion Matrix -------------------------------------------------------------------------------------------------
cm = confusion_matrix(y_true=best_model['True'], y_pred=best_model['Pred'])
cm_normalized = confusion_matrix(y_true=best_model['True'], y_pred=best_model['Pred'], normalize='true')

# Create subfigures
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

# First heatmap (counting)
sn.heatmap(cm_normalized, annot=cm, cmap='Blues', fmt='g',
           xticklabels=ClassName, yticklabels=ClassName, ax=axes[0], cbar=False)

axes[0].set_title("Contagem")

# Add a dedicated colorbar axis to the right of second heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="5%", pad=0.1)

# Second heatmap (normalized)
sn.heatmap(cm_normalized, annot=cm_normalized, cmap='Blues', fmt='.2f',
           xticklabels=ClassName, yticklabels=ClassName, ax=axes[1], cbar=True, cbar_ax=cax)

axes[1].set_title("Normalizado ao longo das linhas")

plt.tight_layout()
plt.show()
