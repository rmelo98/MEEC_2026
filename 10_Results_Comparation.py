import matplotlib.pyplot as plt
import numpy as np

# Data -----------------------------------------------------------------------------------------------------------------
x = np.arange(8)
DA = [0, 81.280159838627, 80.2667267934748, 78.8537713941667, 78.2513493212467, 77.1849602166959, 74.5877992726089, 62.0670921061948]
NDA = [0, 74.7154590197746, 73.006277153, 69.9437496988403, 72.6124837152919, 70.6632044587971, 73.4223288671968, 61.0644867817706]
ensemble = [83.8788, 0, 0, 0, 0, 0, 0, 0]
model_names = ['Ensemble', 'EfficientNet_b6', 'EfficientNet_b5', 'DenseNet201', 'SE-ResNeXt101', 'DenseNet121', 'PNASNet5', 'VGG19']

DA.reverse()
NDA.reverse()
ensemble.reverse()
model_names.reverse()

# Draw Graph -----------------------------------------------------------------------------------------------------------
width = 0.2
plt.barh(x, ensemble, width, color='deepskyblue', linewidth=0.5, edgecolor='black')
plt.barh(x+0.2, DA, width, color='mediumspringgreen', linewidth=0.5, edgecolor='black')
plt.barh(x, NDA, width, color='gold', linewidth=0.5, edgecolor='black')
plt.yticks(x, model_names)
plt.ylabel("Modelos")
plt.xlabel("Exatidão balançada")
plt.legend(["Melhor ensemble", "Aumento de dados", "Sem aumento"], loc='lower right')
plt.xlim(55, 100)
plt.tight_layout()

# Add Bar value
for i in range(len(x)):
    plt.text(ensemble[i] + 1, x[i], f'{ensemble[i]:.2f}', va='center')
    plt.text(DA[i] + 1, x[i] + 0.2, f'{DA[i]:.2f}', va='center')
    plt.text(NDA[i] + 1, x[i]-0.1, f'{NDA[i]:.2f}', va='center')

plt.show()