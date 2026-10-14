from tools import *
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from PIL import Image
import os

# Displays a random image from each class
def visualize_isic_2019(dataset):
    """
     Display images of each class of the isic dataset (labels must be one hot)
    """

    labels_map = {
        0: "MEL",
        1: "NV",
        2: "BCC",
        3: "AK",
        4: "BKL",
        5: "DF",
        6: "VASC",
        7: "SCC"
    }

    figure = plt.figure(figsize=(15, 8))
    cols, rows = 4, 2
    class_found = set()  # To keep track of what classes found so far
    while len(class_found) < len(labels_map):
        while True:
            sample_idx = torch.randint(len(dataset), size=(1,)).item()  # random number from 1 to max size of dataset
            img, label = dataset[sample_idx]  # Get image and label of random image
            class_index = label.argmax().item()  # Convert one-hot encoded tensor to class from dict
            if class_index not in class_found:  # Only show image if the class was not found yet
                class_found.add(class_index)
                # print(class_found)
                break

        # Plot image with matplotlib
        figure.add_subplot(rows, cols, class_index+1)
        plt.title(labels_map[class_index])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))  # Rearrange from [CxHxW] to [HxWxC] for compatibility with Matplotlib
    plt.show(block=False)


# Display 1 image of each class ----------------------------------------------------------------------------------------
# Loads the dataset
# image_folder = '/mnt/D/Projeto/Dados/raw_data'
image_folder = '/mnt/D/Projeto/Dados/preprocessed_data'
ground_truth_path = '/mnt/D/Projeto/Dados/Ground_truth.csv'


# transform_train = v2.Compose([v2.ColorJitter(0.1, 0.1), # Introduce random brightness and contrast
#                               v2.RandomHorizontalFlip(0.5), # Introduce random horizontal flips
#                               v2.RandomVerticalFlip(0.5), # Introduce random vertical flips
#                               v2.RandomRotation((-30, 30), expand=False), # Introduce random rotation
#                               v2.RandomResizedCrop((256, 256), (1, 1), (1, 1)),
#                               v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]) # Converts images to a tensor

transform_custom_test = v2.Compose([CropSquareCenter(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

# transform_config = transforms.Compose([transforms.ToTensor()])  # converts images to a tensor


isic_dataset = Isic2019Dataset(image_dir=image_folder, csv_file=ground_truth_path, transform=transform_custom_test)

# Display one image of each category
visualize_isic_2019(isic_dataset)


# See class distribution -----------------------------------------------------------------------------------------------
# Make pie chart of how many values are in each category
ground_truth = pd.read_csv(ground_truth_path)
sum_gt = ground_truth.drop(columns='image').sum()
print(sum_gt)

fig, ax = plt.subplots()
ax.pie(sum_gt, labels=sum_gt.index, autopct='%1.1f%%')
plt.show()
