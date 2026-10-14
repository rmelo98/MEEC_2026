import numpy as np
from PIL import Image
import os

# Loads the dataset
image_folder = '/mnt/D/Projeto/Dados/preprocessed_data'

# Calculate dataset mean and standard deviation ------------------------------------------------------------------------
# https://stackoverflow.com/questions/10365119/mean-value-and-standard-deviation-of-a-very-huge-data-set
sum_rgb = np.array([0.0, 0.0, 0.0])
sum_square_rgb = np.array([0.0, 0.0, 0.0])
num_pixels = 0

# Iterate through each image path
for img_path in os.listdir(image_folder):
    full_path = os.path.join(image_folder, img_path)
    image = np.array(Image.open(full_path)) / 255.0  # Normalize the image

    # Flatten the image and sum the RGB values
    sum_rgb += np.sum(image, axis=(0, 1))
    sum_square_rgb += np.sum(image ** 2, axis=(0, 1))
    num_pixels += image.shape[0] * image.shape[1]

# Calculate mean and std
mu_rgb = sum_rgb / num_pixels
std_rgb = np.sqrt(sum_square_rgb / num_pixels - mu_rgb ** 2)

print("Mean: ", mu_rgb)
print("Standard Deviation: ", std_rgb)
