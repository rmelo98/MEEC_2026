import numpy as np
import cv2 as cv
import pandas as pd
import os


def crop_circle(image_path):
    """
    Receives and image and determines if the apperture is visible. If so it will detect its coordinartes and will
    return a cropped image.

    :param image_path: Image path in string format.
    """
    image = cv.imread(image_path)

    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Binarize image
    threshold = 50
    bin_image = ((gray_image > threshold) * 255).astype('uint8')

    # Morphological close
    kernel_value = round(min(gray_image.shape) * .05)
    kernel = np.ones((kernel_value, kernel_value), np.uint8)
    morph_close = cv.morphologyEx(bin_image, cv.MORPH_CLOSE, kernel, iterations=1)

    # Blur image
    blurred_image = cv.GaussianBlur(morph_close, (201, 201), cv.BORDER_DEFAULT)

    # Hough transform
    circles = cv.HoughCircles(blurred_image,
                              cv.HOUGH_GRADIENT, dp=1, minDist=int(min(gray_image.shape) * .1),
                              param1=200, param2=40,
                              minRadius=round(min(gray_image.shape) * .1), maxRadius=round(max(gray_image.shape) / 2))

    # Find largest circle
    if circles is None:
        return

    else:
        largest_circle = np.array([0, 0, 0])
        for i in range(circles.shape[1]):
            radius = circles[0, i, 2]
            if radius > largest_circle[2]:
                largest_circle = circles[0, i]

    # Crop circle
    largest_circle = largest_circle.astype('int')
    w = int(largest_circle[0])
    h = int(largest_circle[1])
    r = int(largest_circle[2])

    # Get coordinates for the bounding box
    x = max(0, w - r)  # x of top left corner
    y = max(0, h - r)  # y of top left corner

    xx = min(image.shape[1], w + r)  # x of bottom right corner
    yy = min(image.shape[0], h + r)  # y of bottom right corner

    cropped_img = image[y:yy, x:xx, :]

    return cropped_img

def color_constancy(img, power=6):
    """
    # https://github.com/nickshawn/Shades_of_Gray-color_constancy_transformation/blob/master/color_constancy.py
    # https://pdfs.semanticscholar.org/acf3/6cdadfec869f136602ea41cad8b07e3f8ddb.pdf

    Receives an image in the numpy array format and returns the color corrected version of the image using the shades
    of gray method.

    :param img: image in numpy array format
    :param power: Minkoswki norm value (recommended 6)
    """

    img_dtype = img.dtype
    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Normalize the image to ensure values are between 0 and 255
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max != img_min:
        img = (img - img_min) / (img_max - img_min) * 255

    return img.astype(img_dtype)

# Data path
image_folder_path = '/mnt/D/Projeto/Dados/raw_data/'  # Image path
dataframe = pd.read_csv('/mnt/D/Projeto/Dados/Ground_truth.csv').image  # Get image names dataframe

# Create output directory
save_path = '/mnt/D/Projeto/Dados/preprocessed_data/'
os.makedirs(save_path, exist_ok=True)

# Progress display
dataset_size = len(dataframe)
count = 1
print(f'{count}/{dataset_size}')

count_crop = 0
for image_name in dataframe:
    if count % 100 == 0:
        print(f'{count}/{dataset_size}')
    count += 1

    # Apply preprocessing to images
    full_image_path = image_folder_path + image_name + '.jpg'
    crop_result = crop_circle(full_image_path)
    if crop_result is not None:                                                                  # If image was cropped
        count_crop += 1
        crop_result = color_constancy(crop_result)
        cv.imwrite(save_path + image_name + '.jpg', crop_result)
    else:                                                                                    # If image was not cropped
        original_image = cv.imread(full_image_path)
        colour_corrected_image = color_constancy(original_image)
        cv.imwrite(save_path + image_name + '.jpg', colour_corrected_image)

print(f' --- {count_crop} circles were cropped ---')
