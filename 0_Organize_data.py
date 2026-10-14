import os
import pandas as pd
import shutil

Data_path = '/mnt/D/Projeto/Dados'
new_dir_path = os.path.join(Data_path, 'raw_data')
ground_truth = ['ISIC_2019_Test_GroundTruth.csv', 'ISIC_2019_Training_GroundTruth.csv']


df_test = pd.read_csv(os.path.join(Data_path, ground_truth[0]))
df_test = df_test.drop(['score_weight', 'validation_weight'], axis=1)
df_test = df_test[df_test["UNK"] != 1] # drop rows with value UNK
df_test = df_test.reset_index(drop=True)

df_train = pd.read_csv(os.path.join(Data_path, ground_truth[1]))

gt_frame = pd.concat([df_train, df_test]).reset_index(drop=True).drop('UNK', axis=1)
gt_frame.to_csv(os.path.join(Data_path, 'Ground_truth.csv'), index=False)

# Create folder
try:
    os.mkdir(new_dir_path)
    print(f"Directory '{new_dir_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{new_dir_path}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{new_dir_path}'.")
except Exception as e:
    print(f"An error occurred: {e}")

# write images to new path
for data in [[df_test, 'ISIC_2019_Test_Input'], [df_train, 'ISIC_2019_Training_Input']]:
    print(f'----------------- {data[1]} -----------------')
    dataframe = data[0]
    for i in range(dataframe.shape[0]):
        if i % 500 == 0:
            print(i)
        image_name = dataframe.at[i, 'image'] + '.jpg'
        source = os.path.join(os.path.join(Data_path, data[1]), image_name)
        destination = os.path.join(new_dir_path, image_name)
        shutil.copyfile(source, destination)
