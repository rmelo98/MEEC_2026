from tools import *
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# Select device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select data path
image_path = '/mnt/D/Projeto/Dados/preprocessed_data'
csv_path = '/mnt/D/Projeto/Dados/Ground_truth.csv'
model_weights_path= '/mnt/D/Projeto/Test_no_augmentation'

# [0-model, 1-image size, 2-batch size, 3-weights_name]
no_augmentation = [
              ['densenet121', 224, 32, 'densenet121_41_72-227814_2026-07-04_052454.pth'],
              ['densenet201', 224, 32, 'densenet201_22_73-054518_2026-07-04_091354.pth'],
              ['efficientnet_b5', 456, 16, 'efficientnet_b5_30_76-593684_2026-07-04_162537.pth'],
              ['efficientnet_b6', 528, 8, 'efficientnet_b6_31_76-780258_2026-07-05_072324.pth'],
              ['vgg19', 224, 32, 'vgg19_89_59-9678_2026-07-05_235248.pth'],
              ['se-resnext101', 224, 32, 'se-resnext101_28_74-189844_2026-07-06_035222.pth'],
              ['PNASNet5', 331, 16, 'PNASNet5_16_75-60722_2026-07-06_095600.pth']
              ]

with_augmentation = [
              ['densenet121', 224, 32, 'densenet121_54_77-497619_2026-07-06_215935.pth'],
              ['densenet201', 224, 32, 'densenet201_68_79-816165_2026-07-07_074024.pth'],
              ['efficientnet_b5', 456, 16, 'efficientnet_b5_74_82-434811_2026-07-08_041713.pth'],
              ['efficientnet_b6', 528, 8, 'efficientnet_b6_58_81-711994_2026-07-09_062927.pth'],
              ['vgg19', 224, 32, 'vgg19_108_61-120684_2026-07-10_034458.pth'],
              ['se-resnext101', 224, 32, 'se-resnext101_53_78-60326_2026-07-10_124450.pth'],
              ['PNASNet5', 331, 16, 'PNASNet5_52_79-15957_2026-07-11_081341.pth']
              ]

best_results = [no_augmentation, with_augmentation]

# Change the current model
for model_list in best_results:
    if model_list == with_augmentation:
        model_weights_path = '/mnt/D/Projeto/Test_with_augmentation'
        print(f"\033[92m{'-'*40} With Augmentation {'-'*40}\033[0m")
    else:
        print(f"\033[92m{'-'*40} No Augmentation {'-'*40}\033[0m")

    for model_config in model_list:
        print('\033[93m' + '\n' + '-'*40 + f' Current Model: {model_config[0]} ' + '-'*40 + '\033[0m')

        # Configure transforms
        test_transform = v2.Compose([CropSquareCenter(), v2.Resize(model_config[1]),  # resize image
                                     v2.ToImage(), v2.ToDtype(torch.float32, scale=True),  # convert to tensor
                                     v2.Normalize(mean=[0.62262518, 0.59810682, 0.58398175],
                                                  std=[0.1995504,  0.22172572, 0.23390127])])  # normalize data

        # Load dataset
        train_dataset = Isic2019Dataset(image_dir=image_path, csv_file=csv_path, transform=test_transform)
        test_dataset = Isic2019Dataset(image_dir=image_path, csv_file=csv_path, transform=test_transform)

        # Divide dataset
        labels = train_dataset.csv_table.iloc[:, 1:].idxmax(axis=1).values
        indices = np.arange(len(train_dataset))

        # Split 60% 20% 20%
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(indices, labels, test_size=0.40, stratify=labels,
                                                                          random_state=123)

        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=123)

        test_set = DataLoader(dataset=test_dataset, batch_size=model_config[2], sampler=SubsetRandomSampler(test_idx))

        # Load Trained Model -------------------------------------------------------------------------------------------
        CurrentModel = custom_model_loader(model_config[0])

        # Load Weights
        weights_path = model_weights_path + '/' + model_config[0] + '/Trained_' + model_config[0] + '/' + model_config[3]
        CurrentModel.load_state_dict(torch.load(weights_path, weights_only=True))

        # Send model to device and set to evaluation mode
        CurrentModel = CurrentModel.to(Device)
        CurrentModel.eval()

        # Make inference -----------------------------------------------------------------------------------------------
        all_y_true = []
        all_y_pred = []

        number_of_batches = len(test_set)
        for current_batch, (inputs, labels) in enumerate(test_set):
            with torch.no_grad():
                if current_batch % 50 == 0:
                    print(f'Progress: {current_batch}/{number_of_batches}')

                inputs, labels = inputs.to(Device), labels.to(Device)
                y_pred = torch.softmax(CurrentModel(inputs), dim=1)

                # Convert predictions to class indices
                pred_classes = torch.argmax(y_pred, dim=1)

                # Convert labels from one-hot encoding to class indices
                true_classes = torch.argmax(labels, dim=1)

                # Accumulate predictions and true labels
                all_y_pred.extend(pred_classes.cpu().numpy())
                all_y_true.extend(true_classes.cpu().numpy())


        # Calculate balanced accuracy score
        balanced_accuracy = balanced_accuracy_score(all_y_true, all_y_pred)
        print(balanced_accuracy)
