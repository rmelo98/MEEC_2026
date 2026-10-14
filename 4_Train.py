from tools import *
from torchvision.transforms import v2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# Select device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select data path
image_path = '/mnt/D/Projeto/Dados/preprocessed_data'
csv_path = '/mnt/D/Projeto/Dados/Ground_truth.csv'

# Setup training -------------------------------------------------------------------------------------------------------

# [0-model, 1-image size, 2-batch size, 3-Optimizer, 4-Learning rate]
model_list = [
              ['densenet121', 224, 32, 'Adam', 0.0001],
              ['densenet201', 224, 32, 'Adam', 0.0001],
              ['efficientnet_b5', 456, 16, 'Adam', 0.0001],
              ['efficientnet_b6', 528, 8, 'Adam', 0.0001],
              ['vgg19', 224, 32, 'Adagrad', 0.0001],
              ['se-resnext101', 224, 32, 'Adam', 0.0001],
              ['PNASNet5', 331, 16, 'Adam', 0.0001]
              ]

# Change the current model
for current_test in ['No Augmentation', 'With Data Augmentation']:
    print('-'*40 + f' Testing for: {current_test} ' + '-'*40)

    for model_config in model_list:
        print('\033[93m' + '\n' + '-'*40 + f' Current Model: {model_config[0]} ' + '-'*40 + '\033[0m')

        # Configure transforms
        DataAug_transform = v2.Compose([v2.ColorJitter(0.1, 0.1),  # Introduce random brightness and contrast
                                        v2.RandomHorizontalFlip(0.5),  # Introduce random horizontal flips
                                        v2.RandomVerticalFlip(0.5),  # Introduce random vertical flips
                                        v2.RandomRotation((-30, 30), expand=False),  # Introduce random rotation
                                        v2.RandomResizedCrop(model_config[1], (0.9, 1), (1, 1)),  # Random crop resize
                                        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),  # Converts image to tensor
                                        v2.Normalize(mean=[0.623, 0.598, 0.583],
                                                     std=[0.196, 0.219, 0.232])])  # Normalize image

        default_transform = v2.Compose([CropSquareCenter(), v2.Resize(model_config[1]),  # resize image
                                        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),  # convert to tensor
                                        v2.Normalize(mean=[0.62262518, 0.59810682, 0.58398175],
                                                     std=[0.1995504,  0.22172572, 0.23390127])])  # normalize data

        # Load dataset
        if current_test == 'With Data Augmentation':
            train_dataset = Isic2019Dataset(image_dir=image_path, csv_file=csv_path, transform=DataAug_transform)
        else:
            train_dataset = Isic2019Dataset(image_dir=image_path, csv_file=csv_path, transform=default_transform)

        validation_dataset = Isic2019Dataset(image_dir=image_path, csv_file=csv_path, transform=default_transform)

        # Divide dataset
        labels = train_dataset.csv_table.iloc[:, 1:].idxmax(axis=1).values
        indices = np.arange(len(train_dataset))

        # Split 60% 20% 20%
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(indices, labels, test_size=0.40, stratify=labels,
                                                                          random_state=123)

        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=123)

        # Calculate the weights for weighted loss
        weighted_loss = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        weighted_loss = torch.tensor(weighted_loss, dtype=torch.float).to(Device)

         # Setup Data loader
        train_set = DataLoader(dataset=train_dataset,
                               batch_size=model_config[2],
                               sampler=SubsetRandomSampler(train_idx))

        validation_set = DataLoader(dataset=validation_dataset,
                                    batch_size=model_config[2],
                                    sampler=SubsetRandomSampler(val_idx))

        # Load model
        CurrentModel = custom_model_loader(model_config[0]).to(Device)

        # Load optimizer
        if model_config[3] == 'Adam':
            Optimizer = torch.optim.Adam(CurrentModel.parameters(), lr=model_config[4])
        elif model_config[3] == 'Adagrad':
            Optimizer = torch.optim.Adagrad(CurrentModel.parameters(), lr=model_config[4])
        elif model_config[3] == 'RMS':
            Optimizer = torch.optim.RMSprop(CurrentModel.parameters(), lr=model_config[4])
        else:
            print(f"ERROR: optimizer {model_config[3]} not found")
            exit()

        # Configures training parameters
        training_config = TrainingToolkit(device=Device, model=CurrentModel, mixed_precision=True,
                                          training_data=train_set, validation_data=validation_set, model_name=model_config[0],
                                          criterion=torch.nn.CrossEntropyLoss(weight=weighted_loss),
                                          optimizer=Optimizer, number_epochs=200, loss_freq=0, acc_freq=1,
                                          metrics_to_csv=True, model_checkpoint=True, target_metric='val_acc',
                                          early_stop=True, early_stop_patience=20, ignore_train_acc=True)

        # Starts training
        if current_test == 'With Data Augmentation':
            training_config.start_training(custom_folder='Test_with_augmentation')
        else:
            training_config.start_training(custom_folder='Test_no_augmentation')

