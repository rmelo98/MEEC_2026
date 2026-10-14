import torch, time, os
import pandas as pd
from datetime import datetime
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torchvision.models import vgg19, vgg16, densenet121, densenet201, efficientnet_b5, efficientnet_b6
import torch.hub
import timm


# Train network
class TrainingToolkit:
    """
        Basic setup:
         - device: Device where to train the neural network.
         - model: The torch model used for training.
         - training_data: Dataloader split used for training.
         - validation_data: Dataloader split used for validation.
         - mixed_precision: Boolean to enable automatic mixed precision. It speeds up training without impacting accuracy.
         - criterion: Function used to calculate loss.
         - optimizer: Algorithm used to minimize weights.
         - number_epochs: Maximum number of epochs to training the model.
        
        Metrics:
         - metrics_to_csv: Boolean to enable saving metrics to a csv file.
         - loss_freq: (Loss frequency) how often to calculate validation data loss.
         - acc_freq: (Accuracy frequency) how often to calculate accuracy.
         - ignore_train_acc: Boolean to avoid calculation training accuracy.
         - model_name: String used to change the name used for saving metrics and checkpoints. By default, it will use
           the type of torch model. Useful when training multiple models of the same type (example: "vgg19" and "vgg16"
           have the same torch type "VGG").

        Model checkpoint:
         - model_checkpoint: Boolean to enable model checkpoint. It will save the model weights everytime it improves.
         - target_metric: The metric TrainingToolkit will track. Can be 'train_loss', 'val_loss', 'train_acc', 'val_acc'.

        Early stop:
         - early_stop: Boolean to enable early stopping. It will terminate training once the model stops improving.
         - early_stop_patience: How many epochs without improvement to wait before terminating training.
         - target_metric: The metric TrainingToolkit will track. Can be 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """

    def __init__(self, device=None, model=None, training_data=None, validation_data=None, mixed_precision=False,
                 criterion=None, optimizer=None, number_epochs=1, loss_freq=1, acc_freq=1, metrics_to_csv=False,
                 model_checkpoint=False, target_metric='train_loss', early_stop=False,
                 early_stop_patience=10, ignore_train_acc=False, model_name=''):

        # Basic parameters ---------------------------------------------------------------------------------------------
        self.device = device    # Device where to train nn
        self.model = model  # torch model
        self.training_data = training_data  # data split used for training
        self.validation_data = validation_data  # data split used for validation
        self.mixed_precision = mixed_precision  # Boolean to enable automatic mixed precision
        self.criterion = criterion  # function used to calculate loss
        self.optimizer = optimizer  # algorithm used to minimize weights
        self.number_epochs = number_epochs  # maximum number of epochs for training the model

        # Metrics ------------------------------------------------------------------------------------------------------
        self._metrics = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        self.loss_freq = loss_freq  # (loss frequency) how often to calculate validation data loss
        self.acc_freq = acc_freq    # (accuracy frequency) how often to calculate accuracy
        self.metrics_to_csv = metrics_to_csv  # Boolean to enable saving metrics to a csv file
        self.ignore_train_acc = ignore_train_acc # Boolean to avoid calculation training accuracy
        self.model_name = model_name # String to change the model name when saving metric and checkpoints.

        if self.model_name == '':
            self.model_name = str(type(self.model).__name__)
        else:
            self.model_name = model_name

        # Save best model / Early stop ---------------------------------------------------------------------------------
        self.model_checkpoint = model_checkpoint    # boolean to enable model checkpoint
        self.target_metric = target_metric  # the metric tracked for checkpoint and early stop. Can be one of the
                                            # following: 'train_loss', 'val_loss', 'train_acc', 'val_acc'

        self.__previous_best: float = 0 # stores previous best value
        self.__best_so_far: float = 0   # stores best score
        self.__improved = True  # stores if the model improved this epoch

        self.early_stop = early_stop    # boolean to enable early stopping
        self.early_stop_patience = early_stop_patience  # How manny epochs without improvement to wait before
                                                        # terminating training

        # Other --------------------------------------------------------------------------------------------------------
        self.__storage_path = ''    # Path used for storing metrics and model checkpoints


    # Used to calculate loss of a given dataset using the model currently loaded in the GPU
    def __test_loss(self, data):
        total_loss = 0
        total_samples = 0
        with torch.no_grad():  # Disable gradient tracking
            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
        return total_loss / total_samples

    # # Used to test the prediction accuracy of a given dataset with the model currently loaded in the GPU
    # def __test_accuracy(self, data):
    #     accuracy = 0
    #     count = 0
    #     with torch.no_grad():   # Disable gradient tracking
    #         for inputs, labels in data:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             output = self.model(inputs)
    #             accuracy += torch.eq(torch.argmax(output, dim=1), labels).sum().item()
    #             count += len(labels)
    #     return (accuracy / count) * 100

    # BALANCED ACCURACY (use for dataset with class imbalance)
    def __test_accuracy(self, data):
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for inputs, labels in data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                y_pred = self.model(inputs)

                # Convert predictions to class indices
                pred_classes = torch.argmax(y_pred, dim=1)

                # Convert labels from one-hot encoding to class indices
                true_classes = torch.argmax(labels, dim=1)

                # Accumulate predictions and true labels
                all_y_true.extend(true_classes.cpu().numpy())
                all_y_pred.extend(pred_classes.cpu().numpy())

        # Calculate balanced accuracy score
        balanced_accuracy = balanced_accuracy_score(all_y_true, all_y_pred)

        return balanced_accuracy * 100

    # Checks if target metric is correctly setup
    def __does_metric_exist(self):
        if (self.target_metric in ['train_acc', 'val_acc']) and self.acc_freq <= 0:
            print("Error: No accuracy metrics. acc_freq must be > 0.")
            exit()
        elif self.target_metric == 'train_acc' and self.acc_freq > 0:
            print("Error: Training accuracy is disabled")
            exit()
        elif (self.target_metric == 'val_loss') and self.loss_freq <= 0:
            print("Error: No loss metrics. loss_freq must be > 0.")
            exit()
        elif self.target_metric not in ['train_acc', 'val_acc', 'train_loss', 'val_loss']:
            print("Error: Metric does not exist. Available options: ['train_acc', 'val_acc', 'train_loss', 'val_loss']")
            exit()

    # Checks if model improved
    def __did_model_improve(self, epoch):
        self.__does_metric_exist()

        metric_values = self._metrics[['epoch', self.target_metric]].dropna()

        # If the metric is not tested this epoch it will not consider an improvement
        if metric_values['epoch'].iat[-1] == epoch:
            # Set first value as best so far.
            # Used if else so that first model can be saved
            if metric_values['epoch'].iat[-1] == 0:
                self.__best_so_far = metric_values[self.target_metric].iloc[-1]
                self.__improved = False
            else:
                # For accuracy bigger values are better
                if self.target_metric in ['train_acc', 'val_acc']:
                    self.__improved = metric_values[self.target_metric].iloc[-1] > self.__best_so_far
                else:  # For loss smaller values are better
                    self.__improved = metric_values[self.target_metric].iloc[-1] < self.__best_so_far

            if self.__improved:
                self.__previous_best = self.__best_so_far
                self.__best_so_far = metric_values[self.target_metric].iloc[-1]

                print('\033[32m' + 'Metric "'+ self.target_metric + '" improved by: '
                      + f'{(self.__best_so_far - self.__previous_best):.7f}' + '\033[0m')
        else:
            self.__improved = False

    # Creates a model checkpoint when the tracked metric improves
    def __make_checkpoint(self, custom_name):
        # If the model improved it is saved
        if self.__improved:
            checkpoint_path = os.path.join(self.__storage_path, 'Trained_' + self.model_name)
            os.makedirs(checkpoint_path, exist_ok=True)

            if custom_name == '':
                file_name =  (self.model_name + "_" + str(self._metrics['epoch'].iloc[-1])
                             + "_" + str(round(self.__best_so_far, 6)).replace('.', '-') + "_"
                             + datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".pth")
            else:
                file_name = (custom_name + '_' + str(self._metrics['epoch'].iloc[-1]) + '_'+
                            (self.model_name + "_" + str(round(self.__best_so_far, 6)).replace('.', '-') + "_"
                             + datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".pth"))

            file_path = os.path.join(checkpoint_path, file_name)
            try:
                torch.save(self.model.state_dict(), file_path)
                print(f'Successfully saved model to {file_path}')
            except Exception as e:
                print(f'Error saving model to {file_path}: {e}')


    # start_training trains the model according to the settings defined by the user
    def start_training(self, custom_folder="", custom_name=""):
        print("\n" + "-"*5 + f" Training initialized ({self.device}) " + "-"*5)
        es_count = 0    # initiate early stop counter

        # Setup folder for saving metrics and model checkpoints --------------------------------------------------------
        if self.metrics_to_csv or self.model_checkpoint:
            current_path = os.path.dirname(os.path.abspath(__file__))
            if custom_folder:
                current_path = os.path.join(current_path, custom_folder)
            self.__storage_path = os.path.join(current_path, self.model_name)
            os.makedirs(self.__storage_path, exist_ok=True)

        # Creates a GradScaler once at the beginning of training.
        if self.mixed_precision:
            scaler = torch.GradScaler()

        # Train model --------------------------------------------------------------------------------------------------
        for epoch in range(self.number_epochs):
            total_loss = 0
            total_samples = 0

            # Training without mixed precision
            print(f"\nEpoch {epoch}: Start epoch", end="")
            start = time.time()
            if not self.mixed_precision:
                self.model.train()
                for inputs, labels in self.training_data:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = self.model(inputs)                 # Makes prediction with current weights
                    loss = self.criterion(output, labels)       # Calculate loss
                    self.optimizer.zero_grad()                  # Sets gradient to zero
                    loss.backward()                             # Back-propagation
                    self.optimizer.step()                       # Update weights

                    total_loss += loss.item() * len(labels)     # accumulate training loss of the epoch
                    total_samples += len(labels)                # accumulate number of datapoints

            # Training with mixed precision
            # Check docs: https://pytorch.org/docs/stable/notes/amp_examples.html
            else:
                self.model.train()
                for inputs, labels in self.training_data:
                    self.optimizer.zero_grad()
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Runs the forward pass with auto-casting.
                    with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                        output = self.model(inputs)
                        loss = self.criterion(output, labels)

                    # scales loss and do back pass
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                total_loss += loss.item() * len(labels)  # accumulate training loss of the epoch
                total_samples += len(labels)  # accumulate number of datapoints
            end = time.time()

            elapsed_time = round(end - start)
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(" -> End epoch ({}:{}:{})".format(hours, minutes, seconds))

            # Metrics --------------------------------------------------------------------------------------------------
            # switch model to evaluation mode
            self.model.eval()

            # Initialize metrics
            training_accuracy = pd.NA
            validation_accuracy = pd.NA
            training_loss = pd.NA
            validation_loss = pd.NA
            self._metrics.at[epoch, 'epoch'] = epoch

            # Training loss
            training_loss = total_loss / total_samples
            print(f"Training loss: {training_loss:.7f} ", end="")
            self._metrics.at[epoch, 'train_loss'] = training_loss    # save to dataframe

            # Validation loss
            if (self.loss_freq > 0) and (epoch % self.loss_freq == 0):
                validation_loss = self.__test_loss(self.validation_data)
                print(f" |  Validation loss: {validation_loss:.7f}")
            else:
                print()
            self._metrics.at[epoch, 'val_loss'] = validation_loss    # save to dataframe

            # Accuracy
            if (self.acc_freq > 0) and (epoch % self.acc_freq == 0):
                if not self.ignore_train_acc:
                    training_accuracy = self.__test_accuracy(self.training_data)
                validation_accuracy = self.__test_accuracy(self.validation_data)
                print(f"Training accuracy: {training_accuracy:.7f}  |  Validation accuracy: {validation_accuracy:.3f}")

            self._metrics.at[epoch, 'train_acc'] = training_accuracy
            self._metrics.at[epoch, 'val_acc'] = validation_accuracy

            # check if target variable improved this epoch
            self.__did_model_improve(epoch)

            # Model checkpoint
            if self.model_checkpoint:
                self.__make_checkpoint(custom_name)

            # Early stopping
            if self.early_stop and epoch > 0:
                if not self.__improved:
                    es_count += 1
                    print(f'\033[31mModel did not improve. Patience left: {self.early_stop_patience - es_count}\033[0m')
                else:
                    es_count = 0

                if self.early_stop_patience - es_count == 0:
                    break

        # Store metrics in csv file ------------------------------------------------------------------------------------
        if self.metrics_to_csv:
            if custom_name:
                file_name = "metrics_" + custom_name + ".csv"
            else:
                file_name = "metrics_" + self.model_name + "_" + datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".csv"

            file_path = os.path.join(self.__storage_path, file_name)
            try:
                self._metrics.to_csv(file_path, index=False)
                print(f'Successfully saved metrics to {file_path}')
            except Exception as e:
                print(f'Error saving metrics to {file_path}: {e}')


class Isic2019Dataset(torch.utils.data.Dataset):
    """
     Loads the isic 2019 challenge as a map style dataset. Requires the path for the csv file and the image folder.
     :param csv_file: Path for the csv ground truth.
     :param image_dir: Path for the folder containing the images.
     :param transform: [Optional] transform from "torchvision.transforms.Compose()". Defaults to NONE.
    """

    def __init__(self, image_dir, csv_file, transform=None):
        self.csv_table = (pd.read_csv(csv_file))
        self.root_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_table)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, (self.csv_table.at[index, 'image'] + '.jpg'))
        image = Image.open(img_path).convert("RGB")

        # Convert one-hot encoded labels to class indices
        # label = torch.tensor(self.csv_table.iloc[index, 1:].values.astype(float)).argmax().long()
        label = torch.tensor(self.csv_table.iloc[index, 1:].values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, label


class CropSquareCenter(torch.nn.Module):
    """
    Custom transform that crops the center of the image into a square shape. Used for preprocessing step of testing
    images.
    """

    def forward(self, img):
        # Initialize coordinates
        left = 0
        top = 0
        right = 0
        bottom = 0

        width, height = img.size

        if width == height: # Returns if image is square
            return img
        elif width > height: # Sets cropping coordinates if image is not square
            top = 0
            left = (width - height) // 2
            right = left + height
            bottom = height
        elif height > width:
            top = (height - width) // 2
            left = 0
            right = width
            bottom = top + width

        return img.crop((left, top, right, bottom))


def custom_model_loader(model):
    """
    This function receives a string that selects which model to load. The indicated model will be returned pretrained on
    the imagenet dataset and with the last layer adjusted so that the output will have the appropriate number of classes
    for the ISIC2019 dataset.
    The possible selected models include:
        - 'vgg19'
        - 'vgg16'
        - 'densenet121'
        - 'densenet201'
        - 'efficientnet_b5'
        - 'efficientnet_b6'
        - 'se-resnext101'
        - 'PNASNet5'

    :param model: String indicating which model to load.
    :return: Selected model (torch format).
    """
    # Number of classes in the dataset
    num_classes=8

    # Returns model according to selection
    match model:
        case 'vgg19':
            # Load pretrained vgg19
            vgg19_net = vgg19(weights='IMAGENET1K_V1')
            # Modify last layer
            vgg19_net.classifier[6] = torch.nn.Linear(in_features=vgg19_net.classifier[6].in_features,
                                                      out_features=num_classes)

            return vgg19_net

        case 'vgg16':
            # Load pretrained vgg16
            vgg16_net = vgg16(weights='IMAGENET1K_V1')
            # Modify last layer
            vgg16_net.classifier[6] = torch.nn.Linear(in_features=vgg16_net.classifier[6].in_features,
                                                      out_features=num_classes)

            return vgg16_net

        case 'densenet121':
            # Load pretrained model
            dn121_net = densenet121(weights='IMAGENET1K_V1')
            # Change last layer
            dn121_net.classifier = torch.nn.Linear(in_features=dn121_net.classifier.in_features,
                                                   out_features=num_classes)

            return dn121_net

        case 'densenet201':
            # Load pretrained model
            dn201_net = densenet201(weights='IMAGENET1K_V1')
            # Change last layer
            dn201_net.classifier = torch.nn.Linear(in_features=dn201_net.classifier.in_features,
                                                   out_features=num_classes)

            return dn201_net

        case 'efficientnet_b5':
            # Load pretrained model
            enb5_net = efficientnet_b5(weights='IMAGENET1K_V1')
            # Change last layer
            enb5_net.classifier[1] = torch.nn.Linear(in_features=enb5_net.classifier[1].in_features,
                                                     out_features=num_classes)

            return enb5_net

        case 'efficientnet_b6':
            # Load pretrained model
            enb6_net = efficientnet_b6(weights='IMAGENET1K_V1')
            # Change last layer
            enb6_net.classifier[1] = torch.nn.Linear(in_features=enb6_net.classifier[1].in_features,
                                                     out_features=num_classes)

            return enb6_net

        case 'se-resnext101':
            # Loading custom pretrained SE-resnext101 from repository
            resnext = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
            # Changing last layer
            resnext.fc = torch.nn.Linear(in_features=resnext.fc.in_features, out_features=num_classes)

            return resnext

        case 'PNASNet5':
            # Loading custom pretrained PNASNet5 from repository. And change output classes to 8
            return timm.create_model('pnasnet5large', pretrained=True, num_classes=num_classes)

        case _:
            print(f"ERROR: Model '{model}' is not supported.")
            exit()
