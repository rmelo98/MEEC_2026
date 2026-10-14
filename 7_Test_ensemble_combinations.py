import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
import os
from itertools import combinations


def voting_ensemble(all_values):
    """Receives a list with the raw values of inference returned by the classification models. The raw values
     must have the same dimensions.

        :param all_values: List of lists containing the raw values of the cnn. Example: [[0.2, 0.8], [0.9, 0.15]]
        :return: Returns an integer indicating the position of the maximum argument.
    """

    if not all_values:
        raise ValueError('Error: all_values cannot be empty')

    length_first = len(all_values[0])
    voting_results = np.zeros(length_first)
    accumulate_raw_value = np.zeros(length_first)
    for raw_value in all_values:
        # Check if raw values have the same size
        if len(raw_value) != length_first:
            raise ValueError('Error: Raw values must contain the same length')

        # accumulates raw value to untie in case voting is the same
        raw_value = np.array(raw_value)
        accumulate_raw_value = accumulate_raw_value + raw_value

        # increments voting
        voting_results[np.argmax(raw_value)] += 1

    # Check if there is a tie
    binary_max_value = (voting_results == voting_results[np.argmax(voting_results)])
    if np.sum(binary_max_value) > 1:
        # THERE IS A TIE: use accumulated value to untie
        untie_values = (accumulate_raw_value/len(all_values)) * binary_max_value
        return np.argmax(untie_values)
    else:
        # NO TIE: returns argmax
        return np.argmax(voting_results)


def average_ensemble(all_values):
    """
    Receives a list of probability vectors from multiple models and returns the class with the highest average
    probability.

    :param all_values: List of lists containing the raw values of the cnn. Example: [[0.2, 0.8], [0.9, 0.15]]
    :return: Returns an integer indicating the position of the maximum argument.
    """

    if not all_values:
        raise ValueError("Error: all_values cannot be empty")

    length_first = len(all_values[0])

    for raw_value in all_values:
        if len(raw_value) != length_first:
            raise ValueError("Error: Raw values must contain the same length")

    # Average probabilities across all models
    average = np.mean(np.array(all_values), axis=0)

    return np.argmax(average)

TestSplit = ['val', 'test']
StoragePath = os.path.join(os.getcwd(), 'Test_ensemble_voting')
os.makedirs(StoragePath, exist_ok=True)

# Setup data path and test to perform
all_models = ['densenet121', 'densenet201', 'efficientnet_b5', 'efficientnet_b6', 'vgg19', 'se-resnext101', 'PNASNet5']

tests_list = []
for NumberOfModels in range(2, 8):
    for current_combinations in combinations(all_models, NumberOfModels):
        tests_list.append(current_combinations)

for CurrentSplit in TestSplit:
    Inference_data_path = os.path.join('/mnt/D/Projeto/Ensemble_inference', CurrentSplit)
    results_BA = pd.DataFrame({'Models':tests_list,
                            'Voting':[None]*len(tests_list),
                            'Mean':[None]*len(tests_list)})

    results_F1 = pd.DataFrame({'Models':tests_list,
                            'Voting':[None]*len(tests_list),
                            'Mean':[None]*len(tests_list)})

    # Begin testing
    for test_type in ['Voting', 'Mean']:
        for current_test, test in enumerate(tests_list):
            print(f'Performing test ({test_type}): {test}')

            # Load Test Data
            for i, model in enumerate(test):
                data_name = model + '.json'

                if i == 0:
                    data = pd.read_json(os.path.join(Inference_data_path, data_name))
                    data = data.rename(columns={'Pred': model})
                else:
                    data = pd.concat([data, pd.read_json(os.path.join(Inference_data_path, data_name))['Pred']], axis=1)
                    data = data.rename(columns={'Pred': model})

            ground_truth = pd.DataFrame(data['True'])
            ensemble_result = pd.DataFrame(np.nan, index=ground_truth.index, columns=['Pred'])
            ensemble_data = data.drop(['True'], axis=1)

            # Execute voting
            for line in range(ensemble_data.shape[0]):
                current_line = ensemble_data.loc[line, :].values.tolist()
                if test_type == 'Voting':
                    ensemble_result.at[line, 'Pred'] = voting_ensemble(current_line)
                else:
                    ensemble_result.at[line, 'Pred'] = average_ensemble(current_line)

            balanced_accuracy = balanced_accuracy_score(y_true=ground_truth, y_pred=ensemble_result)
            F1Score = f1_score(y_true=ground_truth, y_pred=ensemble_result, average='weighted')
            results_BA.at[current_test, test_type] = balanced_accuracy
            results_F1.at[current_test, test_type] = F1Score

            # Save results
            models_name = "_".join(test)
            test_name = test_type + '_' + models_name + '.csv'
            StorageDataFrame = pd.DataFrame(pd.concat([ground_truth, ensemble_result], axis=1))
            StorageDataFrame.to_csv(os.path.join(StoragePath, test_name), index=False)


    results_BA.to_csv('Ensemble_' + CurrentSplit + '_BA.csv', index=False)
    results_F1.to_csv('Ensemble_' + CurrentSplit + '_F1.csv', index=False)
