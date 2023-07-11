
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from custom_kernel import *

def evaluate_performance(model, X, y, num_training_sample, num_trials):

    """ 
    Given the number of training samples used, 
    calculate the average and standard deviation of MSE across a certain number of trials.
    For each trial, a specified number of training examples is used to train the model, 
    which is then evaluated on the rest of the data set.

    Args:
        X (ndarray): training data; size (N, m) where N is the number of training examples and m is the number of features
        y (ndarray): target data; size (N, 1)
        num_training_sample (int): the number of samples used for training
        num_trials: the number of trials 
    
    Returns:
        average_error: the average MSE across all trials
        std_dev_error: standard deviation of the error across all trials
    """

    errors = []
    test_size = 1.0 - num_training_sample/X.shape[0]

    for i in range(num_trials):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        error = mean_absolute_error(y_val, y_pred) 
        errors.append(error)
    
    average_error = np.mean(errors)
    std_dev_error = np.std(errors)/np.sqrt(num_trials)
    return average_error, std_dev_error


def evaluate_performance_custom_kernel(model, X, y, num_training_sample, num_trials, similarity_kernel, params):
    """ 
    Given the number of training samples used, 
    calculate the average and standard deviation of MSE across a certain number of trials.
    For each trial, a specified number of training examples is used to train the model, 
    which is then evaluated on the rest of the data set.

    Args:
        X (ndarray): training data; size (N, m) where N is the number of training examples and m is the number of features
        y (ndarray): target data; size (N, 1)
        num_training_sample (int): the number of samples used for training
        num_trials: the number of trials 
        similarity_kernel: the custom kernel used
    
    Returns:
        average_error: the average MSE across all trials
        std_dev_error: standard deviation of the error across all trials
    """

    errors = []
    test_size = 1.0 - num_training_sample/X.shape[0]

    for i in range(num_trials):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42+i)
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_val = X_val.to_numpy()
        y_val = y_val.to_numpy()

        similarity_matrix = create_similarity_matrix(X_train, X_train, similarity_kernel, params)
        model.fit(similarity_matrix, y_train)
        prediction_matrix = create_similarity_matrix(X_train, X_val, similarity_kernel, params)
        y_pred = model.predict(prediction_matrix)
        error = mean_absolute_error(y_val, y_pred) 
        errors.append(error)
    
    average_error = np.mean(errors)
    std_dev_error = np.std(errors)/np.sqrt(num_trials)
    return average_error, std_dev_error