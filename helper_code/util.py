
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from QML_KernelRidge import KRR_local, KRR_global
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
        beta = params['beta']
        epsilon = params['epsilon']
        gamma = params['gamma']
        similarity_matrix = create_similarity_matrix_nb(X_train, X_train, similarity_kernel, beta, epsilon, gamma)
        model.fit(similarity_matrix, y_train)
        prediction_matrix = create_similarity_matrix_nb(X_train, X_val, similarity_kernel, beta, epsilon, gamma)
        y_pred = model.predict(prediction_matrix)
        error = mean_absolute_error(y_val, y_pred) 
        errors.append(error)
    
    average_error = np.mean(errors)
    std_dev_error = np.std(errors)/np.sqrt(num_trials)
    return average_error, std_dev_error



def evaluate_performance_vectorized_kernel(model, X, y, num_training_sample, num_trials, similarity_kernel, params):
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

        similarity_matrix = vectorized_similarity_matrix(X_train, X_train, similarity_kernel, params)
        model.fit(similarity_matrix, y_train)
        prediction_matrix = vectorized_similarity_matrix(X_train, X_val, similarity_kernel, params)
        y_pred = model.predict(prediction_matrix.T)
        error = mean_absolute_error(y_val, y_pred) 
        errors.append(error)
    
    average_error = np.mean(errors)
    std_dev_error = np.std(errors)/np.sqrt(num_trials)
    return average_error, std_dev_error



def evaluate_performance_global(params, X, y, num_training_sample, num_trials):

    errors = []
    test_size = 1.0 - num_training_sample/X.shape[0]

    for i in range(num_trials):
        train_indices, test_indices = train_test_split(range(X.shape[0]), test_size=test_size, shuffle=True, random_state=i)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        preds = KRR_global(X_train, y_train, X_test, best_params=params, kernel='Gaussian')
        error = mean_absolute_error(preds.reshape(-1, 1), y_test)
        errors.append(error)
    
    average_error = np.mean(errors)
    std_dev_error = np.std(errors)/np.sqrt(num_trials)
    return average_error, std_dev_error



def evaluate_performance_local(params, X, y, Q, num_training_sample, num_trials):

    errors = []
    test_size = 1.0 - num_training_sample/X.shape[0]

    for i in range(num_trials):
        train_indices, test_indices = train_test_split(range(X.shape[0]), test_size=test_size, shuffle=True, random_state=i)
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        Q_train, Q_test = Q[train_indices], Q[test_indices]
        preds = KRR_local(X_train, Q_train, y_train, X_test, Q_test, best_params=params, kernel='Gaussian')
        error = mean_absolute_error(preds.reshape(-1, 1), y_test)
        errors.append(error)
    
    average_error = np.mean(errors)
    std_dev_error = np.std(errors)/np.sqrt(num_trials)
    return average_error, std_dev_error



def generate_dx_arrays(num_dopant, num_mutant):
    mutations = np.zeros((num_mutant, 24), dtype=int)
    
    for i in range(num_mutant):
        dopant_indices = np.random.choice(24, num_dopant, replace=False)
        anti_dopant_indices = np.random.choice(
            np.setdiff1d(np.arange(24), dopant_indices), 
            num_dopant, 
            replace=False)
        
        mutations[i, dopant_indices] = 1
        mutations[i, anti_dopant_indices] = -1
    
    return mutations



def convert_to_charge_array(dx_array):
    charge_array = dx_array + 6
    h_elements = np.full((charge_array.shape[0], 12), 1)
    concatenated_array = np.concatenate((charge_array, h_elements), axis=1)
    return concatenated_array



def convert_to_string_array(dx_array):
    string_array = np.where(dx_array == 1, 'N', np.where(dx_array == -1, 'B', 'C'))
    h_elements = np.full((string_array.shape[0], 12), 'H')
    concatenated_array = np.concatenate((string_array, h_elements), axis=1)
    return concatenated_array



def compress_string(lst):
    compressed = ""
    count = 1
    for i in range(1, len(lst)):
        if lst[i] == lst[i-1]:
            count += 1
        else:
            compressed += lst[i-1]
            if count > 1:
                compressed += str(count)
            count = 1
    # Append the last letter and its count
    compressed += lst[-1]
    if count > 1:
        compressed += str(count)
    return compressed



def compress_str_arrays(str_array):
    return [compress_string(str_arr.tolist()) for str_arr in str_array]



def generate_mutant_str(num_dopant, num_mutant):
    doped_dx_arrays = generate_dx_arrays(num_dopant, num_mutant)
    doped_str_arrays = convert_to_string_array(doped_dx_arrays)
    compressed_str_array = compress_str_arrays(doped_str_arrays)
    return compressed_str_array



def charge_arr_to_str(charge_arr):
    mapping = {1: 'H', 5: 'B', 6: 'C', 7: 'N'}
    string_arr = [mapping.get(integer, '') for integer in charge_arr]
    compressed_str = compress_string(string_arr)
    return compressed_str