import numpy as np
import pandas as pd
import numba
import math

def extended_gaussian_kernel(x, y, params):
    """
    Calculates the similarity between two vectors using an extended gaussian kernel.
    The kernel takes into account distance between vectors, norm difference, and angular difference
    Args:
        x (numpy.ndarray): Input vector x.
        y (numpy.ndarray): Input vector y.
        params (dict): Dictionary of hyperparameters:
        - gamma (float): Hyperparameter for the distance term.
        - epsilon (float): Hyperparameter for the norm difference term.
        - beta (float): Hyperparameter for the angular difference term.
    Returns:
        float: Similarity value between the input vectors.
    """
    gamma = params['gamma']
    epsilon = params['epsilon']
    beta = params['beta']
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    cos_theta = np.dot(x, y) / (x_norm * y_norm)
    distance = np.linalg.norm(x - y)
    phi = np.exp(-gamma*(distance**2)/2 - epsilon*(x_norm-y_norm)**2 - beta*(1-cos_theta**2))
    return phi

def create_similarity_matrix(X_ref, X_query, similarity_kernel, params):
    """
    Create a similarity matrix using a specified similarity kernel.
    Args:
        X_ref (numpy.ndarray): Reference training examples.
        X_quary (numpy.ndarray): Query input data to be compared with X_ref
        similarity_kernel (function): Function to calculate similarity between two vectors.
        params (dict): Dictionary of hyperparameters for the similarity kernel.
    Returns:
        numpy.ndarray: Similarity matrix
    """
    # X_ref = X_ref.astype(np.float64)
    # X_query = X_query.astype(np.float64)
    a, b = len(X_query), len(X_ref)
    similarity_matrix = np.zeros((a, b))
    for i in numba.prange(a):
        for j in numba.prange(b):
            similarity_matrix[i, j] = similarity_kernel(X_query[i], X_ref[j], params)
    return similarity_matrix


@numba.jit(nopython=True)
def extended_gaussian_kernel_nb(x, y, gamma, epsilon, beta):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    dot_product = np.dot(x, y)
    diff = x - y
    distance = np.linalg.norm(diff)
    cos_theta = dot_product / (x_norm*y_norm)
    phi = math.exp(-gamma*(distance**2)/2 - epsilon*(x_norm-y_norm)**2 - beta*(1-cos_theta))
    return phi

@numba.jit(parallel=True, nopython=True)
def create_similarity_matrix_nb(X_ref, X_query, similarity_kernel, gamma, epsilon, beta):
    a, b = len(X_query), len(X_ref)
    similarity_matrix = np.zeros((a, b))
    for i in numba.prange(a):
        for j in numba.prange(b):
            similarity_matrix[i, j] = similarity_kernel(X_query[i], X_ref[j], gamma, epsilon, beta)
    return similarity_matrix


@numba.jit(nopython=True)
def extended_gaussian_kernel_v2(x, y, gamma, epsilon, beta):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    dot_product = np.dot(x, y)
    diff = x - y
    distance = np.linalg.norm(diff)
    cos_theta = dot_product / (x_norm*y_norm)
    phi = math.exp(-gamma*(distance**4)/2 - epsilon*(x_norm-y_norm)**2 - beta*(1-cos_theta))
    return phi

@numba.jit(parallel=True, nopython=True)
def create_similarity_matrix_v2(X_ref, X_query, similarity_kernel, gamma, epsilon, beta):
    a, b = len(X_query), len(X_ref)
    similarity_matrix = np.zeros((a, b))
    for i in numba.prange(a):
        for j in numba.prange(b):
            similarity_matrix[i, j] = similarity_kernel(X_query[i], X_ref[j], gamma, epsilon, beta)
    return similarity_matrix



def vectorized_extended_gaussian_kernel(X, Y, params):
    """
    Calculates the similarity between two vectors using an extended gaussian kernel.
    The kernel takes into account distance between vectors, norm difference, and angular difference
    Args:
        X (numpy.ndarray): Input vector x.
        Y (numpy.ndarray): Input vector y.
        params (dict): Dictionary of hyperparameters:
        - gamma (float): Hyperparameter for the distance term.
        - epsilon (float): Hyperparameter for the norm difference term.
        - beta (float): Hyperparameter for the angular difference term.
    Returns:
        float: Similarity value between the input vectors.
    """
    gamma = params['gamma']
    epsilon = params['epsilon']
    beta = params['beta']
    x_norm = np.linalg.norm(X, axis=-1)
    y_norm = np.linalg.norm(Y, axis=-1)
    cos_theta = np.dot(X, Y.T) / np.outer(x_norm, y_norm)
    distance = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
    phi = np.exp(-gamma * (distance**2)/2 - epsilon * (np.subtract.outer(x_norm, y_norm))**2 - beta * (1 - cos_theta**2))
    return phi




def vectorized_similarity_matrix(X_ref, X_query, similarity_kernel, params):
    """
    Create a similarity matrix using a specified similarity kernel.
    Args:
        X_ref (numpy.ndarray): Reference training examples.
        X_query (numpy.ndarray): Query input data to be compared with X_ref
        similarity_kernel (function): Function to calculate similarity between two vectors.
        params (dict): Dictionary of hyperparameters for the similarity kernel.
    Returns:
        numpy.ndarray: Similarity matrix
    """
    similarity_matrix = similarity_kernel(X_ref, X_query, params)
    return similarity_matrix
