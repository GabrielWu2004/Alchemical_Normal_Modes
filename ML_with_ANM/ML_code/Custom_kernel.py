import numpy as np
import pandas as pd

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

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    cos_theta = np.dot(x, y) / (x_norm * y_norm)
    distance = np.linalg.norm(x - y)
    
    phi = np.exp(-gamma * (distance**2)/2 - epsilon * (x_norm - y_norm)**2 - beta * (1 - cos_theta**2))
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
    
    similarity_matrix = np.zeros((X_query.shape[0], X_ref.shape[0]))
    for i in range(X_query.shape[0]):
        for j in range(X_ref.shape[0]):
            similarity_matrix[i, j] = similarity_kernel(X_query[i], X_ref[j], params)

    return similarity_matrix
