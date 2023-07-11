import sys
sys.path.append('../APDFT')

import numpy as np
import pickle
from pyscf import gto, scf, dft, cc
import numpy as np
import pandas as pd
import pyscf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import basis_set_exchange as bse
from APDFT.FcMole import *
import os
import ast
from copy import deepcopy
from IPython.display import display
from APDFT.AP_class import APDFT_perturbator as AP



def create_coord_array(molecule, num_atoms):
    """ 
    Create a numpy array of coordinates based on the coordinate string

    Args:
        molecule (str): the string representation of molecule, in xyz file format
        num_atoms (int): the number of atoms in the molecule
    Returns:
        ndarray (num_atoms, 3): the coordinates of atoms in the molecule, where each row represents on atom
    """
    
    lines = molecule.strip().split("\n")
    coord_array = np.zeros((num_atoms, 3))
    for i, line in enumerate(lines):  
        parts = line.split()
        coordinates = [float(coord) for coord in parts[1:]] # Skip the first line (column headers)
        coord_array[i] = coordinates
    return coord_array



def get_DFT(molecule, basis):
    """ 
    Generate the DFT object, total energy, and electronic electronic energy

    Args: 
        molecule (str): the string representation of molecule, in xyz file format
        basis (dict): a dictionary representing the basis functions used to approximate wave function for each element type
    Returns:
        pyscf.dft.rks.RKS: the DFT object for the molecule
        float: the total energy of the molecule
        float: the electronic energy of the molecule
    """
    
    mol = gto.M(atom=molecule, basis=basis, unit='Angstrom')
    # run DFT calculation
    mol_DFT = scf.RKS(mol)
    mol_DFT.xc = "PBE0" # specify the exchange-correlation functional used for DFT
    mol_DFT.kernel(verbose=False) # run self-consistent field calculation
    mol_total_energy = mol_DFT.energy_tot()
    mol_electronic_energy = mol_DFT.energy_elec()
    return mol_DFT, mol_total_energy, mol_electronic_energy



def load_data(molecule, basis, dest_csv_path, raw_tot_data_path, raw_elec_data_path):
    """ 
    Load the energy of the molecule and other useful information into a pandas dataframe.
    The dataframe will be stored as an csv file at dest_csv_path.
    The raw data will be extracted from raw_data_path.

    Args:
        molecule (str): the string representation of molecule, in xyz file format
        basis (dict): a dictionary representing the basis functions used to approximate wave function for each element type
        dest_csv_path (str): the path to the destination at which the final csv file is saved
        raw_tot_data_path (str): the path where the raw total energy data is stored
        raw_elec_data_path (str): the path where the raw electronic energy data is stored
    Returns:
        DataFrame: the complete molecule energy data
    """
    global mol_DFT
    
    if os.path.isfile(dest_csv_path):
        molecule_energy_data = pd.read_csv(dest_csv_path, index_col=0, header=0)
        molecule_energy_data['charges'] = molecule_energy_data['charges'].apply((lambda x: ast.literal_eval(x)))
        molecule_energy_data['elements'] = molecule_energy_data['elements'].apply((lambda x: ast.literal_eval(x)))
        print("Load data complete!")
    else:
        # Load the raw dataset
        print("No exisiting dataset. Start running calculations.")
        print("Unpacking raw data ...", end=" ")
        total_energy_data = np.load(raw_tot_data_path, allow_pickle=True)
        electronic_energy_data = np.load(raw_elec_data_path, allow_pickle=True)

        # Unpack the data into numpy arrays
        charges, total_energy, electronic_energy = total_energy_data['charges'], total_energy_data['energies'], electronic_energy_data['energies']
        
        # Creating pandas dataframe for the data
        columns = ['charges', 'total energy', 'electronic energy']
        molecule_energy_data = pd.DataFrame(columns=columns)
        molecule_energy_data['charges'] = charges.tolist()
        molecule_energy_data['total energy'] = total_energy.tolist()
        molecule_energy_data['electronic energy'] = electronic_energy.tolist()
        print("complete!")
        
        # Calculate delta total energy and electronic energy
        print("Running DFT Calculation ...", end= " ") 
        mol = gto.M(atom=molecule, basis=basis, unit='Angstrom')
        # run DFT calculation
        mol_DFT = scf.RKS(mol)
        mol_DFT.xc = "PBE0" # specify the exchange-correlation functional used for DFT
        mol_DFT.kernel(verbose=False) # run self-consistent field calculation
        mol_total_energy = mol_DFT.energy_tot()
        mol_electronic_energy = mol_DFT.energy_elec()
        print("Total energy:", mol_total_energy)
        print("Electronic energy (electronic energy, nuclear repulsion energy):", mol_electronic_energy) 

        molecule_energy_data['delta total energy'] = molecule_energy_data['total energy'].apply(lambda x: mol_total_energy - x)
        molecule_energy_data['delta electronic energy'] = molecule_energy_data['electronic energy'].apply(lambda x: mol_electronic_energy[0] - x)
        molecule_energy_data.to_csv(dest_csv_path, index=True)
        print("Calculation complete!")

    return molecule_energy_data



def get_hessian():
    if os.path.isfile('CCS_basis/hessian_PBE0.txt'):
        H = np.loadtxt('CCS_basis/hessian_PBE0.txt')
    else:
        C_idxs = [0, 1, 2, 3, 4, 5]
        mol_ap = AP(mol_DFT, sites=C_idxs)
        H = mol_ap.build_hessian()
        np.savetxt('CCS_basis/hessian_PBE0.txt', H)
    return H



def get_inv_dist_M(coord):
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i != j:
                r_ij = np.linalg.norm(coord[i] - coord[j])  # Calculate Euclidean distance between i-th and j-th rows of coord
                M[i, j] = 1/(r_ij + 1e-8)
            else:
                M[i, j] = 0.5 * (2.4)**2 * 6**0.4
            
    return M


def lexi_transformation(arr):
    """ 
    This function maps cyclic arrays that are rotational or reflectional identical onto the same vector.
    The function iterate through all rotaional and reflectional variants of the array,
    and select the lexicographically minimum array as the final representation.

    Args:
        arr (ndarray): a numpy array to be transformed
    Return:
        ndarray: transformed array
    """
    
    # Create all possible rotations of the cycle
    shift = np.arange(len(arr))
    shifted_arrays = []
    for s in shift:
        shifted = np.roll(arr, shift=s)
        shifted_arrays.append(shifted)
    rotations = np.vstack(shifted_arrays)

    # Create the corresponding reverse traversal patterns for each rotation
    reverse_traversals = np.flip(rotations, axis=1)

    # Combine rotations and reverse traversals
    all_patterns = np.vstack((rotations, reverse_traversals))

    # Find the lexicographically smallest representation (left to right)
    sorted_indices = np.lexsort(all_patterns.T[::-1])
    min_pattern = all_patterns[sorted_indices[0]]
    return min_pattern



def lexi_transformation_2d(arr):
    transformed_arr = []
    for row in arr:
        transformed_row = lexi_transformation(row)
        transformed_arr.append(transformed_row)
    return np.array(transformed_arr)



def sort_by_norm(c, coord):
    """
    Sorts the elements of a 6-element array, c, based on pseudo Coulombic matrix

    Args:
        c (list or array-like): The input array with 6 elements.
        coord (numpy.ndarray): The XYZ coordinates of atoms benzene ring.
    Returns:
        numpy.ndarray: A sorted numpy array containing the elements of c, arranged based on the magnitude of the norm array.
    """

    # Convert c to a numpy array
    c_array = np.array(c)

    # Calculate the pseudo Coulombic matrix
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i != j:
                z_ij = np.linalg.norm(coord[i] - coord[j])  # Calculate Euclidean distance between i-th and j-th rows of coord
            else:
                z_ij = 1 # for diagonal elements, set z_ij to 1
            M[i, j] = c_array[i] * c_array[j] / (z_ij + 1e-8)

    # Compute the norm of each column of M
    norms = np.linalg.norm(M, axis=0)

    # Sort c_array based on the magnitude of elements in norms (in descending order)
    sorted_indices = np.argsort(-norms)
    sorted_c_array = c_array[sorted_indices]
    return sorted_c_array



def generate_coef_with_specific_basis(molecule_energy_data, basis_matrix, coord, ref_charge):
    
    data_with_specific_basis = molecule_energy_data.copy(deep=True)

    # Compute the dx value as the different between target charge and reference charge
    ref_charge_array = np.tile(ref_charge, (17, 1))
    target_charge_array = np.array(data_with_specific_basis['charges'].tolist())
    dx_array = (target_charge_array - ref_charge_array)[:, :6] # only take the dx for C atoms

    # try different ways to sort the dx vector
    sorted_dx_array = np.sort(dx_array, axis=1) # sorted dx elements
    lexi_dx_array = lexi_transformation_2d(dx_array) # use lexi mapping

    # Compute the c array, which represents the ANM coordinates
    c_array = (basis_matrix @ dx_array.T).T
    c_inv_array = (np.linalg.inv(basis_matrix) @ dx_array.T).T
    sorted_c_array = (basis_matrix @ sorted_dx_array.T).T
    lexi_c_array = (basis_matrix @ lexi_dx_array.T).T
    lexi_c_inv_array = (np.linalg.inv(basis_matrix) @ lexi_dx_array.T).T

    # Append the data onto the dataframe
    data_with_specific_basis['dx'] = dx_array.tolist()
    data_with_specific_basis['sorted_dx'] = sorted_dx_array.tolist()
    data_with_specific_basis['lexi_dx'] = lexi_dx_array.tolist()

    data_with_specific_basis['c'] = c_array.tolist()
    data_with_specific_basis['c_inv'] = c_inv_array.tolist()
    data_with_specific_basis['sorted_c'] = sorted_c_array.tolist()
    data_with_specific_basis['lexi_c'] = lexi_c_array.tolist()
    data_with_specific_basis['lexi_c_inv'] = lexi_c_inv_array.tolist()
    data_with_specific_basis['coulomb_sort_c'] = data_with_specific_basis['c'].apply(lambda c: sort_by_norm(c, coord).tolist())
    data_with_specific_basis['num_dope'] = data_with_specific_basis['dx'].apply(lambda my_list: sum(1 for elem in my_list if elem != 0))

    dx_columns = ['dx', 'sorted_dx', 'lexi_dx', 'num_dope']
    c_columns = ['c', 'c_inv', 'sorted_c', 'lexi_c', 'lexi_c_inv', 'coulomb_sort_c']
    return data_with_specific_basis, dx_columns, c_columns



def generate_input_training_data(molecule_data):
    columns = [f"coord{i}" for i in range(6)]
    X = pd.DataFrame(columns=columns)
    X_inv = pd.DataFrame(columns=columns)
    X_sorted = pd.DataFrame(columns=columns)
    X_lexi = pd.DataFrame(columns=columns)
    X_lexi_inv = pd.DataFrame(columns=columns)
    X_coulomb = pd.DataFrame(columns=columns)
    X_square_eig = pd.DataFrame(columns=columns)
    X_inv_square_eig = pd.DataFrame(columns=columns)
    X_lexi_square_eig = pd.DataFrame(columns=columns)
    X_lexi_inv_square_eig = pd.DataFrame(columns=columns)
    
    num_dope = molecule_data['num_dope']

    for i in range(6):
        X[f"coord{i}"] = molecule_data['c'].apply(lambda x: x[i]*10)
        X_inv[f"coord{i}"] = molecule_data['c_inv'].apply(lambda x: x[i]*10)
        X_sorted[f"coord{i}"] = molecule_data['sorted_c'].apply(lambda x: x[i]*10)
        X_lexi[f"coord{i}"] = molecule_data['lexi_c'].apply(lambda x: x[i]*10)
        X_lexi_inv[f"coord{i}"] = molecule_data['lexi_c_inv'].apply(lambda x: x[i]*10)
        X_coulomb[f"coord{i}"] = molecule_data['coulomb_sort_c'].apply(lambda x: x[i]*10)
        X_square_eig[f"coord{i}"] = molecule_data['c_square_eig'].apply(lambda x: x[i]*10)
        X_inv_square_eig[f"coord{i}"] = molecule_data['c_inv_square_eig'].apply(lambda x: x[i]*10)
        X_lexi_square_eig[f"coord{i}"] = molecule_data['lexi_c_square_eig'].apply(lambda x: x[i]*10)
        X_lexi_inv_square_eig[f"coord{i}"] = molecule_data['lexi_c_inv_square_eig'].apply(lambda x: x[i]*10)

    X_nd = pd.concat([X, num_dope.rename('num_dope')], axis=1)
    X_lexi_nd = pd.concat([X_lexi, num_dope.rename('num_dope')], axis=1)
    datasets = [X, X_inv, X_sorted, X_lexi, X_lexi_inv, X_nd, X_lexi_nd, X_coulomb, X_square_eig, X_inv_square_eig, X_lexi_square_eig, X_lexi_inv_square_eig]
    return datasets



def generate_target_training_data(molecule_data):
    y_energy = molecule_data['total energy']
    y_elec = molecule_data['electronic energy']
    y_delta_energy = molecule_data['delta total energy']
    y_delta_elec = molecule_data['delta electronic energy']
    return y_energy, y_elec, y_delta_energy, y_delta_elec



def export_to_csv_custom(datasets, dataset_names, prefix, dest_folder):
    for dataset, dataset_name in zip(datasets, dataset_names):
        csv_filename = f"{dest_folder}/[Benz] {prefix}_{dataset_name}.csv"  # Create the CSV file name
        dataset.to_csv(csv_filename, index=False)  # Save the dataframe as CSV



def compute_lambda_c_square(c_arr, eig_val_arr):
    """ 
    square each coefficient and multiply it by the ANM eigenvalue

    Args:
        c_arr (list): a list of the ANM coefficients
        eig_val_arr (list): a list of the ANM eigenvalues
    Returns:
        list: the transformed coefficient
    """
    transformed_c = [eig_val * coef**2 for eig_val, coef in zip(eig_val_arr, c_arr)]
    return transformed_c



def compute_lambda_c(c_arr, eig_val_arr):
    transformed_c = [eig_val * coef for eig_val, coef in zip(eig_val_arr, c_arr)]
    return transformed_c