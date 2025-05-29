# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:50:23 2025

@author: Bálint
"""

import numpy as np
import pandas as pd
#%% definitions
#%%% trigonometry
def dist_hyplane(Th,x):
    '''
    Parameters
    ----------
    Th : array (d+1)
        Contains [th,th0] as the parameters of a hyperplane.
        In case of larger dimensions th is an array (d).
    x : array (d)
        Contains coordinates of a point.

    Returns:
    -------
    Distance of point x from hyperplane characterized by Th.
    '''
    th=Th[0]
    th0=Th[1]
    dist=(np.dot(th,x)+th0)/np.linalg.norm(th)
    return dist

def dist_point(p1,p2):
    '''
    Parameters
    ----------
    p1 : array (d)
        Coordinates of p1 point.
    p2 : array (d)
        Coordinates of p2 point.

    Returns:
    -------
    Distance of the two points.
    '''
    dist=np.linalg.norm(p2-p1)
    return dist

def point_margin(Th,x,y):
    '''
    Parameters
    ----------
    Th : array (d+1)
        Contains [th,th0] as the parameters of a hyperplane.
        In case of larger dimensions th is an array (d).
    x : array (d)
        Contains coordinates of a point.
    y : int
        The label of the point.

    Returns:
    -------
    The margin of point th in reference to th,th0 hyperplane.
    '''
    th=Th[0]
    th0=Th[1]
    marg=y*(np.dot(th,x)+th0)/np.linalg.norm(th)
    return marg
#%%% loss calculation    
def point_hinge_loss(Th,x,y,margin_ref):
    '''
    Parameters
    ----------
    Th : array (d+1)
        Contains [th,th0] as the parameters of a hyperplane.
        In case of larger dimensions th is an array (d).
    x : array (d)
        Contains coordinates of a point.
    y : int
        The label of the point.
    margin_ref : int,float
        The reference margin for the hinge loss.

    Returns
    -------
    The hinge loss of a point in reference to th,th0 hyperplane.
    '''
    if point_margin(Th,x,y)<margin_ref:
        Lh=1-point_margin(Th,x,y)/margin_ref
    else:
        Lh=0  
    return Lh
#%%% vector manipulation
def cv(value_list):
    '''
    Parameters
    ----------
    value_list : list (l)
        List to turn into column vector.

    Returns
    -------
    Column vector (lth1) with the elements of the list.
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Parameters
    ----------
    value_list : list (l)
        List to turn into row vector.

    Returns
    -------
    Row vector (1thl) with the elements of the list.
    '''
    return np.array([value_list])
#%%% step functions
def step_const(eta,x):
    '''
    Parameters
    ----------
    eta : scalar
        The starting step size.
    x : int
        The iteration indeth.

    Returns
    -------
    eta : scalar
        Returns a constant eta value, independent of the step indeth.
    '''
    return eta

def step_dec(eta,x):
    '''
    Parameters
    ----------
    eta : scalar
        The starting step size.
    x : int
        The iteration indeth.

    Returns
    -------
    step : scalar
        Returns an eta value that decreases with the step indeth.
    '''
    step=eta/(x+1)
    return step

def step_svm_min(eta,x):
    step=2/(x+1)**(1/2)
    return step

#%%% data manipulation
def separable_medium():
    x = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]]).T
    y = np.array([[1, -1, 1, -1]]).T
    return x, y

def prep_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    return df

def split_data(x, y, z):
    """
    Split data into z folds.
    
    Parameters:
    x : numpy array, shape (n, d)
        Feature matrix.
    y : numpy array, shape (n, 1)
        Target vector.
    z : int
        Number of folds.
        
    Returns:
    x_folds : list of numpy arrays
        List of z feature arrays
    y_folds : list of numpy arrays
        List of z target arrays
    """
    n, d = x.shape
    if n != y.shape[0]:
        raise ValueError("x and y must have same number of samples")
    if z < 2 or z > n:
        raise ValueError("z must be between 2 and number of samples")
    # Shuffle indices
    indices = np.random.permutation(n)
    
    # Calculate fold sizes
    fold_sizes = np.full(z, n // z, dtype=int)
    fold_sizes[:n % z] += 1  # Distribute remainder
    
    # Initialize lists to store folds
    x_folds = []
    y_folds = []
    
    # Split data into z folds
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_indices = indices[start:stop]
        x_folds.append(x[fold_indices])
        y_folds.append(y[fold_indices])
        current = stop
    
    return x_folds, y_folds

#%%% packing, unpacking
def homogen_th_unpack(th_hom):
    th = th_hom[:-1]
    th0 = th_hom[-1]
    return th, th0

def homogen_ths_unpack(ths_hom):
    ths = ths_hom[:,:-1]
    th0s = ths_hom[:,-1]
    return ths, th0s

def homogenize_th(th, th0):
    th_hom = np.append(th, th0)
    return th_hom

def homogenize_x(x):
    n, d = x.shape
    identity = np.ones((n, 1))
    x_hom = np.hstack((x, identity))
    return x_hom
 #%%% feature transformations
def standardize(feature, mu = None, sig = None):
    '''
    Standardizes a feature with it's own parameters, or with the ones provided.'
    
    Parameters
    ----------
    feature : array (n, 1)
        The feature array to be standardized.
    mu : scalar
        The mean of the standardization to be applied.
    sig : scalar
        The standard deviation of the standardization to be applied.

    Returns
    -------
    standardized_feature (always): array (n, 1)
        The standardized feature array.
    mu (if not input): scalar
        The mean of the feature elements.
    sig (if not input): scalar
        The standard deviation of the feature elements.
    '''
    if mu == None and sig == None:
        mu = np.mean(feature)  
        sig = np.std(feature)
        standardized_feature = (feature - mu) / sig
        return standardized_feature, mu, sig
    else:
        standardized_feature = (feature - mu) / sig
        return standardized_feature
 
def standardize_featureset(features, feature_config, mus = None, sigs = None):
    '''
    Standardizes a featureset with it's own parameters, or with the ones provided.'

    Parameters
    ----------
    features : array (n, d)
        The featureset array to be standardized.
    feature_config : dict
        Dictionary holding the names of the features, indices, and which features to standardize.
    mus : list (s, 1)
        The means of the featureset's elements.
    sigs : list (s, 1)
        The standard deviations of the featureset's elements.

    Returns
    -------
    features (always): array (n, d)
        The standardized featureset array.
    mus (if not input): list (s, 1)
        DESCRIPTION.
    sigs (if not input): list (s, 1)
        DESCRIPTION.
    '''
    if mus == None and sigs == None:
        mus = [0] * len(feature_config["standardize"])
        sigs = [0] * len(feature_config["standardize"])
        if feature_config["standardize"] != None:
            n, d = features.shape
            paramindex = 0
            for j in range(d):
                if feature_config["map"][j] in feature_config["standardize"]:
                    features[:,j], mus[paramindex], sigs[paramindex] = standardize(features[:, j])
                    paramindex += 1
        return features, mus, sigs
    else:
        if feature_config["standardize"] != None:
            n, d = features.shape
            paramindex = 0
            for j in range(d):
                if feature_config["map"][j] in feature_config["standardize"]:
                    features[:,j] = standardize(features[:, j], mus[paramindex], sigs[paramindex])
                    paramindex += 1
        return features

def inv_standardize(feature, mu, sig):
    inverse_standardized_feature = feature * sig + mu
    return inverse_standardized_feature

def poly_basis(feature, order):
    '''
    Applies polinomial basis to the feature.

    Parameters
    ----------
    feature : array (n ,1)
        The feature array to be standardized.
    order : scalar
        Order of the polinomial basis.

    Returns
    -------
    output : array (n, order)
        Array of the polinomial features.
    '''
    n = feature.shape[0]
    output = np.zeros((n, order))
    for i in range(order):
        output[:, i:i+1] = np.power(feature, i + 1)
    return output

def poly_basis_featureset(features, order, feature_config):
    '''
    Applies polinomial basis to the featureset.

    Parameters
    ----------
    features : array (n, d)
        The featureset array to be standardized.
    order : scalar
        Order of the polinomial basis.
    feature_config : dict
        Dictionary holding the names of the features, indices, and which features to bring to polinomial basis.

    Returns
    -------
    features : array (n, d + order - 1)
        Featureset with the polinomial features added.
    '''
    if feature_config["polynomial"] != None:
        # calculating the polynomials
        n, d = features.shape
        expansion = [0 for _ in feature_config["polynomial"]]
        expansion_index = [0 for _ in feature_config["polynomial"]]
        j = 0
        for i in range(d):
            if feature_config["map"][i] in feature_config["polynomial"]:
                expansion[j] = poly_basis(features[:,i].reshape(-1,1), order)
                expansion_index[j] = i
                j += 1
        expansion = np.asarray(expansion)
        
        # expanding the feature space
        mod = 0
        idx = 0
        for k in expansion_index:
            features = np.insert(features, [k + 1 + mod], expansion[idx, :, 1:], axis = 1)
            mod += order-1
            idx += 1
    return features

def one_hot(cv):
    '''
    One-hot encoding of a feature.

    Parameters
    ----------
    cv : array (n, 1)
        Feature to encode.

    Returns
    -------
    one_hot : array (n, n_uniqueelements)
        Array of encoded feature.
    '''
    cv = cv.flatten()
    # Get unique values and create mapping
    unique_vals = np.unique(cv)
    num_classes = len(unique_vals)
    value_to_index = {val: idx for idx, val in enumerate(unique_vals)}
    # Map values to indices
    indices = np.array([value_to_index[val] for val in cv])
    # Create one-hot encoded array
    one_hot = np.eye(num_classes)[indices]
    return one_hot

#%%% evaluate
def RMSE_results(x_test, y_test, results):
    '''
    Calculates the root mean square error of the training or test set.

    Parameters
    ----------
    x_test : array (n, d)
        Array containing the features for the control.
    y_test : array (n, 1)
        Array of actual values.
    results : dict
        Dictionary containing the results of the run function.

    Returns
    -------
    RMSE : scalar
        Root mean square eroor value.
    '''
    th = results["th"]
    th0 = results["th0"]
    n, d = x_test.shape
    summa = 0
    for i in range(n):
        guess = np.dot(th.T,x_test[i, :])+th0
        summa += (guess - y_test[i])**2
    RMSE = np.sqrt(summa / n)
    return RMSE

def eval_crossval(x_test, y_test, crossval_results):
    n = len(crossval_results)
    fold_RMSEs = [0] * n
    for i in range(n):
        fold_RMSEs[i] = RMSE_results(x_test, y_test, crossval_results[i])
    return fold_RMSEs

#%%% Setup
def setup_config(x, y, th_init, th0_init, lam, tmax = None, eta = None, obj_func = None, 
        homogenity = None, solve = None, gradient = None, stochastic = None):
    '''
    Function to create config variable as input for the "run" and "cross_validate" functions.

    Parameters
    ----------
    x : array (n, d)
        Features.
    y : array (n, 1)
        Target variable or labels.
    th_init : array (d, 1)
        Initial weights.
    th0_init : scalar
        Initial bias.
    lam : scalar
        Regularization parameter.
    tmax : scalar, optional
        The maximum number of iterations. The default is 100.
    eta : scalar, optional
        The (initial) step size (learning rate). The default is 0.1.
    obj_func : string, optional (svm/llc/ols)
        Setting for what objective function to use. The default is None.
    homogenity : string, optional (hom/het)
        Setting for using homogen or heterogen form for solution. The default is None.
    solve : string, optional (dir/gd)
        Setting for direct solution, or approximation with gradient descent. The default is None.
    gradient : string, (ana/num)
        Setting for using analytical or numerical gradient. The default is None.
    stochastic : string, optional (batch/stoch)
        Setting for batch or stochastic gradient descent. The default is None.

    Returns
    -------
    config : list
        Variable containing inputs required for "run" and "cross_validate" functions.
    '''
    
    if obj_func == None:
        obj_func = input("What is the method of solution? (svm/llc/ols)")
    valid_obj_func = ["svm", "llc","ols"]
    if obj_func not in valid_obj_func:
        raise ValueError(f"Method must be one of {valid_obj_func}")
    
    if homogenity == None:
        homogenity = input("Form homogen or heterogen? (hom/het)")
    valid_homogenity = ["hom", "het"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")  
    
    if obj_func == "ols" and homogenity == "hom":
        if solve == None:
            solve = input("Solve directly or with gradient descent? (dir/gd)")
        valid_solve = ["dir", "gd"]
        if solve not in valid_solve:
            raise ValueError(f"Solve must be one of {valid_solve}")
    else:
        solve = "gd"
    
    if obj_func != "ols" or solve != "dir":
        if gradient == None:
            gradient = input("Use analytical or numerical gradient? (ana/num)")
        valid_gradient = ["ana", "num"]
        if gradient not in valid_gradient:
            raise ValueError(f"Gradient must be one of {valid_gradient}")
            
        if stochastic == None:
            stochastic = input("Use batch or stochastic gradient descent? (batch/stoch)")
        valid_stochastic = ["batch", "stoch"]
        if stochastic not in valid_stochastic:
            raise ValueError(f"Stochasticity must be one of {valid_stochastic}")
            
    if tmax == None:
        tmax = input("Maximum number of iterations: ")
        if type(tmax) != int:
            raise ValueError("tmax must be an integer")
            
    if eta == None:
        eta = input("The learning rate: ")
        if type(eta) not in [int, float]:
            raise ValueError("tmax must be an integer, or a float")
            
    config = [x, y, th_init, th0_init, lam, tmax, eta, obj_func, 
              homogenity, solve, gradient, stochastic]
    
    return config