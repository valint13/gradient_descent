# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:09:14 2025

@author: Bálint
"""
import numpy as np
import src.utilities as util
import src.objectives as obj
import src.optimization as opt
import src.plotting as plot
#%% Single optimization
def run(x, y, th_init, th0_init, lam, tmax = 100, eta = 0.1, obj_func = None, 
        homogenity = None, solve = None, gradient = None, stochastic = None):
    '''
    Wrapper function for running gradient descent algorithm with a variety of settings, 
    or OLS analytically for linear regression.
    
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
    results : dict
        Conatins relevant weights, bias, training data, plots, and config.
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
    
    n, d = x.shape
    # Define supporting dictionaries for function mapping
    objectives = {'svm': obj.svm_obj, 'llc': obj.llc_obj, 'ols': obj.OLS_obj}
    analytical_grads = {'llc': obj.llc_grad, 'ols': obj.ridge_grad}
    descent_hom = {'batch': opt.gd_homogen, 'stoch': opt.sgd_homogen}
    descent_het = {'batch': opt.gd_heterogen, 'stoch': opt.sgd_heterogen}
    
    # Map abbreviated form names to full names used by underlying functions
    homogenity_map = {'hom': 'homogen', 'het': 'heterogen'}
    
    # Direct solution for OLS
    if obj_func == "ols" and solve == "dir":
        th_hom = opt.OLS(x, y, lam, "solve")
        th, th0 = util.homogen_th_unpack(th_hom)
        
        if d == 2:
            ax1 = plot.setup_plot(min(x[:, 0]), max(x[:, 0]), min(x[:, 1]), max(x[:, 1]), center = True, grid = True, 
                               title = f"Lin sep characterized by th = {th}, th0 = {th0}, lambda = {lam}", 
                               xlabel = "x1", ylabel = "x2", aspect = "equal")
            ax1 = plot.plot_data(x, y, ax1)
            plot.plot_separator(ax1, th, th0, x)
        else: ax1 = None
        
        config = [x, y, th_init, th0_init, lam, tmax, eta, obj_func, homogenity, solve, gradient, stochastic]
        results = {"th": th, "th0": th0, "ax1": ax1, "config": config}
        return results
    # Solutions with gradient descent
    elif gradient == "ana":
        if obj_func in analytical_grads.keys():
            grad = analytical_grads[obj_func]
        else:
            print(f"Warning: Analytical gradient not available for {obj_func}; using numerical gradient instead.")
            homogenity_full = homogenity_map[homogenity]
            grad = obj.num_grad(objectives[obj_func], delta=eta/10, homogenity=homogenity_full, x=x, y=y, lam=lam)
    else:
        homogenity_full = homogenity_map[homogenity]
        grad = obj.num_grad(objectives[obj_func], delta=eta/10, homogenity=homogenity_full, x=x, y=y, lam=lam)
        
    if obj_func == "svm":
        step_size_fn = util.step_svm_min
    elif stochastic == "batch":
        step_size_fn = util.step_const
    else:
        step_size_fn = util.step_dec
        
    if homogenity == "hom":
        descent_func = descent_hom[stochastic]
        th_hom, ths_hom, fs = descent_func(objectives[obj_func], grad, th_init, 
                                        th0_init, eta, step_size_fn, 1e-6, tmax, x, y, lam)
        th, th0 = util.homogen_th_unpack(th_hom)
        ths, th0s = util.homogen_ths_unpack(ths_hom)
    else:
        descent_func = descent_het[stochastic]
        th, th0, ths, th0s, fs = descent_func(objectives[obj_func], grad, th_init, 
                                              th0_init, eta, step_size_fn, 1e-6, tmax, x, y, lam)
    
    # Plotting
    if d == 2:    
        ax1 = plot.setup_plot(min(x[:, 0]), max(x[:, 0]), min(x[:, 1]), max(x[:, 1]), center = True, grid = True, 
                           title = f"Lin sep characterized by th = {th}, th0 = {th0}, lambda = {lam}", 
                           xlabel = "x1", ylabel = "x2", aspect = "equal")
        ax1 = plot.plot_data(x, y, ax1)
        plot.plot_separator(ax1, th, th0, x)
    else: ax1 = None
    ax2 = plot.plot_obj(fs)
    
    # Package and return
    config = [x, y, th_init, th0_init, lam, tmax, eta, obj_func, homogenity, solve, gradient, stochastic]
    results = {"th": th, "th0": th0, "ths": ths, "th0s": th0s, "fs": fs, "ax1": ax1, "ax2": ax2, "config": config}
    return results

#%% Cross validation
def cross_validate(x, y, z, config, feature_config):
    '''
    Function running z-fold cross validation on a dataset.

    Parameters
    ----------
    x : array (n, d)
        Features.
    y : array (n, 1)
        Target variable or labels.
    z : int
       Number of folds.
    config : list
        Variable holding arguments for "run" function.
    feature_config : dict
        Dictionary holding the names of the features, indices, and feature transformation information.

    Returns
    -------
    crossval_results : list (z, 1)
        List holding the training results for each fold.
    fold_RMSEs : list (z, 1)
        List holding the RMSE values for each fold.
    standardization_params : dict
        Holds the lists (z, 1) of training standardization parameters for each fold.
    '''
    x_folds, y_folds = util.split_data(x, y, z)
    crossval_results = [0] * z
    fold_RMSEs = [0] * z
    mus_train = [[0] * len(feature_config["standardize"]) for _ in range(z)]
    sigs_train = [[0] * len(feature_config["standardize"]) for _ in range(z)]
    for i in range(z):
        # compose training and test sets
        x_test = x_folds[i].copy()
        y_test = y_folds[i].copy()
        x_train_folds = x_folds[:i] + x_folds[i+1:]
        y_train_folds = y_folds[:i] + y_folds[i+1:]
        x_train = np.vstack(x_train_folds)
        y_train = np.vstack(y_train_folds)
        # standardize features in training set
        if feature_config["standardize"] != None:
            n, d = x_train.shape
            paramindex = 0
            for j in range(d):
                if feature_config["map"][j] in feature_config["standardize"]:
                    x_train[:,j], mus_train[i][paramindex], sigs_train[i][paramindex] = util.standardize(x_train[:, j])
                    paramindex += 1
            # standardize features in test set
            paramindex = 0
            for k in range(d):
                if feature_config["map"][k] in feature_config["standardize"]:
                    x_test[:,k] = util.standardize(x_test[:, k], mus_train[i][paramindex], sigs_train[i][paramindex])
                    paramindex += 1
        # run cross validation on fold i
        config[0] = x_train
        config[1] = y_train
        crossval_results[i] = run(*config)
        fold_RMSEs[i] = util.RMSE_results(x_test, y_test, crossval_results[i])
    standardization_params = {"mus": mus_train, "sigs": sigs_train}  
    return crossval_results, fold_RMSEs, standardization_params