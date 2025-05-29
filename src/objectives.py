# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:50:40 2025

@author: Bálint
"""

import numpy as np
import math
import inspect
#%% definitions
#%%% general
def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)
#%%% SVM
def hinge(h):
    '''
    Parameters
    ----------
    h : scalar
        The margin relative to the reference margin in hinge loss.

    Returns
    -------
    The value of the hinge loss for a given point (margin).
    '''
    if h<1:
        ans=1-h
    else:
        ans=0
    return ans

def hinge_loss(x,y,th,th0):
    '''
    Parameters
    ----------
    x : array (d)
        Datapoint coordinates.
    y : scalar
        Datapoint label.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).

    Returns
    -------
    Lh : scalar
        Hinge loss of a given point.
    '''
    x=np.atleast_1d(x)
    th=np.atleast_1d(th)
    h=y*(np.dot(th.T,x)+th0)
    Lh=float(hinge(h))
    return Lh

def svm_obj(x, y, th, th0, lam = 0, homogenity = "heterogen"):
    '''
    Parameters
    ----------
    x : array (n x d)
        Coordinates of all the data points.
    y : array (n)
        Labels of all the data points.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).
    lam : scalar
        Regularizer hyperparameter.
    homogenity: string
        The form in which the hypothesis is expressed.
        NOTE: The homogen form expects th that is expanded with th0 as its last element, 
        and x that is expanded with 1 as its last element.

    Returns
    -------
    The value of the SVM objective function for hinge-loss.
    '''
    valid_homogenity = ["homogen", "heterogen"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")
        
    n, d = x.shape
    th = np.atleast_1d(th)
    Lh_sum = 0
    if homogenity == "homogen":
        th_mod = np.append(th[:-1], 0)
        for i in range(n):
            Lh_sum += hinge_loss(x[i,:], y[i], th, 0)
        J = 1 / n * Lh_sum + lam / 2 * sum(th_mod**2)
        
    elif homogenity == "heterogen":
        for i in range(n):
            Lh_sum += hinge_loss(x[i,:], y[i], th, th0)
        J = 1 / n * Lh_sum + lam / 2 * sum(th**2)
    
    return J
#%%% LLC
def sigmoid(th,th0,x):
    '''
    Parameters
    ----------
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).
    x : array (d)
        Coordinates of given data point.

    Returns
    -------
    sig : scalar
        The value of the sigmoid function for th,th0 in point x.
    '''
    sig=1/(1+math.e**-(np.dot(th,x)+th0))
    return sig

def nll_loss(x,y,th,th0):
    '''
    Parameters
    ----------
    x : array (d)
        Coordinates of given data point.
    y : scalar
        Label of given data point.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).

    Returns
    -------
    L_nll : scalar
        The negative log-likelihood loss for given th,th0 at given point x.
    '''
    y = y.item() if isinstance(y, np.ndarray) else y    # convert y into scalar for consistency
    guess=sigmoid(th,th0,x)
    # Clamp guess to avoid log(0) or log(1)
    epsilon = 1e-10
    guess = np.clip(guess, epsilon, 1 - epsilon)
    if y==-1: y_adjusted=0  #transform the labels to fit the logarithmic formulation
    else: y_adjusted=y
    
    L_nll=-(y_adjusted*np.log(guess)+(1-y_adjusted)*np.log(1-guess))
    return L_nll

def llc_obj(x, y, th, th0, lam, homogenity = "heterogen"):
    '''
    Parameters
    ----------
    x : array (n x d)
        Coordinates of all the data points.
    y : array (n)
        Labels of all the data points.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).
    lam : scalar
        Regularizer hyperparameter.
    homogenity: string
        The form in which the hypothesis is expressed.
        NOTE: The homogen form expects th that is expanded with th0 as its last element, 
        and x that is expanded with 1 as its last element.

    Returns
    -------
    J : scalar
        The value of the LLC objective function with nll loss.
    '''
    valid_homogenity = ["homogen", "heterogen"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")
        
    n, d = x.shape
    L_sum = 0
    if homogenity == "homogen":
        th_mod = np.append(th[:-1], 0)
        for i in range(n):
            L_sum += nll_loss(x[i,:], y[i], th, 0)
        J = 1 / n * L_sum + lam / 2 * sum(th_mod**2)
        
    elif homogenity == "heterogen":
        for i in range(n):
            L_sum += nll_loss(x[i,:], y[i], th, th0)
        J = 1 / n * L_sum + lam / 2 * sum(th**2)
    
    return J

def llc_grad(th, th0, x, y, lam, homogenity = "heterogen"):
    '''
    Parameters
    ----------
    x : array (n x d)
        Coordinates of all the data points.
    y : array (n)
        Labels of all the data points.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).
    lam : scalar
        Regularizer hyperparameter.
    homogenity: string
        The form in which the hypothesis is expressed.
        NOTE: The homogen form expects th that is expanded with th0 as its last element, 
        and x that is expanded with 1 as its last element.

    Returns
    -------
    grad : array (d+1)
        Gradient of the LLC objective function with respect to th and th0.
        First d elements are the gradient with respect to th, last element is for th0.
    '''
    valid_homogenity = ["homogen", "heterogen"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")
        
    n, d = x.shape
    grad_th = np.zeros(d)
    grad_th0 = 0
    y_adjusted=np.zeros(n)
    for i in range(n):
        if y[i] == -1: y_adjusted[i] = 0
        else: y_adjusted[i] = y[i]

    if homogenity == "homogen":
        th_mod = np.append(th[:-1], 0)
        # loop through gradient components
        for j in range(d):
            summa = 0
            # loop through each data point
            for i in range(n):
                summa += (sigmoid(th, 0, x[i, :]) - y_adjusted[i]) * x[i, j]
            
            grad_comp = summa/n + lam * th_mod[j]
            grad_th[j] = grad_comp
        grad = grad_th
        
    elif homogenity == "heterogen":
        # loop through gradient components
        for j in range(d):
            summa = 0
            if j == 0: summa_th0 = 0
            # loop through each data point
            for i in range(n):
                summa += (sigmoid(th, th0, x[i, :]) - y_adjusted[i]) * x[i, j]
                if j == 0: summa_th0 += sigmoid(th, th0, x[i, :]) - y_adjusted[i]
            
            grad_comp = summa / n + lam * th[j]
            grad_th[j] = grad_comp
        grad_th0 = summa_th0 / n
        grad = np.append(grad_th, grad_th0) # Combine gradients into a single vector
        
    return grad
#%%% OLS (linear regression)
def squared_loss(x, y, th, th0):
    '''
    Parameters
    ----------
    x : array (d)
        Coordinates of given data point.
    y : scalar
        Actual value belonging to given data point.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).

    Returns
    -------
    L_sl : scalar
        The squared loss for given th,th0 at given point x.
    '''
    g = np.dot(th, x) + th0
    L_sl = (g - y)**2
    return L_sl

def OLS_obj(x, y, th, th0, lam, homogenity = "heterogen"):
    '''
    Parameters
    ----------
    x : array (n x d)
        Coordinates of all the data points.
    y : array (n x 1)
        Labels of all the data points.
    th : array (d x 1)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).
    lam : scalar
        Regularizer hyperparameter.
    homogenity: string
        The form in which the hypothesis is expressed.
        NOTE: The homogen form expects th that is expanded with th0 as its last element, 
        and x that is expanded with 1 as its last element.

    Returns
    -------
    J : scalar
        The value of the linear regression objective function with squared loss.
    '''
    valid_homogenity = ["homogen", "heterogen"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")
        
    n, d = x.shape
    th = th.reshape(-1,1)
    if homogenity == "homogen":
        th_mod = np.append(th[:-1], 0).reshape(-1,1)
        J = 1 / n * np.dot((np.dot(x, th) - y).T, (np.dot(x, th) - y)) + lam / 2 * sum(th_mod**2)
    elif homogenity == "heterogen":
        TH0 = np.full((n, 1), th0)
        J = 1 / n * np.dot((np.dot(x, th) + TH0 - y).T, (np.dot(x, th) + TH0 - y)) + lam / 2 * sum(th**2)
    return J

def ridge_grad(th, th0, x, y, lam, homogenity = "heterogen"):
    '''
    Parameters
    ----------
    x : array (n x d)
        Coordinates of all the data points.
    y : array (n)
        Labels of all the data points.
    th : array (d)
        Separator parameter (weight).
    th0 : scalar
        Separator parameter (bias).
    lam : scalar
        Regularizer hyperparameter.
    homogenity: string
        The form in which the hypothesis is expressed.
        NOTE: The homogen form expects th that is expanded with th0 as its last element, 
        and x that is expanded with 1 as its last element.

    Returns
    -------
    grad : array (d+1)
        Gradient of the OLS objective function with respect to th and th0.
        First d elements are the gradient with respect to th, last element is for th0.
    '''
    valid_homogenity = ["homogen", "heterogen"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")
    
    n, d = x.shape
    th = th.reshape(-1,1)
    if homogenity == "homogen":
        th_mod = np.append(th[:-1], 0).reshape(-1,1)
        grad_th = 2 / n * np.dot(x.T, np.dot(x, th) - y) + lam * th_mod
        grad = grad_th.flatten()
        
    elif homogenity == "heterogen":
        grad_th = 2 / n * np.dot(x.T, np.dot(x, th) - y) + lam * th
        ones = np.ones(n).reshape(-1,1)
        grad_th0 = 2 / n * np.dot(ones.T, np.dot(x, th) + th0 - y)
        grad = np.append(grad_th, grad_th0).flatten() # Combine gradients into a single vector
        
    return grad
#%%% Numerical gradient
def num_grad(f,delta=0.001,method="fw",homogenity="heterogen",x=None,y=None,lam=None):
    '''
    Parameters
    ----------
    f : function
        The function that we want to numerically determine the gradient of.
    delta : scalar
        The arbitrarily small step for the finite difference form.
    method : string, can be: fw,bw,sym
        Method for the finite difference approximation of grad_f. The default is "fw".

    Raises
    ------
    ValueError
        Choose one of the valid methods.

    Returns
    -------
    Returns a function that calculates the gradient of f in given point th.
    '''
    valid_methods = ["fw", "bw", "sym"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    valid_homogenity = ["homogen", "heterogen"]
    if homogenity not in valid_homogenity:
        raise ValueError(f"Homogenity must be one of {valid_homogenity}")

    def grad_f(th,th0=None,x=x,y=y,lam=lam,homogenity=homogenity):
        '''
        Parameters
        ----------
        th : array (d)
            Parameters.

        Returns
        -------
        gradf : array (d)
            The gradient in point th.
        '''
        # Get the arguments of f
        sig = inspect.signature(f)
        f_param_names = list(sig.parameters.keys())
        all_args = {"x": x, "y": y, "th": th, "th0": th0, "lam": lam, "homogenity": homogenity}
        args = tuple(all_args[name] for name in f_param_names)
        
        th=np.atleast_1d(th)
        if th0 != None: th0=np.atleast_1d(th0)
        dim=th.size
        gradf=np.zeros(dim)
        if homogenity == "homogen":
            if method=="fw":
                for i in range(dim):
                    dv=np.zeros(dim); dv[i]=delta
                    # adjust argument th+ 
                    all_args_plus = all_args.copy()
                    all_args_plus['th'] = th + dv
                    args_plus = tuple(all_args_plus[name] for name in f_param_names)
                    # calculate gradient component
                    gradf[i]=(f(*args_plus)-f(*args))/delta
            elif method=="bw":
                for i in range(dim):
                    dv=np.zeros(dim); dv[i]=delta
                    # adjust argument th-
                    all_args_minus = all_args.copy()
                    all_args_minus['th'] = th - dv
                    args_minus = tuple(all_args_minus[name] for name in f_param_names)
                    # calculate gradient component
                    gradf[i]=(f(*args)-f(*args_minus))/delta
            elif method=="sym":
                for i in range(dim):
                    dv=np.zeros(dim); dv[i]=delta
                    # adjust argument th+ 
                    all_args_plus = all_args.copy()
                    all_args_plus['th'] = th + dv
                    args_plus = tuple(all_args_plus[name] for name in f_param_names)
                    # adjust argument th-
                    all_args_minus = all_args.copy()
                    all_args_minus['th'] = th - dv
                    args_minus = tuple(all_args_minus[name] for name in f_param_names)
                    # calculate gradient component
                    gradf[i]=(f(*args_plus)-f(*args_minus))/(2*delta)
            return gradf
        
        elif homogenity == "heterogen":
            if method=="fw":
                for i in range(dim):
                    dv=np.zeros(dim); dv[i]=delta
                    # adjust argument th+ 
                    all_args_plus = all_args.copy()
                    all_args_plus['th'] = th + dv
                    args_plus = tuple(all_args_plus[name] for name in f_param_names)
                    # adjust argument th0+ 
                    all_th0_args_plus=all_args.copy()
                    all_th0_args_plus["th0"]=th0+delta
                    th0_args_plus = tuple(all_th0_args_plus[name] for name in f_param_names)
                    # calculate gradient component
                    gradf[i]=(f(*args_plus)-f(*args))/delta
                    df_dth0=(f(*th0_args_plus)-f(*args))/delta
            elif method=="bw":
                for i in range(dim):
                    dv=np.zeros(dim); dv[i]=delta
                    # adjust argument th-
                    all_args_minus = all_args.copy()
                    all_args_minus['th'] = th - dv
                    args_minus = tuple(all_args_minus[name] for name in f_param_names)
                    # adjust argument th0-
                    all_th0_args_minus=all_args.copy()
                    all_th0_args_minus["th0"]=th0-delta
                    th0_args_minus = tuple(all_th0_args_minus[name] for name in f_param_names)
                    # calculate gradient component
                    gradf[i]=(f(*args)-f(*args_minus))/delta
                    df_dth0=(f(*args)-f(*th0_args_minus))/delta
            elif method=="sym":
                for i in range(dim):
                    dv=np.zeros(dim); dv[i]=delta
                    # adjust argument th+ 
                    all_args_plus = all_args.copy()
                    all_args_plus['th'] = th + dv
                    args_plus = tuple(all_args_plus[name] for name in f_param_names)
                    # adjust argument th-
                    all_args_minus = all_args.copy()
                    all_args_minus['th'] = th - dv
                    args_minus = tuple(all_args_minus[name] for name in f_param_names)
                    # adjust argument th0+ 
                    all_th0_args_plus=all_args.copy()
                    all_th0_args_plus["th0"]=th0+delta
                    th0_args_plus = tuple(all_th0_args_plus[name] for name in f_param_names)
                    # adjust argument th0-
                    all_th0_args_minus=all_args.copy()
                    all_th0_args_minus["th0"]=th0-delta
                    th0_args_minus = tuple(all_th0_args_minus[name] for name in f_param_names)
                    # calculate gradient component
                    gradf[i]=(f(*args_plus)-f(*args_minus))/(2*delta)
                    df_dth0=(f(*th0_args_plus)-f(*th0_args_minus))/(2*delta)
            return np.append(gradf,df_dth0) #this way grad_f will return one array, that will include the element corresponding to th0 (the bias)
    return grad_f
