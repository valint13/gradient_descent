# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:54:44 2025

@author: Bálint
"""

import numpy as np
from src.utilities import homogenize_x
from src.utilities import homogenize_th
from src.utilities import step_const
from src.utilities import step_dec
from src.objectives import svm_obj
from src.objectives import llc_obj
from src.objectives import OLS_obj
from src.objectives import num_grad
#%% definitions
#%%% homogen batch gradient descent
def gd_homogen(f, df=None, th_init=0, th0_init=0, eta=0.1, step_size_fn=step_const, epsilon=1e-6, 
               tmax=100, x=None, y=None, lam=None):
    '''
    Parameters
    ----------
    f : function   input: array (d)    output: scalar
        The objective function of the gradient descent.
    df : function   input: array (d)    output: array
        The gradient of the objective function.
    th_init : array (d)
        The initial condition of th.
    eta : scalar
        The (initial) step size.
    step_size_fn : function     input: scalar,int  output: scalar
        A function that determines the step size.
    epsilon: scalar
        Convergence condition.
    tmax : int
        The maximum number of iterations.
    x : array (n, d)
        The input data (features).
    y : array (n)
        The labels.
    lam : scalar
        Regularization parameter.

    Returns
    -------
    The value th at the final step, the list of values of f and th for each iteration.
    '''
    x = homogenize_x(x)
    th_init = homogenize_th(th_init, th0_init)
    if f in [svm_obj,llc_obj,OLS_obj]:
        if df == None:
            df = num_grad(f,eta/10,homogenity="homogen",x=x,y=y,lam=lam)
        
        th_init = np.atleast_1d(th_init)
        d = th_init.size
        ths = th_init.reshape(1, d)
        fs = np.atleast_1d(f(x, y, th_init, 0., lam, homogenity = "homogen"))
        for i in range(tmax):
            # Compute new values
            step = step_size_fn(eta, i)
            th_new = ths[-1] - step * df(ths[-1], 0, x, y, lam, homogenity = "homogen")
            f_new = f(x, y, th_new, 0, lam, homogenity = "homogen")
            # Append new values
            ths = np.vstack((ths, th_new.T))
            fs = np.append(fs,f_new)
            # Check convergence
            if abs(fs[-1] - fs[-2]) < epsilon:
                print(f"Solution converged after {i} iterations")
                break
    else:
        if df == None:
            df = num_grad(f,eta/10)
        
        th_init = np.atleast_1d(th_init)
        d = th_init.size
        ths = th_init.reshape(1, d)
        fs = np.atleast_1d(f(th_init))
        for i in range(tmax):
            # Compute new values
            step = step_size_fn(eta, i)
            th_new = ths[-1].reshape(-1,1) - step * df(ths[-1], homogenity = "homogen")
            f_new = f(th_new, homogenity = "homogen")
            # Append new values
            ths = np.vstack((ths, th_new))
            fs = np.append(fs,f_new)
            # Check convergence
            if abs(fs[-1] - fs[-2]) < epsilon:
                print(f"Solution converged after {i} iterations")
                break
            
    # Final values    
    th = ths[-1]
    return th,ths,fs
#%%% heterogen batch gradient descent
def gd_heterogen(f, df=None, th_init=0, th0_init=0, eta=0.1, step_size_fn=step_const, epsilon=1e-6, 
                 tmax=100, x=None, y=None, lam=None):
    '''
    Parameters
    ----------
    f : function   output: scalar
        The objective function of the gradient descent.
    df : function   output: array
        The gradient of the objective function.
    th_init : array (d)
        The initial condition of th.
    th0_init : scalar
        The initial condition of th0.
    eta : scalar
        The (initial) step size.
    step_size_fn : function     input: scalar,int  output: scalar
        A function that determines the step size.
    epsilon: scalar
        Convergence condition.
    tmax : int
        The maximum number of iterations.
    x : array (n, d)
        The input data (features).
    y : array (n)
        The labels.
    lam : scalar
        Regularization parameter.

    Returns
    -------
    The values th,th0 at the final step, the list of values of f and th,th0 for each iteration.
    '''
    if f in [svm_obj,llc_obj,OLS_obj]:
        if df == None:
            df = num_grad(f, eta/10, homogenity = "heterogen", x = x, y = y, lam = lam)
            
        th_init = np.atleast_1d(th_init)
        d = th_init.size
        ths = th_init.reshape(1, d)
        th0s = np.atleast_1d(th0_init)
        fs = np.atleast_1d(f(x, y, th_init, th0_init, lam, homogenity = "heterogen"))
        for i in range(tmax):
            # Compute new values
            step = step_size_fn(eta, i)
            th_new = ths[-1] - step * df(ths[-1], th0s[-1], x, y, lam, homogenity = "heterogen")[:-1]
            th0_new = th0s[-1] - step * df(ths[-1], th0s[-1], x, y, lam, homogenity = "heterogen")[-1]
            f_new = f(x, y, th_new, th0_new, lam, homogenity = "heterogen")
            # Append new values
            ths = np.vstack((ths, th_new))
            th0s = np.append(th0s,th0_new)
            fs = np.append(fs,f_new)
            # Check convergence
            if abs(fs[-1] - fs[-2]) < epsilon:
                print(f"Solution converged after {i} iterations")
                break
    else:
        if df == None:
            df = num_grad(f,eta/10)
        
        th_init = np.atleast_1d(th_init)
        d = th_init.size
        ths = th_init.reshape(1, d)
        th0s = np.atleast_1d(th0_init) 
        fs = np.atleast_1d(f(th_init, th0_init, homogenity = "heterogen"))
        for i in range(tmax):
            # Compute new values
            step = step_size_fn(eta, i)
            th_new = ths[-1] - step * df(ths[-1],th0s[-1],homogenity = "heterogen")
            th0_new = th0s[-1] - step * df(ths[-1],th0s[-1],homogenity = "heterogen")
            f_new = f(th_new, th0_new, homogenity = "heterogen")
            # Append new values
            ths = np.vstack((ths, th_new))
            th0s = np.append(th0s,th0_new)
            fs = np.append(fs,f_new)
            # Check convergence
            if abs(fs[-1] - fs[-2]) < epsilon:
                print(f"Solution converged after {i} iterations")
                break
    # Final values    
    th = ths[-1]
    th0 = th0s[-1]
    return th,th0,ths,th0s,fs
#%%% homogen stochastic gradient descent
def sgd_homogen(f, df=None, th_init=0, th0_init=0, eta=0.1, step_size_fn=step_dec, epsilon=1e-6,
                tmax=100, x=None, y=None, lam=None):
    '''
    Parameters
    ----------
    f : function   output: scalar
        The objective function of the stochastic gradient descent.
    df : function   output: array
        The gradient of the objective function for a single data point.
    th_init : array (d)
        The initial condition of th.
    eta : scalar
        The (initial) step size.
    step_size_fn : function     input: scalar,int  output: scalar
        A function that determines the step size.
    epsilon: scalar
        Convergence condition.
    tmax : int
        The maximum number of iterations.
    x : array (n, d)
        The input data (features).
    y : array (n)
        The labels.
    lam : scalar
        Regularization parameter.

    Returns
    -------
    The value th at the final step, the list of values of f and th for each iteration.
    '''
    x = homogenize_x(x)
    th_init = homogenize_th(th_init, th0_init)
    if df is None:
        df = num_grad(f, eta/10, homogenity = "homogen", x = x, y = y, lam = lam)
    
    th_init = np.atleast_1d(th_init)
    d = th_init.size
    ths = th_init.reshape(1, d)
    fs = np.atleast_1d(f(x[:1], y[:1], th_init, 0, lam, homogenity = "homogen"))  # Initial f on first point
    n = x.shape[0]  # Number of data points
    
    for i in range(tmax):
        # Randomly select a single data point
        idx = np.random.randint(0, n)
        x_i = x[idx:idx+1]
        y_i = y[idx:idx+1]
        # Compute new values
        step = step_size_fn(eta, i)
        th_new = ths[-1] - step * df(ths[-1], 0, x_i, y_i, lam, homogenity = "homogen")
        f_new = f(x_i, y_i, th_new, 0, lam, homogenity = "homogen")
        # Append new values
        ths = np.vstack((ths, th_new.T))
        fs = np.append(fs, f_new)
        # Check convergence
        if abs(fs[-1] - fs[-2]) < epsilon:
            print(f"Solution converged after {i} iterations")
            break
    
    # Final values    
    th = ths[-1]
    return th, ths, fs
#%%% heterogen stochastic gradient descent
def sgd_heterogen(f, df=None, th_init=0, th0_init=0, eta=0.1, step_size_fn=step_dec, epsilon=1e-6, 
                  tmax=100, x=None, y=None, lam=None):
    '''
    Parameters
    ----------
    f : function   output: scalar
        The objective function of the stochastic gradient descent.
    df : function   output: array
        The gradient of the objective function for a single data point.
    th_init : array (d)
        The initial condition of th.
    th0_init : scalar
        The initial condition of th0.
    eta : scalar
        The (initial) step size.
    step_size_fn : function     input: scalar,int  output: scalar
        A function that determines the step size (not used in current implementation).
    epsilon: scalar
        Convergence criterion.
    tmax : int
        The maximum number of iterations.
    x : array (n, d)
        The input data (features).
    y : array (n)
        The labels.
    lam : scalar
        Regularization parameter.

    Returns
    -------
    The values th, th0 at the final step, the list of values of f, th, th0 for each iteration.
    '''
    if df is None:
        df = num_grad(f, eta/10, homogenity = "heterogen", x = x, y = y, lam = lam)
    
    th_init = np.atleast_1d(th_init)
    d = th_init.size
    ths = th_init.reshape(1, d)
    th0s = np.atleast_1d(th0_init)
    fs = np.atleast_1d(f(x[:1], y[:1], th_init, th0_init, lam, homogenity = "heterogen"))  # Initial f on first point
    n = x.shape[0]  # Number of data points
    
    for i in range(tmax):
        # Randomly select a single data point
        idx = np.random.randint(0, n)
        x_i = x[idx:idx+1]
        y_i = y[idx:idx+1]
        # Compute new values
        step = step_size_fn(eta, i)
        th_new = ths[-1] - step * df(ths[-1], th0s[-1], x_i, y_i, lam, homogenity = "heterogen")[:-1]
        th0_new = th0s[-1] - step * df(ths[-1], th0s[-1], x_i, y_i, lam, homogenity = "heterogen")[-1]
        f_new = f(x_i, y_i, th_new, th0_new, lam)  # Evaluate f on single point
        # Append new values
        ths = np.vstack((ths, th_new))
        th0s = np.append(th0s, th0_new)
        fs = np.append(fs, f_new)
        # Check convergence
        if abs(fs[-1] - fs[-2]) < epsilon:
            print(f"Solution converged after {i} iterations")
            break
    
    # Final values    
    th = ths[-1]
    th0 = th0s[-1]
    return th, th0, ths, th0s, fs
#%%% OLS solver
def OLS(x, y, lam, method = "solve"):
    '''
    Parameters
    ----------
    x : array (n, d)
        The input data (features).
    y : array (n)
        The labels.
    lam : scalar
        Regularization parameter.
    method : string ("solve", "invert")
        Switches the solution method of the system of linear equations.
        solve: directly solving for A,b
        invert: invert A and calculate
        The default is "solve".

    NOTE: Expects the hypothesis to be in homogen form with th that is expanded 
    with th0 as its last element, and x that is expanded with 1 as its last element.
    
    Returns
    -------
    th_opt : array (d)
        DESCRIPTION.
    '''
    valid_methods = ["solve", "invert"]
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")
    
    x = homogenize_x(x)
    n, d = x.shape
    
    if method == "solve":
        A = x.T @ x + n * lam * np.identity(d)
        A_inv = np.linalg.inv(A)
        b = np.dot(x.T, y)
        th_opt = np.dot(A_inv, b)
        
    elif method == "invert":
        A = x.T @ x + n * lam * np.identity(d)
        b = np.dot(x.T, y)
        th_opt = np.linalg.solve(A, b)
            
    return th_opt