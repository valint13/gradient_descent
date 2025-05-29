# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:55:02 2025

@author: Bálint
"""

import numpy as np
import matplotlib.pyplot as plt
#%% definitions
def setup_plot(xmin, xmax, ymin, ymax, center = False, grid = False, title = None,
                 xlabel = None, ylabel = None, aspect="auto"):
    """
    Set up axes for plotting
    xmin, xmax, ymin, ymax = (float) plot extents
    aspect = "equal" or "auto"
    Return matplotlib axes
    """
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_label_position("bottom")
        ax.yaxis.set_label_position("left")
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    x_label_offset = eps * (xmax - xmin)  # 5% of xe x range for padding
    y_label_offset = eps * (ymax - ymin)  # 5% of xe y range for padding
    plt.xlim(xmin-x_label_offset, xmax+x_label_offset)
    plt.ylim(ymin-y_label_offset, ymax+y_label_offset)
    if title: ax.set_title(title,y=1.05)
    if xlabel: ax.text(1.0, 0.45, xlabel, transform=ax.transAxes, ha="left", va="center")
    if ylabel: ax.text(0.55, 1.0, ylabel, transform=ax.transAxes, ha="center", va="bottom")
    if grid: ax.grid(True)
    ax.set_aspect(aspect)
    return ax

def plot_gd_1D(f,ths,fs,ax=None):
    '''
    Parameters
    ----------
    f : function
        the objective function of the gradient descent.
    ths : array (t)
        The th values at each timestep.
    fs : array (t)
        The objective function values at each timestep.
    ax : ax object
        Existing ax object. The default is None.

    Returns
    -------
    Draws the plot of the function, and the points touched during the gradient descent for 1D feature space.
    '''
    x_min=np.min(ths)
    x_max=np.max(ths)
    y_min=np.min(fs)
    y_max=np.max(fs)
    if abs(x_max-ths[-1])>=abs(x_min-ths[-1]):
        x_limh=x_max+(np.max(ths)-np.min(ths))/100*5
        x_liml=ths[-1]-abs(x_limh-ths[-1])
    else:
        x_liml=x_min-(np.max(ths)-np.min(ths))/100*5
        x_limh=ths[-1]+abs(ths[-1]-x_liml)
    
    if ax is None:
        ax = setup_plot(x_liml, x_limh, y_min, y_max,True,True,"Gradient descent","x","f(x)")

    x_func=np.linspace(x_liml,x_limh,100)
    pts=np.array([f(i) for i in x_func])
    ax.plot(x_func,pts,"b-", lw=1)
    ax.plot(ths,fs,"r-",lw=1)
    return

def plot_data(data, labels, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    """
    Make scatter plot of data.
    data = (numpy array)
    ax = (matplotlib plot)
    clear = (bool) clear current plot first
    xmin, xmax, ymin, ymax = (float) plot extents
    returns matplotlib plot on ax 
    """
    if ax is None:
        if xmin == None: xmin = np.min(data[:, 0]) - 0.5
        if xmax == None: xmax = np.max(data[:, 0]) + 0.5
        if ymin == None: ymin = np.min(data[:, 1]) - 0.5
        if ymax == None: ymax = np.max(data[:, 1]) + 0.5
        ax = setup_plot(xmin, xmax, ymin, ymax)

    elif clear:
        ax.clear()
    colors=["green" if yi>0 else "red" for yi in labels]
    ax.scatter(data[:,0], data[:,1], c = colors, marker="x")
    return ax

def plot_separator(ax, th, th0=0, x=None):
    """
    Plot separator in 2D
    ax = (matplotlib plot) plot axis
    th = (numpy array) theta
    th_0 = (float) theta_0
    x = array of datapoints (for choosing plot limit)
    """
    x_min=np.min(x[:,0])
    x_max=np.max(x[:,0])
    y_min=np.min(x[:,1])
    y_max=np.max(x[:,1])
    #hyperplane
    if abs(th[0])>1e-6:
        if -(y_min*th[1]+th0)/th[0]>x_min:
            x_liml=-(y_min*th[1]+th0)/th[0]
        else:
            x_liml=x_min
        if -(y_max*th[1]+th0)/th[0]<x_max:
            x_limh=-(y_max*th[1]+th0)/th[0]
        else:
            x_limh=x_max
    else:
        x_liml=x_min
        x_limh=x_max

    if abs(th[1])>1e-6:
        d=(x_limh-x_liml)*0.05
        xhp=np.linspace(x_liml-d,x_limh+d,100)
        pts=np.array([[x1,(-th0-th[0]*x1)/th[1]] for x1 in xhp])
    else:
        xhp=-th0/th[0]
        pts=np.array([[xhp,x2] for x2 in (y_min,y_max)])
    ax.plot(pts[:,0],pts[:,1],"b-", lw=1)
    #normal vector
    midp=pts[len(pts)//2]
    endp=midp+th
    ax.plot([midp[0],endp[0]],[midp[1],endp[1]],"m-", lw=1)
    return

def plot_obj(fs, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    ts=np.arange(fs.size)
    if ax is None:
        if xmin == None: xmin = np.min(ts) - 0.5
        if xmax == None: xmax = np.max(ts) + 0.5
        if ymin == None: ymin = np.min(fs) - 0.5
        if ymax == None: ymax = np.max(fs) + 0.5
        ax = setup_plot(xmin, xmax, ymin, ymax, grid=True, 
                        title="The objective function over the iterations", xlabel="iterations [1]", ylabel="J [1]")

    elif clear:
        ax.clear()
    ax.plot(ts, fs)
    return ax