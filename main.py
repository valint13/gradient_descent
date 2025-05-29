# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 20:48:42 2025

@author: Bálint
"""

import numpy as np
import src.utilities as util
import src.objectives as obj
import src.optimization as opt
import src.plotting as plot
import src.wrapper as wrp
#%% data
from dataprep import features1, target1, feature_config1, features2, target2, feature_config2
# features1, mus, sigs = util.standardize_featureset(features1, feature_config1)

# x1 = data[]
# y1 = data[]
# data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
# labels = np.array([[1, -1, -1]])
# p1=np.array([data[0,0],data[1,0]])
# p2=np.array([data[0,1],data[1,1]])
# p3=np.array([data[0,2],data[1,2]])

# x1,y1=util.separable_medium()

#%% initialization
# inputs
th_init=np.array([0, 0, 0, 0, 0, 0, 0, 0])
th0_init=0
lam=0.01

config = util.setup_config(features1, target1, th_init, th0_init, lam, 1000, 0.1, "ols", "hom", "dir")
# initialize variables
# th = np.zeros_like(th_init)
# th0 = 0
# create numerical gradient function
# grad_f = obj.num_grad(obj.svm_obj,0.001,homogenity="heterogen",x=x1,y=y1,lam=lam)
# grad_f = obj.num_grad(obj.llc_obj,0.001,homogenity="heterogen",x=x1,y=y1,lam=lam)
# grad_f = obj.num_grad(obj.OLS_obj,0.001,homogenity="heterogen",x=x1,y=y1,lam=lam)
#%% solution
# Cross validation
crossval_results, fold_RMSEs, standardization_params = wrp.cross_validate(features2, target2, 10, config, feature_config2)

# With wrapper function
# results1 = wrp.run(*config)

# Functions directly
# th_hom,ths_hom,fs= opt.gd_homogen(obj.svm_obj,grad_f,th_init,th0_init,eta=0.1,step_size_fn=util.step_svm_min,tmax=100,x=x1,y=y1,lam=lam)
# th,th0,ths,th0s,fs= opt.sgd_heterogen(obj.svm_obj,grad_f,th_init,th0_init,eta=0.1,step_size_fn=util.step_const,tmax=100,x=x1,y=y1,lam=lam)

# th_hom,ths_hom,fs= opt.gd_homogen(obj.llc_obj,obj.llc_grad,th_init,th0_init,eta=0.1,step_size_fn=util.step_const,tmax=1000,x=x1,y=y1,lam=lam)
# th,th0,ths,th0s,fs= opt.gd_heterogen(obj.llc_obj,grad_f,th_init,th0_init,eta=0.1,step_size_fn=util.step_const,tmax=1000,x=x1,y=y1,lam=lam)

# th_hom,ths_hom,fs= opt.gd_homogen(obj.OLS_obj,obj.ridge_grad,th_init,th0_init,eta=0.1,step_size_fn=util.step_const,tmax=100,x=x1,y=y1,lam=lam)
# th,th0,ths,th0s,fs= opt.gd_heterogen(obj.OLS_obj,obj.ridge_grad,th_init,th0_init,eta=0.1,step_size_fn=util.step_const,tmax=1000,x=x1,y=y1,lam=lam)

# th_hom = opt.OLS(x1, y1, lam, "solve")

# th, th0 = util.homogen_th_unpack(th_hom)
# ths, th0s = util.homogen_ths_unpack(ths_hom)
#%% plot
# ax1= plot.setup_plot(min(x1[:,0]),max(x1[:,0]),min(x1[:,1]),max(x1[:,1]),center=True,grid=True,
#                    title=f"Lin sep characterized by th={th},th0={th0},lambda={lam} ",
#                    xlabel="x1",ylabel="x2",aspect="equal")
# ax1= plot.plot_data(x1,y1,ax1)
# plot.plot_separator(ax1,th,th0,x1)
# ax2= plot.plot_obj(fs)

#%% post processing
# training error
# RMSE = util.RMSE_results(features1, target1, results1)
#%% test

