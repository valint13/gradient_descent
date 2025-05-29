# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:42:16 2025

@author: User
"""
import numpy as np
import pandas as pd
import src.utilities as util
#%%
data = util.prep_data('D:\Spyder\gradient descent\MIT 6.036 Lab 5 - auto-mpg-regression.tsv')
n, d = data.shape

# converting to arrays
mpg = data["mpg"].to_numpy().reshape(-1,1)
cylinders = data["cylinders"].to_numpy().reshape(-1,1)
cylinders_oh = util.one_hot(cylinders)
displacement = data["displacement"].to_numpy().reshape(-1,1)
horsepower = data["horsepower"].to_numpy().reshape(-1,1)
weight = data["weight"].to_numpy().reshape(-1,1)
acceleration = data["acceleration"].to_numpy().reshape(-1,1)
origin = data["origin"].to_numpy().reshape(-1,1)
origin_oh = util.one_hot(origin)

# Feature set 1
target1 = mpg
features1 = np.hstack((cylinders, displacement, horsepower, weight, acceleration, origin_oh))
feature_index_map1 = {0: "cylinders", 1: "displacement", 2: "horsepower", 3: "weight", 
                      4: "acceleration", 5: "origin_oh_0", 6: "origin_oh_1", 7: "origin_oh_2"}
feature_config1 = {"features": ["cylinders", "displacement", "horsepower", "weight", "acceleration", 
                                "origin_oh_0", "origin_oh_1", "origin_oh_2"], 
                  "standardize": ["cylinders", "displacement", "horsepower", "weight", "acceleration"],
                  "polynomial": ["cylinders", "weight"], 
                  "map": feature_index_map1} # use list comprehension for features

# Feature set 2
target2 = mpg
features2 = np.hstack((cylinders_oh, displacement, horsepower, weight, acceleration, origin_oh))
feature_index_map2 = {0: "cylinders_oh_0", 1: "cylinders_oh_1", 2: "cylinders_oh_2", 3: "cylinders_oh_3", 
                      4: "cylinders_oh_4", 5: "displacement", 6: "horsepower", 7: "weight", 8: "acceleration", 
                      9: "origin_oh_0", 10: "origin_oh_1", 11: "origin_oh_2"}
feature_config2 = {"features": ["cylinders_oh_0", "cylinders_oh_1", "cylinders_oh_2", "cylinders_oh_3", 
                                "cylinders_oh_4", "displacement", "horsepower", "weight", "acceleration", 
                                "origin_oh_0", "origin_oh_1", "origin_oh_2"], 
                  "standardize": ["displacement", "horsepower", "weight", "acceleration"], 
                  "polynomial": [], 
                  "map": feature_index_map2}

