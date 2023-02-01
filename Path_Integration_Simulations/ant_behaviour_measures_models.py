# Models of ant behaviour measures
# ================================

import numpy as np

# Import functions to use for regression to data
from regression_functions import *

# Inverted sigmoid model
PopulationLoss_str = 'K / (1 + Nl/(K - Nl) * np.exp(-r * t))'
# Inverted sigmoid model
def PopulationLoss(t, K, Nl, r):
    return K / (1 + Nl/(K - Nl) * np.exp(-r * t))


# Median ant homing distance vs waiting time
def predict_ant_homing_distance(wait_t):
    K, Nl, r = 12.79111889, 0.09954786, -0.0220668
    y_1 = PopulationLoss(wait_t, K, Nl, r)
    return y_1

# Plot the ant median homing distance vs waiting time
def plot_ant_homing_distance(ax, wait_t, style={'color': 'grey', 'linestyle': '--'}): # wait_t In hours
    x_1 = np.linspace(np.min(wait_t), np.max(wait_t))
    y_1 = predict_ant_homing_distance(x_1)
    ax.plot(x_1, y_1, **style)


# Ant homing distance error dispersion vs waiting time
def plot_ant_homing_distance_error_MAD(ax, wait_t, style={'color': 'grey', 'linestyle': '--'}): # wait_t In hours
    x_1 = np.linspace(np.min(wait_t), np.max(wait_t))
    y_1 = predict_ant_homing_distance_error_MAD(x_1)
    ax.plot(x_1, y_1, **style)

# Plot the ant homing distance error dispersion vs waiting time
def predict_ant_homing_distance_error_MAD(x_1): # x_1 In hours
    y_1 = 0.03422619*x_1 + 1.14285714
    return y_1

# Ant homing distance error dispersion squared vs waiting time
def plot_ant_homing_distance_error_MAD_squared(ax, wait_t, style={'color': 'grey', 'linestyle': '--'}): # wait_t In hours
    x_1 = np.linspace(np.min(wait_t), np.max(wait_t))
    y_1 = predict_ant_homing_distance_error_MAD_squared(x_1)
    ax.plot(x_1, y_1, **style)

# Plot the ant homing distance error dispersion squared vs waiting time
def predict_ant_homing_distance_error_MAD_squared(x_1): # x_1 In hours
    y_1 = 0.34318452*x_1 - 4.78380962
    return y_1

# Ant homing distance accuracy vs waiting time
def plot_ant_homing_distance_accuracy_1_over_MAD(ax, wait_t, style={'color': 'grey', 'linestyle': '--'}):
    x_1 = np.linspace(np.min(wait_t), np.max(wait_t))
    y_1 = predict_ant_homing_distance_accuracy_1_over_MAD(x_1)
    ax.plot(x_1, y_1, **style)

# Plot the ant homing distance accuracy vs waiting time
#def predict_ant_homing_distance_accuracy_1_over_MAD(x_1): # x_1 In hours
#    # This is what Ziegler1995 claims
#    y_1 = np.exp(-0.41*x_1/24)
#
#    # 1 parameter model
#    b = -0.35141012542366507
#    y_1 = func_exp1(x_1/24, b)
#
#    # 2 parameters model
#    b, c = [-0.3659817337644034, 0.009440745668943097] # Optimiser found parameters
#    b, c = [-0.46618463692052803, 0.10822417806285799] # Manually set parameters to the b, c of the 3 parameter model
#    y_1 = func_exp2(x_1/24, b, c)
#    
#    # 3 parameters model This is my better regression model, see: analyse_trajectories_step_2_3params.ipynb
#    #a, b, c = [0.7973826267733032, -0.46618463692052803, 0.10822417806285799]
#    #y_1 = func_exp3(x_1/24, a, b, c)
#    
#    return y_1

# Plot the ant homing distance accuracy vs waiting time
def predict_ant_homing_distance_accuracy_1_over_MAD(x_1): # x_1 In hours
    # This is what Ziegler1995 claims
    y_1 = np.exp(-0.41*x_1/24)

    # 1 parameter model
    b = -0.014642073404910706 # R^2 = 0.9333851761623976
    y_1 = func_exp1(x_1, b)

    # 2 parameters model
    b, c = [-0.015247651588355771, 0.009420153878402254] # R^2 = 0.933544212040003
    y_1 = func_exp2(x_1, b, c)
    
    # 3 parameters model This is my better regression model, see: analyse_trajectories_step_2_3params.ipynb
    #a, b, c = [0.7973826032614079, -0.019424366808324416, 0.10822423002334482] # R^2 = 0.9916380812139618
    #y_1 = func_exp3(x_1, a, b, c)
    
    return y_1


# Ant homing direction accuracy vs waiting time
def plot_ant_homing_angle_accuracy_1_over_sigma_squared(ax, wait_t, style={'color': 'grey', 'linestyle': '--'}):
    x_1 = np.linspace(np.min(wait_t), np.max(wait_t))
    y_1 = predict_ant_homing_angle_accuracy_1_over_sigma_squared(x_1)
    ax.plot(x_1, y_1, **style)

# Plot homing ant direction accuracy vs waiting time
#def predict_ant_homing_angle_accuracy_1_over_sigma_squared(x_1): # x_1 In hours
#    # This is what Ziegler1995 claims
#    y_1 = np.exp(-0.23/24*x_1)
#    
#    # 1 parameter model
#    b = -0.22712638972154658 # R^2 = 0.7165837920105825
#    y_1 = func_exp1(x_1/24, b)
#    
#    # 2 parameters model
#    b, c = [-0.6713943442561833, 0.2008628721736255] # R^2 = 0.7163794674438515
#    y_1 = func_exp2(x_1/24, b, c)
#    
#    # 3 parameters model
#    #a, b, c = [0.6832929060666405, -0.6139620623345058, 0.2763541594080284] # R^2 = 0.9111461097341679
#    #y_1 = func_exp3(x_1/24, a, b, c)
#    
#    return y_1

# Plot homing ant direction accuracy vs waiting time
def predict_ant_homing_angle_accuracy_1_over_sigma_squared(x_1): # x_1 In hours
    # This is what Ziegler1995 claims
    y_1 = np.exp(-0.23/24*x_1)
    
    # 1 parameter model
    b = -0.00946379302436708 # R^2 = 0.7165837920509617
    y_1 = func_exp1(x_1, b)
    
    # 2 parameters model
    b, c = [-0.0279738468097591, 0.20085827280761143] # R^2 = 0.7163794667057858
    y_1 = func_exp2(x_1, b, c)
    
    # 3 parameters model
    #a, b, c = [0.683292934034979, -0.02558181367635523, 0.276354395318332] # R^2 = 0.9111461097421923
    #y_1 = func_exp3(x_1, a, b, c)
    
    return y_1
