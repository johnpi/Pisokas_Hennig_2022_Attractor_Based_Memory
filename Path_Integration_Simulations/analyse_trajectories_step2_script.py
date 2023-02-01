#
# DESCRIPTION
# Reads the filename_input_npz and calculates the Mean Squared Errors (MSE) between
# the values in the file and the actual ants behaviour.
# The resulting MSE values are stored in the file filename_results_npz
#

import os
from pathlib import Path
import numpy as np
from numpy import percentile
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection # For multi-coloured lines by time
from matplotlib.colors import ListedColormap, BoundaryNorm # For multi-coloured lines by time
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
import glob
import scipy.stats
from scipy.stats import *
from scipy.signal import savgol_filter, find_peaks
import astropy.stats
from scipy.optimize import curve_fit
from IPython.display import display, HTML # For displaying pandas tables
import re

# Data files
# Load the data (search_dispersion_list dict) from previous run
# The data in this file have the structure dict[wait_noise_sd_str][mem_Nl_str][mem_r_str][measure] = [list of values]
filename_input_npz = 'path-integration-forget/data/3_parameters_results_correct_Nl_range_combined_01Ded2022/path_analysis_calculation_results_3parameters.npz'
#filename_input_npz = 'path-integration-forget/data/3_parameters_results_extra_search_space_05Ded2022/path_analysis_calculation_results_3parameters.npz'

# Store resulting MSE values to this file
# The data in this file will have the structure dict[measure][wait_noise_sd_str][mem_Nl_str][mem_r_str] = MSE value
filename_results_npz = 'path-integration-forget/data/3_parameters_results_correct_Nl_range_combined_01Ded2022/path_analysis_calculation_results_3parameters_MSE_values.npz'
#filename_results_npz = 'path-integration-forget/data/3_parameters_results_extra_search_space_05Ded2022/path_analysis_calculation_results_3parameters_MSE_values.npz'

measure = ''
mem_wait_list = [0, 1, 24, 48, 96, 144, 192, 240, 288, 336, 384, 432]
mem_noise_list = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02]
#mem_noise_list = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.009, 0.01, 0.015, 0.02]

# First data collection: collected using wrong Nl range: data in 3_params_scanning_wrong_Nl_range/
mem_Nl_list = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
mem_r_list = [-0.008, -0.010, -0.012, -0.014, -0.016, -0.018, -0.020, -0.022, -0.024, -0.026, -0.028, -0.030, -0.032]

# New data collection: collected using the correct Nl range: data in 3_params_scanning_correct_Nl_range/
mem_Nl_list = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
mem_r_list = [-0.008, -0.010, -0.012, -0.014, -0.015, -0.016, -0.017, -0.018, -0.019, -0.020, -0.021, -0.022, -0.023, -0.024, -0.025, -0.026, -0.027, -0.028, -0.029, -0.030, -0.031, -0.032]

# Newest data collection: combining the last and additional collected data: data in 3_params_scanning_correct_Nl_range/
mem_Nl_list =  [0.0, 0.001, 0.01, 0.016, 0.018, 0.019, 0.02, 0.021, 0.022, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.5, 0.999, 1.3] # 25
mem_r_list = [-0.008, -0.01, -0.012, -0.014, -0.015, -0.016, -0.017, -0.018, -0.019, -0.02, -0.021, -0.022, -0.023, -0.024, -0.025, -0.026, -0.027, -0.028, -0.029, -0.03, -0.031, -0.032, -0.035, -0.04, -0.045, -0.05, -0.1] # 27

# Extra data collection: explore additional region of the search space
#mem_Nl_list =  [0.0, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
#mem_r_list =   [-0.033, -0.034, -0.035, -0.036, -0.037, -0.038, -0.039, -0.04, -0.041, -0.042]

slice_t_max = 8


# Camera recordings frame rate
fps = 25 # in frames per sec

# Plot styling
column_single = 89.0 / 25.4
column_double = 183.0 / 25.4
column_full = 247.0 / 25.4
column_80mm = 80 / 25.4
column_174mm = 174 / 25.4
column_210mm = 8.2
figsize = (3.8, 2.6)
figsize = (2.3, 1.6)

# Colors
line_props_full_traj     = dict(color='#444444', alpha=1.0)
point_props_nest         = dict(color='darkorange', alpha=1.0)
line_props_platform_traj = dict(color='#444444', alpha=1.0)
arrow_props_mean_traj    = dict(color='#444444', linewidth=2, alpha=1.0)
arrow_props_mean_all     = dict(color='darkorange', linewidth=2, alpha=1.0)
arrow_props_mean_all     = dict(color='darkorange', alpha=0.7)
time_props_errorbar      = dict(color='#444444', alpha=1.0)
stats_props_errorbar_1   = dict(color='#444444', alpha=1.0)
stats_props_errorbar_2   = dict(color='#444444', alpha=1.0)
stats_props_errorbar_3   = dict(color='#444444', alpha=1.0)
stats_props_errorbar_4   = dict(color='#444444', alpha=1.0)
stats_props_errorbar_5   = dict(color='#444444', alpha=1.0)
stats_props_errorbar_6   = dict(color='#444444', alpha=1.0)
stats_props_errorbar_7   = dict(color='#444444', alpha=1.0)

plot_style = {
    "font.family": "Arial",     # specify font family here
    "font.size"  : 10,
    "axes.spines.top"    : False, 
    "axes.spines.right"  : False, 
    "xtick.direction"     : "out",
    "ytick.direction"     : "out",
    "xtick.color"         : "black",
    "ytick.color"         : "black"
}
plt.rcParams.update(plot_style) # Update the style


# Some useful functions
def cart2pol(x, y):
    """ 
    Convert from Cartesian to polar coordinates.
    Based on https://ocefpaf.github.io/python4oceanographers/blog/2015/02/09/compass/

    Example
    -------
    >>> theta, radius = pol2cart(x, y)
    """
    
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    theta[theta<0] += 2*np.pi
    return theta, radius


def compass(u, v, ax, arrowprops=None):
    """
    compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    """

    angles, radii = cart2pol(u, v)

    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0), arrowprops=kw) for angle, radius in zip(angles, radii)]


def compass_pol(angles, radii, ax, arrowprops=None):
    """
    compass_pol draws a graph that displays the vectors with
    components `angles` and `radii` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> compass(angles, radii)
    """

    kw = dict(arrowstyle="->", color='k')
    if arrowprops:
        kw.update(arrowprops)
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0), arrowprops=kw) for angle, radius in zip(angles, radii)]
    
    ax.set_ylim(0, np.max(radii))

    
def compass_sector_pol(angle_from, angle_to, radius, ax, sectorprops=None):
    """
    Plots a solid sector on a polar graph. The sector start and end angles 
    angle_from and angle_to are in radians. The radius of the sector is radius. 

    Examples
    --------
    >>> import numpy as np
    >>> compass(np.pi-np.pi/36, np.pi+np.pi/36, 1.0)
    """

    kw = dict()
    if sectorprops:
        color = sectorprops.get('color', 'black')
        alpha = sectorprops.get('alpha', 1.0)
    else:
        color = 'black'
        alpha = 1.0

    ax.add_artist(Wedge((.5,.5), radius, np.degrees(angle_from), np.degrees(angle_to), transform=ax.transAxes, color=color, alpha=alpha))
    ax.set_ylim(0, np.max(radius))


def circ_r_alpha(alpha, w=None):
    """ Based on matlab circular statistics toolbox 
        https://uk.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics 
    """
    if w is None:
        w = np.ones(alpha.shape)
    
    x_total = np.sum(np.cos(alpha) * w)
    y_total = np.sum(np.sin(alpha) * w)
    r = np.sqrt(x_total**2 + y_total**2)
    return r / np.sum(w)


def circ_r_xy(x, y, w=None):
    """ Based on matlab circular statistics toolbox 
        https://uk.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics 
    """
    if w is None:
        w = np.ones(x.shape)
    
    x_total = np.sum(x * w)
    y_total = np.sum(y * w)
    r = np.sqrt(x_total**2 + y_total**2)
    return r / np.sum(w)

# This is wrong delete
def circ_mean_vector_v_alt(x, y):
    """ 
        My alternative version of calculating v without needing the 
        trajectory to be rediscretized. Redisretized trajectories 
        (equal step lengths) are a requirement of standard methods. 
    """    
    dx = np.diff(x)
    dy = np.diff(y)

    dx_total = np.sum(dx)
    dy_total = np.sum(dy)
    r = np.sqrt(dx_total**2 + dy_total**2)

    dh = np.hypot(dx, dy)
    path_length = np.sum(dh)

    # Mean vector length
    v = r / path_length
    
    # The angle of the vector
    dx = x.iloc[-1] - x[0]
    dy = y.iloc[-1] - y[0]
    theta = np.arctan2(dy, dx) # The angle of the vector
    if theta < 0:
        theta += 2*np.pi # Convert all angles to be only positive 0 to 2*pi

    return (v, theta)

def circ_mean_vector_v(x, y, rediscretization_step=0.01, ref_dir=None):
    """ 
        Calculates the mean vector v after rediscretizing the
        trajectory (equal step lengths) according to o Batschelet (1981). 
    """    
    trj = pd.DataFrame(data = {'x': x, 'y': y})
    
    # Redisretize the trajectory to make all segments equal in length
    resampled = rediscretize_points(trj, rediscretization_step)
    
    # Calculate the mean vector length and angle
    r, theta = TrajMeanVectorOfTurningAngles(resampled, ref_dir=ref_dir)

    if theta < 0:
        theta += 2*np.pi # Convert all angles to be only positive 0 to 2*pi

    return (r, theta)

def reject_outliers(data):
    """ Rejects outliers from a dataset """
    
    # calculate interquartile range
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25

    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    outliers = [x for x in data if x < lower or x > upper]

    # remove outliers
    without_outliers = [x for x in data if x >= lower and x <= upper]
    
    return (without_outliers, outliers)


def circ_stat_tests(data, print_data=True):
    """ Calculates statistical significance of circular data between multiple conditions.
        data: a dictionary with each entry being the measurements in radians for each condition as 1D arrays. 
        The keys of the dict are the names of the conditions. 
        Returns the stats in a dictionary with structure stat_test_dict[<TEST>][<COND_ROW>][<COND_COLUMN>]
    """

    # The conditions
    conditions = list(data.keys())

    if print_data: 
        print()
        header_string = ' {:^10} ' * (len(conditions)+1)
        row_string = ' {:^10} ' + ' {:^10.3f} ' * len(conditions)
        print(header_string.format(*([' '] + conditions)))

    # Test Circular Distribution Uniformity
    # The stats to apply
    stat_test_str = ['V test']
    stat_test = [astropy.stats.circstats.vtest]

    # Initialise uniformity stats dict
    # uniformity_test_dict[<TEST>][<COLUMN>]
    uniformity_test_dict = {}
    
    # For each statistic test
    for i, test in enumerate(stat_test):
        uniformity_test_dict[stat_test_str[i]] = {}
        
        list_p = [] # p-values list
        list_n = [] # sample size n list
        
        # Try for each condition
        for condition_i in conditions:
            if test == astropy.stats.circstats.vtest:
                p = test(data[condition_i], mu = np.radians(100.0))
            else:    
                p = test(data[condition_i])
            
            # Store p-value
            uniformity_test_dict[stat_test_str[i]][condition_i] = p
            list_p.append(p) # For printing
            list_n.append(len(data[condition_i])) # For printing
            
        if print_data: 
            print(row_string.format(*([stat_test_str[i]] + list_p)))
            print(row_string.format(*([stat_test_str[i] + '(n)'] + list_n)))
            
    return uniformity_test_dict
            

def stat_tests(data, print_data=True):
    """ Calculates statistical significance between multiple conditions.
        data: a dictionary with each entry being the measurements for each condition as 1D arrays. 
        The keys of the dict are the names of the conditions. 
        Returns the stats in a dictionary with structure stat_test_dict[<TEST>][<COND_ROW>][<COND_COLUMN>]
    """
    
    # The stats to apply
    stat_test_str = ['Mann-Whitney U']
    stat_test = [mannwhitneyu]
    # From ranksums doc: It does not handle ties between measurements in x and y. For tie-handling and an optional continuity correction see scipy.stats.mannwhitneyu.
    
    # The conditions
    conditions = list(data.keys())
    
    # Initialise stats dict
    # stat_test_dict[<TEST>][<ROW>][<COLUMN>]
    stat_test_dict = {}
    
    # For each statistic test
    for i, test in enumerate(stat_test):
        stat_test_dict[stat_test_str[i]] = {}
        if print_data: 
            print()
            print(stat_test_str[i])
        if print_data: 
            header_string = ' {:^10} ' * (len(conditions)+1)
            row_string = ' {:^10} ' + ' {:^10.3f} ' * len(conditions)
            print(header_string.format(*([' '] + conditions)))
        
        # Try all combinations of conditions
        for condition_j in conditions:
            list_p = [] # p-values list
            stat_test_dict[stat_test_str[i]][condition_j] = {}
            for condition_i in conditions:
                if test != wilcoxon: # Wilcoxon test needs special treatment
                    stat, p = test(data[condition_j], data[condition_i])
                    #print('stat={:3f}, p={:3f}'.format(stat, p))
                else:
                    if condition_j != condition_i:
                        keep = min(len(data[condition_j]), len(data[condition_i]))
                        stat, p = test(data[condition_j][:keep], data[condition_i][:keep])
                        #print('stat={:3f}, p={:3f}'.format(stat, p))
                    else:
                        p = 1.0
                
                # Store p-value
                stat_test_dict[stat_test_str[i]][condition_j][condition_i] = p
                list_p.append(p) # For printing
            
            if print_data: 
                print(row_string.format(*([condition_j] + list_p)))
        
    return stat_test_dict


def stars(p):
    """ For returning stars string according to p value """
    
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"


def split_exp_condition_column(lst):
    """ 
    Splits the strings in a list into two lists one with only the 
    experimental condition part (string) and the second the numerical 
    value if exists or empty string otherwise. Used for processing
    the file names. 
    """
    
    r=re.compile(r"(\D+)([\d+.]*)")
    res = []
    for item in lst:
        m = r.match(item)
        res.append([m.group(1), m.group(2)])
    exp_cond, exp_val = zip(*res)
    exp_val = list(exp_val)
    for i,v in enumerate(exp_val):
        try: 
            exp_val[i] = float(v)
        except:
            exp_val[i] = None
    return exp_cond, exp_val


def draw_sizebar(ax, size=1.0, label='1m', location='lower center', label_top=False):
    """
    Draws a horizontal scale bar with length size in plot coordinates,
    with a scale label underneath.
    """
    
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    asb = AnchoredSizeBar(ax.transData,           # transform, 
                          size,                   # size, 
                          label,                  # label, 
                          loc=location,           # loc
                          label_top=label_top,    # 
                          pad=0.1, borderpad=0.5, sep=5, 
                          color = 'black', 
                          frameon=False)
    ax.add_artist(asb)


def adjust_spines(ax, spines, x_values=None, y_values=None, margin=10):
    """ 
    Adjusts the appearance of axis spines on a plot. 
    The spines will not touch at the corner and the ticks will be outward. 
        spines: a list of strings specifying the spines to keep.
        x_values: the min and max range of the plot. Needed when ticks do nto appear at the ends of the spines.
        y_values: the min and max range of the plot. Needed when ticks do nto appear at the ends of the spines.
    """
    
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', margin)) # outward by 10 points
            #spine.set_smart_bounds(True)
            if (loc=='bottom' or loc=='top') and x_values is not None:
                spine.set_bounds(x_values.min(), x_values.max())
            if (loc=='left' or loc=='right') and y_values is not None:
                spine.set_bounds(y_values.min(), y_values.max())
        else:
            spine.set_visible(False) # don't draw spine


def get_start_end_of_path(filename):
    """ Returns the coordinates of the beginning and the end of a path """
    
    filename_csv = filename.replace('.npz', '.csv')            
    pd_i = pd.read_csv(filename_csv)

    x = pd_i["x"]
    y = pd_i["y"]
    
    start = (x.iloc[0], y.iloc[0])
    end = (x.iloc[-1], y.iloc[-1])
    
    return (start, end)

def get_center_of_path(x, y, beyond=2):
    """ Estimates the x,y coordinates of the center of the ant search pattern. """
    # Get point of moving <beyond> distance away from start
    h = np.hypot(x-x[0], y-y[0])
    beyond_crossing = np.argmax(h > beyond)
    # print(beyond_crossing)
    
    # Get the median location boyond that point
    x_med = np.median(x[beyond_crossing:])
    y_med = np.median(y[beyond_crossing:])
    
    return (x_med, y_med)

def spread_of_2D_points(x_list, y_list):
    # Calculate the dispersion of x,y points
    x_med = np.median(x_list)
    y_med = np.median(y_list)
    h = np.hypot(x_list-x_med, y_list-y_med)
    dispersion = np.median(h)
    return dispersion

def homing_distance_spread(x_list, y_list):
    """ Spread of homing distance """    
    h = np.hypot(x_list, y_list)
    distance_median = np.median(h)
    distance_std = np.std(h)
    return (distance_median, distance_std, h)

def homing_heading_spread(x_list, y_list):
    """ Spread of homing distance """    
    h = np.hypot(x_list, y_list)
    angles = np.arctan2(y_list, x_list)
    angles[angles<0] += 2*np.pi

    heading_mean = scipy.stats.circmean(angles)
    heading_std = scipy.stats.circstd(angles)
    #heading_std = astropy.stats.circstats.circstd(angles)
    
    return (heading_mean, heading_std)


def calc_angle(trj, ref_dir=None):
    """ Turning angles of a Trajectory
        Calculates the step angles (in radians) of each segment, either relative to 
        the previous segment or relative to the specified compass direction.
        The turning angle before and after every zero-length segment will be NaN, 
        since the angle of a zero-length segment is undefined.
    Args:
       trj: The trajectory whose angles to calculate.
       ref_dir: If not None, step angles are calculated relative to this angle (in radians), 
       otherwise they are calculated relative to the previous step angle.
       unit (str): return angle in radians or degrees (Default value: 'degrees')
       lag (int) : time steps between angle calculation (Default value: 1)
    Returns:
      angle: The angles between steps in radians, normalised to -pi<angle<=pi. 
      If ref_dir is None (the default), the returned array will have length len(trj)-2, 
      i.e. one angle for every pair of adjacent segments. If ref_dir is not None, the 
      returned array will have length len(trj)-1, i.e. one angle for every segment.
      Based on trajr TrajAngles function.
    """
    dx = np.diff(trj['x'])
    dy = np.diff(trj['y'])
    if ref_dir is None:
        angles = np.diff(np.arctan2(dy, dx))
    else:
        angles = np.arctan2(dy, dx) - ref_dir
    
    # Normalise to +/-0-360deg
    angles = angles % (2*np.pi)
    
    # Normalise to -pi<angle<=pi
    angles[angles<=-np.pi] += 2*np.pi
    angles[angles>np.pi]   -= 2*np.pi

    return angles


def TrajMeanVectorOfTurningAngles(trj, ref_dir=None):
    # Angular changes
    angles = calc_angle(trj, ref_dir=ref_dir)
    
    # Mean vector
    # The value as defined in Batschelet
    phi = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

    r = np.sqrt(np.sum(np.cos(angles))**2 + np.sum(np.sin(angles))**2) / len(angles)
    # complex(modulus = r, argument = phi)
    return (r, phi)


def rediscretize_points(trj, R, time_out=False):
    """Returns a ``TrajaDataFrame`` rediscretized to a constant step length `R`.
    Args:
      trj (:class:`traja.frame.TrajaDataFrame`): Trajectory
      R (float): Rediscretized step length (eg, 0.02)
      time_out (bool): Include time corresponding to time intervals in output
    Returns:
      rt (:class:`numpy.ndarray`): rediscretized trajectory
    """
    if not isinstance(R, (float, int)):
        raise TypeError(f"R should be float or int, but is {type(R)}")

    results = _rediscretize_points(trj, R, time_out)
    rt = {}
    rt['x'] = results["rt"][:,0]
    rt['y'] = results["rt"][:,1]
    if len(rt) < 2:
        raise RuntimeError(
            f"Step length {R} is too large for path (path length {len(trj)})"
        )

    if time_out:
        rt["time"] = results["time"]
    return rt

def _rediscretize_points(
    trj, R, time_out=False
):
    """Helper function for :func:`traja.trajectory.rediscretize`.
    Args:
      trj (:class:`traja.frame.TrajaDataFrame`): Trajectory
      R (float): Rediscretized step length (eg, 0.02)
    Returns:
      output (dict): Containing:
        result (:class:`numpy.ndarray`): Rediscretized coordinates
        time_vals (optional, list of floats or datetimes): Time points corresponding to result
    """
    # TODO: Implement with complex numbers
    points = trj[["x", "y"]].dropna().values.astype("float64")
    n_points = len(points)
    result = np.empty((128, 2))
    p0 = points[0]
    result[0] = p0
    step_nr = 0
    candidate_start = 1  # running index of candidate

    time_vals = []
    if time_out:
        time_col = _get_time_col(trj)
        time = trj[time_col][0]
        time_vals.append(time)

    while candidate_start <= n_points:
        # Find the first point `curr_ind` for which |points[curr_ind] - p_0| >= R
        curr_ind = np.NaN
        for i in range(
            candidate_start, n_points
        ):  # range of search space for next point
            d = np.linalg.norm(points[i] - result[step_nr])
            if d >= R:
                curr_ind = i  # curr_ind is in [candidate, n_points)
                if time_out:
                    time = trj[time_col][i]
                    time_vals.append(time)
                break
        if np.isnan(curr_ind):
            # End of path
            break

        # The next point may lie on the same segment
        candidate_start = curr_ind

        # The next point lies on the segment p[k-1], p[k]
        curr_result_x = result[step_nr][0]
        prev_x = points[curr_ind - 1, 0]
        curr_result_y = result[step_nr][1]
        prev_y = points[curr_ind - 1, 1]

        # a = 1 if points[k, 0] <= xk_1 else 0
        lambda_ = np.arctan2(
            points[curr_ind, 1] - prev_y, points[curr_ind, 0] - prev_x
        )  # angle
        cos_l = np.cos(lambda_)
        sin_l = np.sin(lambda_)
        U = (curr_result_x - prev_x) * cos_l + (curr_result_y - prev_y) * sin_l
        V = (curr_result_y - prev_y) * cos_l - (curr_result_x - prev_x) * sin_l

        # Compute distance H between (X_{i+1}, Y_{i+1}) and (x_{k-1}, y_{k-1})
        H = U + np.sqrt(abs(R ** 2 - V ** 2))
        XIp1 = H * cos_l + prev_x
        YIp1 = H * sin_l + prev_y

        # Increase array size progressively to make the code run (significantly) faster
        if len(result) <= step_nr + 1:
            result = np.concatenate((result, np.empty_like(result)))

        # Save the point
        result[step_nr + 1] = np.array([XIp1, YIp1])
        step_nr += 1

    # Truncate result
    result = result[: step_nr + 1]
    output = {"rt": result}
    if time_out:
        output["time"] = time_vals
    return output


# Map between strings and printed labels
measures_labels = {
    'Emax_a': 'Emax$_a$', 
    'Emax_b': 'Emax$_b$',
    'Emax_b_100deg': 'Emax$_b$', 
    'TrajNestNearest': 'Nearest distance (m)', 
    'Emax_a_unrediscretised': 'Emax_a_unrediscretised', 
    'Emax_b_unrediscretised': 'Emax_b_unrediscretised', 
    'EDiffusionDistance': 'Diffusion distance (m)', 
    'ESqDiffusionDistance': 'Squared diffusion distance', 
    'TrajExpDrift': 'Expected drift', 
    'TrajExpDriftDist': 'Expected drift distance', 
    'mean_speed': 'mean_speed', 
    'min_C': 'min_C', 
    'min_deltaS': 'min_deltaS', 
    'sd_speed': 'sd_speed', 
    'sinuosity': 'Sinuosity', 
    'sinuosity_redisc': 'Sinuosity', 
    'sinuosity_smooth_redisc': 'Sinuosity_smooth_redisc', 
    'straightness_DL_ratio': 'Straightness_DL_ratio', 
    'straightness_r': 'Straightness (r)'
}


save_figures = False # True
save_figures_as_filetype = '.svg' # '.pdf'
images_path = 'images/'
if save_figures:
    Path(images_path).mkdir(parents=True, exist_ok=True)

np.set_printoptions(threshold=np.inf)


distance_scaling_factor = 3
#distance_scaling_factor = 2.87 # Corrected scaling factor that results in median homing distance of 12.79m for 0h of captivity time as reported in Ziegler1997. With factor x3 we get mean homing distance of 13.349528374417343, so the correct factor should be 3*12.7911162/13.349528374417343=2.874509684816851

# This is the file with the outbound path used for the collected data in case we want to plot it
outbound_path_filename = 'path-integration-forget/data/outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz'


def get_outbound_path_details(outbound_path_filename): # Path to the file that was used in the simulations as the outbound path
    """ Returns the given path's start and end coordinates,
    the start to end distance, the start to end direction,
    and the end to start direction. """
    # Get the path beginning and end points
    path_start, path_end = get_start_end_of_path(outbound_path_filename)
    # The outbound distance
    outbound_distance = np.hypot(path_end[0]-path_start[0], path_end[1]-path_start[1])*distance_scaling_factor
    print('Outbound distance is {:}'.format(outbound_distance))

    # The outbound direction calculated end-to-end
    outbound_traj_vector = pd.DataFrame({'x': [0, path_end[0]-path_start[0]], 'y': [0, path_end[1]-path_start[1]]})
    outbound_angle = calc_angle(outbound_traj_vector, ref_dir=0)
    print('Outbound angle is {:}'.format(np.degrees(outbound_angle[0])))

    # The corresponding inbound direction (release point to fictive nest) is outbound_traj_vector+180deg
    homing_ref_angle = (outbound_angle + np.pi) % (2*np.pi)
    
    return (path_start, path_end, outbound_distance, outbound_angle, homing_ref_angle)


# Load the data (search_dispersion_list dict) from previous run
results_dict = np.load(filename_input_npz, allow_pickle=True)['arr_0'][()]


# Get the outbound path beginning and end points, distance, direction and reverse direction
path_start, path_end, outbound_distance, outbound_angle, homing_ref_direction = get_outbound_path_details(outbound_path_filename)


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def fit_data_points(ax, xdata, ydata, func, func_str='', init_vals=None, color=None):
    xvalues = np.linspace(np.min(xdata), np.max(xdata))
    popt, pcov = curve_fit(func, xdata, ydata, p0=init_vals)
    yvalues = func(xvalues, *popt)
    ax.plot(xvalues, yvalues, '-', color=color)
    print('The fitted curve parameters ' + func_str + ' are:', *popt)
    print('R^2 = {}'.format(r2_score(ydata, func(np.array(xdata), *popt))))


# Import functions to use for regression to data
from regression_functions import *


func_sigmoid_logistic_modified_2_str = 'a / (1 + k * np.exp(r*x))'
def func_sigmoid_logistic_modified_2(x, a, r, k):
    """ 
        Parameters:
            x0 : the centre of the function along the x axis
            k  : the slope of the logistic function
            a  : the maximum value
    """
    return a / (1 + k * np.exp(r*x))



def data_analysis(measure,                                                     # Dict key of measure data points to use 
                  mem_noise_list,                                              # List of memory diffusion coefficients
                  mem_Nl_list,                                                 # List of memory logistic decay Nl parameters
                  mem_r_list,                                                  # List of memory logistic decay r parameters
                  slice_t_max,                                                 # Use only the first slice_t_max wait_t times, use -1 to include all wait_t times, use 8 to include up to 240h
                  square                  = False,                             # Raise the y values to the power or 2
                  inverse                 = False,                             # Plot the 1 / y of the data
                  normalise               = False,                             # Normalise data by plotting the y/max(y)
                  func_target_values      = None                               # Function for producing target y values
                  ):
    
    MSE_dic = {} # To store the MSE values
    
    # Calculate the MSE from target ant behaviour measure for each combination of parameters
    for i,wait_noise_sd in enumerate(mem_noise_list):
        wait_noise_sd_str = str(wait_noise_sd)    
        MSE_dic[wait_noise_sd_str] = {}
        for mem_Nl in mem_Nl_list:
            mem_Nl_str = str(mem_Nl)
            MSE_dic[wait_noise_sd_str][mem_Nl_str] = {}
            for mem_r in mem_r_list:
                mem_r_str = str(mem_r)
                MSE_dic[wait_noise_sd_str][mem_Nl_str][mem_r_str] = None
                
                print('LogisticLoss(t) + Difussive Noise: wait_noise_sd={:<6.3}, mem_Nl={:<6.3}, mem_r={:<6.3},'.format(wait_noise_sd, mem_Nl, mem_r), end='')
                
                # Get the waiting times of the data points
                wait_t = results_dict["wait_hours"][0:slice_t_max]
                # Get the data points
                y = results_dict[wait_noise_sd_str][mem_Nl_str][mem_r_str][measure][0:slice_t_max]
                # Square data points
                if square:
                    y = np.array(y)**2
                # Inverse data points
                if inverse:
                    y = 1 / np.array(y)
                # Normalise data points
                if normalise:
                    y = y / np.max(y)
                
                if not func_target_values:
                    raise Exception("func_target_values formal parameter is required.")
                
                # Get the target values
                y_pred = func_target_values(np.array(wait_t))
                
                # The meas squared error
                MSE = mean_squared_error(y, y_pred)
                MSE_dic[wait_noise_sd_str][mem_Nl_str][mem_r_str] = MSE
                print(' MSE={:<6.3}'.format(MSE))
    
    return MSE_dic


# Import models of ant behaviour measures
# =======================================
from ant_behaviour_measures_models import *
# Imports the variables and functions:
#  PopulationLoss_str = 'K / (1 + Nl/(K - Nl) * np.exp(-r * t))'
#  PopulationLoss(t, K, Nl, r)
# Plot the ant median homing distance vs waiting time
#  predict_ant_homing_distance(wait_t)
#  plot_ant_homing_distance(ax, wait_t, color='grey')
# Plot the ant homing distance error dispersion vs waiting time
#  plot_ant_homing_distance_error_MAD(ax, wait_t, color='grey')
#  predict_ant_homing_distance_error_MAD(x_1)
# Plot the ant homing distance error dispersion squared vs waiting time
#  plot_ant_homing_distance_error_MAD_squared(ax, wait_t, color='grey')
#  predict_ant_homing_distance_error_MAD_squared(x_1)
# Plot the ant homing distance accuracy vs waiting time
#  plot_ant_homing_distance_accuracy_1_over_MAD(ax, wait_t, color='grey')
#  predict_ant_homing_distance_accuracy_1_over_MAD(x_1)
# Plot homing ant direction accuracy vs waiting time
#  plot_ant_homing_angle_accuracy_1_over_sigma_squared(ax, wait_t, color='grey')
#  predict_ant_homing_angle_accuracy_1_over_sigma_squared(x_1)



MSE_results_dic = {}
for key,measure,func_target_values,square,inverse,normalise in [
	# Key to use in the dict           Key of dict in the file  Function producing target y sqred invrs norm
    ("Homing_Distance_Median_tocntr", "Distance_Median_tocntr", predict_ant_homing_distance,False,False,False), 
    ("Homing_Distance_Median_toturn", "Distance_Median_toturn", predict_ant_homing_distance,False,False,False), 
    ("Distance_to_Nest_Dispersion_MAD_tocntr", "Distance_to_Nest_Dispersion_tocntr",predict_ant_homing_distance_error_MAD,False,False,False), 
    ("Distance_to_Nest_Dispersion_MAD_toturn", "Distance_to_Nest_Dispersion_toturn",predict_ant_homing_distance_error_MAD,False,False,False), 
    ("Distance_to_Nest_Dispersion_MAD2_tocntr", "Distance_to_Nest_Dispersion_tocntr",predict_ant_homing_distance_error_MAD_squared,True,False,False), 
    ("Distance_to_Nest_Dispersion_MAD2_toturn", "Distance_to_Nest_Dispersion_toturn",predict_ant_homing_distance_error_MAD_squared,True,False,False), 
    ("Distance_to_Nest_Accuracy_tocntr","Distance_to_Nest_Dispersion_tocntr",predict_ant_homing_distance_accuracy_1_over_MAD,False,True,True), 
    ("Distance_to_Nest_Accuracy_toturn","Distance_to_Nest_Dispersion_toturn",predict_ant_homing_distance_accuracy_1_over_MAD,False,True,True), 
    ("Homing_Angle_Accuracy_tocntr","Angle_Dispersion_tocntr",predict_ant_homing_angle_accuracy_1_over_sigma_squared,False,True,True),
    ("Homing_Angle_Accuracy_toturn","Angle_Dispersion_toturn",predict_ant_homing_angle_accuracy_1_over_sigma_squared,False,True,True)]:
    MSE_results_dic[key] = data_analysis(measure,                       # Dict key of measure data points to use 
                           mem_noise_list,                              # List of memory diffusion coefficients
                           mem_Nl_list,                                 # List of memory logistic decay Nl parameters
                           mem_r_list,                                  # List of memory logistic decay r parameters
                           slice_t_max             = slice_t_max,       # Use only the first slice_t_max wait_t times, use -1 to include all wait_t times, use 8 to include up to 240h
                           square                  = square,            # Raise the y values to the power or 2
                           inverse                 = inverse,           # Plot the 1 / y of the data
                           normalise               = normalise,         # Normalise data by plotting the y/max(y)
                           func_target_values      = func_target_values # Function for producing target y values
                           )

# Store resulting MSE values to file
# The data in this file will have the structure dict[measure][wait_noise_sd_str][mem_Nl_str][mem_r_str] = MSE value
np.savez(filename_results_npz, MSE_results_dic)
