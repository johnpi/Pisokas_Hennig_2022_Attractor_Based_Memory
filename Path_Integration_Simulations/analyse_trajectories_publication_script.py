import os
import sys
import re
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

# For saving files on the big disk on Matthias' server
# if available use the big disk on Matthias' server
if os.path.isdir("/disk/scratch/ipisokas/tmp"):
    os.environ["TMP"] = "/disk/scratch/ipisokas/tmp"
    os.environ["TMPDIR"] = "/disk/scratch/ipisokas/tmp"
    os.environ["TEMPDIR"] = "/disk/scratch/ipisokas/tmp"

# This is the file with the outbound path that was used to collected the data in the csv files
outbound_path_filename = 'path-integration-forget/data/outbound_route_only_S_to_N_1500steps_noiseSyn0.1_noiseRot7.0_noiseSlope9.0_route_Generate_outbound_route_001.npz'


def usage():
    print('Analyses csv files that contain x,y path coordinates and calculates')
    print('descriptive statistics for the paths.')
    print('')
    print('SYNTAX:')
    print('    python analyse_trajectories_publication_script.py <DIRECTORY PATH> <OUTPUT FILE>')
    print('    <DIRECTORY PATH>')
    print('           Path to the csv files to read and process.')
    print('    <OUTPUT FILE>')
    print('           File to store the results in.')
    print('           The data in this file will have the structure: ')
    print('    dict[wait_noise_sd_str][mem_Nl_str][mem_r_str][measure] = [list of values]')
    print('')
    print('')
    print('It expects to find the file {}'.format(outbound_path_filename))
    print('that contains the outbound path that was used for generating the')
    print('simulated homing paths in the csv files.')
    print('')
    print('E.g.')
    print('    python analyse_trajectories_publication_script.py ./data/Converted_to_CSV/Conditions/Memory/ path-integration-forget/data/path_analysis_calculation_results_3parameters.npz')

# Get the data path
if len(sys.argv) > 2:
    # The second argument must be the path to the source csv files
    data_path = sys.argv[1]
    # The third argument must be the path to the source csv files
    output_results_filename_npz = sys.argv[2]
elif len(sys.argv) > 1 and (sys.argv[1] == '-h' or sys.argv[1] == '--help'):
    usage()
    exit(0)
else:
    print('ERROR: Expected two arguments, the directory to the csv data files and the output file.')
    print('')
    usage()
    exit(1)

# Check if the directory path exists
if not os.path.isdir(data_path):
    print('ERROR: Directory not found. Expected to find a directory with the data files:')
    print('       {}'.format(data_path))
    exit(1)

# Check if the output file already exists
if os.path.isfile(output_results_filename_npz):
    print('ERROR: The output file already exists: ')
    print('       {}'.format(output_results_filename_npz))
    exit(1)
    
# The dict results_dict that contains the calculated path statistics will be stored in this file:
# The data in this file will have the structure dict[wait_noise_sd_str][mem_Nl_str][mem_r_str][measure] = [list of values]
#Path('path-integration-forget/data/').mkdir(parents=True, exist_ok=True)
#output_results_filename_npz = 'path-integration-forget/data/path_analysis_calculation_results_3parameters.npz'
# If we were given a path to the file create the directory structure if not existant
dir_name = os.path.dirname(output_results_filename_npz)
if dir_name != '':
    Path(dir_name).mkdir(parents=True, exist_ok=True)


# Set global variables
save_figures = False # True
save_figures_as_filetype = '.pdf'
images_path = 'images/'
if save_figures:
    Path(images_path).mkdir(parents=True, exist_ok=True)

# Dictionary to store the calculated statistics
results_dict = {}
# The structure in the dictionary is
# results_dict['wait_hours'] = [0, 1, 24, 48, 96, 144, 192, 240, 288, 336, 384, 432]
# results_dict['FV'] = []
# results_dict['ZV'] = []
# results_dict[wait_noise_sd][mem_Nl][mem_r] = []

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
    h = np.array(np.hypot(x-x[0], y-y[0]))
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
    
    return (path_start, path_end, outbound_distance, outbound_angle[0], homing_ref_angle[0])


# Plots the full trajectories of simulated agents for all conditions and calculates statistics
def calc_stats_plot_trajectories(
    path,                         # Path where the input files are located
    outbound_path_filename,       # Path to the file that was used in the simulations as the outbound path
    first_file_num_id=1001,       # ID number in the filename of the first path file
    num_of_files=100,             # Number of path files/paths. The file name IDs are first_file_num_id+i in {0, num_of_files}
    show_labels = True,           # Show labels on plots
    show_axis = False,            # Show axes on plots
    distance_scaling_factor = 3,  # Path coordinates must be scaled by this factor to be in meters
    conditions = [],              # Experimental conditions list
    conditions_labels = [],       # Labels to be used to refer to the experimental conditions
    filename_extra=None,          # String to append to the saved plot filename
    get_center_of_path_beyond=0,  # Distance from the release point to ignore in the center calculation
    show_plots=True,              # Plot paths
    plot_centers_of_paths = True, # Plot the center of the trajectories
    plot_end_of_straight_paths = False): # Plot the point of first significant turn
    
    # Plots the full trajectories of simulated agents for all conditions and calculates statistics
    
    # Dictionary to store stats results
    search_dispersion_list = {"Wait": [], "Dispersion": [], "Distance_Median": [], "Distance_Median_tocntr": [], "Distance_Median_toturn": [], "Distance_Dispersion": [], "Distance_Dispersion_tocntr": [], "Distance_Dispersion_toturn": [], "Exit_Angle_Median": [], "Exit_Angle_Median_Dev": [], "Angle_Dispersion": [], "Angle_Dispersion_tocntr": [], "Angle_Dispersion_toturn": [], "Distance_hypot": [], "Distance_hypot_tocntr": [], "Distance_hypot_toturn": [], "Mean_vector_v": [], "Mean_vector_v_proj": [], "Distance_to_Nest_Dispersion": [], "Distance_to_Nest_Dispersion_tocntr": [], "Distance_to_Nest_Dispersion_toturn": []}
  
    if show_plots:
        #fig, axs = plt.subplots(1, len(conditions), sharex=True, sharey=True, figsize=(column_full/4*len(conditions), column_full/5))
        fig, axs = plt.subplots(1, len(conditions), sharex=True, sharey=True, figsize=(column_full/8*len(conditions), column_full/5))
    
    # Get the outbound path beginning and end points, distance, direction and reverse direction
    path_start, path_end, outbound_distance, outbound_angle, homing_ref_direction = get_outbound_path_details(outbound_path_filename)
        
    # Plot the trajectories of simulated agent
    # ----------------------------------------
    print('{:8} {:13} {:16} {:13} {:20} {:16} {:13} {:18}'.format('Wait (h)', '2D dispersion', 'Hom. dist median', 'Hom. dist std', 'Hom. exit ang median', 'Hom. heading std', 'Mean vect len', 'Mean vect len proj'))
    # Plot the simulated trajectories
    for idx_i,condition in enumerate(conditions):
        if show_plots:
            if isinstance(axs, np.ndarray):
                ax = axs[idx_i]
            else:
                ax = axs
            axs_row = axs
    
        start = first_file_num_id
        num = num_of_files
        noise_syn=0.1
        noise_rot = 2.0
        noiseSlope = 9.0
        sampling_rate = 25
        plot_npz_or_csv = 'csv' # 'npz' or 'csv' which files to use original or preprocessed
        condition_sim = condition
        if condition_sim == 'FVNoIce':
            condition_sim = 'FV'
        if condition_sim == 'ZVNoIce':
            condition_sim = 'FVIce=0.5'
            condition_sim = 'ZV'
        if condition_sim == 'FVIce':
            condition_sim = 'FVIcex0.85'
        
        # For storing the center of each search
        path_ends_list = {}
        path_ends_list['x'] = []
        path_ends_list['y'] = []
        center_of_search = {}
        center_of_search['x'] = []
        center_of_search['y'] = []
        mean_vector_v = {}
        mean_vector_v['length'] = []
        mean_vector_v['theta'] = []
        mean_vector_v['length_exit'] = []
        mean_vector_v['theta_exit'] = []

        for i in range(start, start+num):
            filename = path + condition_sim + '/with_Pontin_Holonomic_noiseSyn' + str(noise_syn) + '_noiseRot' + str(noise_rot) + '_noiseSlope' + str(noiseSlope) + '_route_' + condition_sim + '_' + str(i) + '.npz'
        
            # If we use the already processed files (data/Converted_to_CSV/)
            if plot_npz_or_csv == 'csv':
                filename_csv = filename.replace('.npz', '.csv')            
                pd_i = pd.read_csv(filename_csv)

            # Get the path coordinates and make the end of the outbound trip the origin
            x = pd_i["x"] + path_end[0]
            y = pd_i["y"] + path_end[1]
            x = x * distance_scaling_factor
            y = y * distance_scaling_factor
            if show_plots:
                ax.plot(x, y, color='#444444', alpha=0.3)
            
            # We cannot reliably detect the beginning of the search phase for all conditions so we calculate both
            # the point of first significant turn and the center of the path. 
            # Get the start of the search
            x_end, y_end, first_turn_index, first_turn_indeces_lst = get_start_of_search(x, y) # First turn point
            # Get the center of the search (considers the whole path but most of it is the search)
            x_centre, y_centre = get_center_of_path(x, y, beyond=get_center_of_path_beyond) # Center of path
            # Get the point where the agent first exits the circle with radius around the release point
            #first_cross_index = get_first_cross_radius(x, y, radius=5)
            
            # Some agents make a U-turn in the beginning of the path to turn towards the nest, 
            # if the detected first turn is too near the release point use go through the subsequent turning points
            # until finding the one that is further than 0.25 of the start to end path straight line distance
            #if first_turn_index is not None and len(first_turn_indeces_lst) > 1: # Have we found turning points?
            #    if np.hypot(x[0]-x[first_turn_index], y[0]-y[first_turn_index]) < 0.25*np.hypot(x[0]-x.iloc[-1], y[0]-y.iloc[-1]): # Too near to the release point is relative to start-end distance to avoid messing up with ZV paths
            #        x_end, y_end, first_turn_index = (x[first_turn_indeces_lst[1]], y[first_turn_indeces_lst[1]], first_turn_indeces_lst[1])        
            if first_turn_index is not None and len(first_turn_indeces_lst) > 1: # Have we found turning points?
                j = 0
                while (j<len(first_turn_indeces_lst)) and (np.hypot(x[0]-x[first_turn_indeces_lst[j]], y[0]-y[first_turn_indeces_lst[j]]) < 0.25*np.hypot(x[0]-x.iloc[-1], y[0]-y.iloc[-1])):
                    j += 1
                x_end, y_end, first_turn_index = (x[first_turn_indeces_lst[j]], y[first_turn_indeces_lst[j]], first_turn_indeces_lst[j])           
            # Later we use this variable, if no turning points were found point to the end of the path
            if first_turn_index is None:
                first_turn_index = -1
            
            path_ends_list['x'].append(x_end)
            path_ends_list['y'].append(y_end)
            center_of_search['x'].append(x_centre)
            center_of_search['y'].append(y_centre)
            
            # Get the mean vector length
            #homing_heading_v, homing_heading_theta = circ_mean_vector_v(x[:first_turn_index], y[:first_turn_index], rediscretization_step=0.1, ref_dir=homing_ref_direction)
            # Alternative
            radius = 4
            first_cross_index = get_first_cross_radius(x, y, radius=radius)
            homing_heading_v, homing_heading_theta = circ_mean_vector_v(x[:first_cross_index], y[:first_cross_index], rediscretization_step=0.1, ref_dir=homing_ref_direction)
            mean_vector_v['length'].append(homing_heading_v)
            mean_vector_v['theta'].append(homing_heading_theta)
            # Calculate the exit angle
            exit_angle = np.arctan2(y[first_cross_index]-y[0], x[first_cross_index]-x[0])
            mean_vector_v['length_exit'].append(radius)
            mean_vector_v['theta_exit'].append(exit_angle)

        # Get the mean vector of all mean vectors
        median_x = np.median(np.array(mean_vector_v['length']) * np.cos(np.array(mean_vector_v['theta'])))
        median_y = np.median(np.array(mean_vector_v['length']) * np.sin(np.array(mean_vector_v['theta'])))
        total_mean_homing_vec_length = np.hypot(median_x, median_y)
    
        # Get the mean vector of all mean exit vectors
        median_exit_x = np.median(np.array(mean_vector_v['length_exit']) * np.cos(np.array(mean_vector_v['theta_exit'])))
        median_exit_y = np.median(np.array(mean_vector_v['length_exit']) * np.sin(np.array(mean_vector_v['theta_exit'])))
        total_mean_homing_vec_exit_length = np.hypot(median_x, median_y)
        #total_mean_homing_vec_exit_angle = np.arctan2(median_exit_y, median_exit_x)
        total_mean_homing_vec_exit_angle = scipy.stats.circmean(mean_vector_v['theta_exit'])
        # Convert to numpy array to simplify computation in the next steps and keep them as they were in previous version
        total_mean_homing_vec_exit_angle = np.array([total_mean_homing_vec_exit_angle])
        # Projection on the homing direction
        total_mean_homing_vec_exit_angle = total_mean_homing_vec_exit_angle - homing_ref_direction
        # Normalise to 0-2*pi
        total_mean_homing_vec_exit_angle = total_mean_homing_vec_exit_angle % (2*np.pi)

        # Calculate the deviation of the mean exit angle from the homing_ref_direction
        total_mean_homing_vec_exit_angle_dev = total_mean_homing_vec_exit_angle.copy()
        # Normalise to -pi<angle<=pi
        total_mean_homing_vec_exit_angle_dev[total_mean_homing_vec_exit_angle_dev<=-np.pi] += 2*np.pi
        total_mean_homing_vec_exit_angle_dev[total_mean_homing_vec_exit_angle_dev>np.pi]   -= 2*np.pi
        #total_mean_homing_vec_exit_angle_dev = np.abs(total_mean_homing_vec_exit_angle_dev)
        # Keep only the element from the array with one element
        total_mean_homing_vec_exit_angle_dev = total_mean_homing_vec_exit_angle_dev[0]
        
        # Keep only the element from the array with one element
        total_mean_homing_vec_exit_angle = total_mean_homing_vec_exit_angle[0]
        # Project the vector to the release-nest direction and get the length
        total_mean_homing_vec_projection_length_v = total_mean_homing_vec_exit_length * np.cos(total_mean_homing_vec_exit_angle)
        
        # Dispersion of the search center distance across trials (dispersion)
        homing_distance_median_tocntr, homing_distance_std_tocntr, homing_distance_h_tocntr = homing_distance_spread(np.array(center_of_search['x'])-path_end[0]*distance_scaling_factor, np.array(center_of_search['y'])-path_end[1]*distance_scaling_factor)
        # Dispersion of the first turn distance across trials (dispersion)
        homing_distance_median_toturn, homing_distance_std_toturn, homing_distance_h_toturn = homing_distance_spread(np.array(path_ends_list['x'])-path_end[0]*distance_scaling_factor, np.array(path_ends_list['y'])-path_end[1]*distance_scaling_factor)
    
        # Dispersion of the distance between the search center and the nest location across trials (dispersion)
        homing_distance_error_MAD_tocntr = np.median(np.abs(np.hypot(np.array(center_of_search['x'])-path_start[0]*distance_scaling_factor, np.array(center_of_search['y'])-path_start[1]*distance_scaling_factor)))
        # Dispersion of the distance between the first turn and the nest location across trials (dispersion)
        homing_distance_error_MAD_toturn = np.median(np.abs(np.hypot(np.array(path_ends_list['x'])-path_start[0]*distance_scaling_factor, np.array(path_ends_list['y'])-path_start[1]*distance_scaling_factor)))
    
        # Dispersion of the search center angle across trials (dispersion)
        homing_heading_median_tocntr, homing_heading_std_tocntr = homing_heading_spread(np.array(center_of_search['x'])-path_end[0]*distance_scaling_factor, np.array(center_of_search['y'])-path_end[1]*distance_scaling_factor)
        # Dispersion of the first turn angle across trials (dispersion)
        homing_heading_median_toturn, homing_heading_std_toturn = homing_heading_spread(np.array(path_ends_list['x'])-path_end[0]*distance_scaling_factor, np.array(path_ends_list['y'])-path_end[1]*distance_scaling_factor)
    
        # How precise is the search center across trials (dispersion)
        search_dispersion = spread_of_2D_points(center_of_search['x'], center_of_search['y'])
    
        # Store results
        print('{:^8} {:^13.6f} {:^8.3f}/{:^8.3f} {:^6.2f}/{:^6.2f} {:^8.2f} ({:^8.2f}) {:^8.1f}/{:^8.1f} {:^13.3f} {:^18.3f}'.format(conditions_labels[idx_i], search_dispersion, homing_distance_median_tocntr, homing_distance_median_toturn, homing_distance_std_tocntr, homing_distance_std_toturn, np.degrees(total_mean_homing_vec_exit_angle), np.degrees(total_mean_homing_vec_exit_angle_dev), np.degrees(homing_heading_std_tocntr), np.degrees(homing_heading_std_toturn), total_mean_homing_vec_length, total_mean_homing_vec_projection_length_v))
        search_dispersion_list["Wait"].append(conditions_labels[idx_i])
        search_dispersion_list["Dispersion"].append(search_dispersion)
        search_dispersion_list["Distance_Median"].append(homing_distance_median_tocntr)
        search_dispersion_list["Distance_Median_tocntr"].append(homing_distance_median_tocntr)
        search_dispersion_list["Distance_Median_toturn"].append(homing_distance_median_toturn)
        search_dispersion_list["Distance_Dispersion"].append(homing_distance_std_tocntr)
        search_dispersion_list["Distance_Dispersion_tocntr"].append(homing_distance_std_tocntr)
        search_dispersion_list["Distance_Dispersion_toturn"].append(homing_distance_std_toturn)
        search_dispersion_list["Exit_Angle_Median"].append(total_mean_homing_vec_exit_angle)
        search_dispersion_list["Exit_Angle_Median_Dev"].append(total_mean_homing_vec_exit_angle_dev)
        search_dispersion_list["Angle_Dispersion"].append(homing_heading_std_tocntr)
        search_dispersion_list["Angle_Dispersion_tocntr"].append(homing_heading_std_tocntr)
        search_dispersion_list["Angle_Dispersion_toturn"].append(homing_heading_std_toturn)
        search_dispersion_list["Distance_hypot"].append(homing_distance_h_tocntr)
        search_dispersion_list["Distance_hypot_tocntr"].append(homing_distance_h_tocntr)
        search_dispersion_list["Distance_hypot_toturn"].append(homing_distance_h_toturn)
        search_dispersion_list["Mean_vector_v"].append(total_mean_homing_vec_length)
        search_dispersion_list["Mean_vector_v_proj"].append(total_mean_homing_vec_projection_length_v)
        search_dispersion_list["Distance_to_Nest_Dispersion"].append(homing_distance_error_MAD_tocntr)
        search_dispersion_list["Distance_to_Nest_Dispersion_tocntr"].append(homing_distance_error_MAD_tocntr)
        search_dispersion_list["Distance_to_Nest_Dispersion_toturn"].append(homing_distance_error_MAD_toturn)

        if show_plots:
            ax.axis('scaled')
            draw_sizebar(ax, size=5.0, label='5m', location='lower center')
        
        if show_plots:
            # Plot the nest location
            nest_x, nest_y = path_start[0]*distance_scaling_factor, path_start[1]*distance_scaling_factor
            ax.plot(nest_x, nest_y, '.', color='darkorange')
            # Plot the release location
            ax.plot(path_end[0]*distance_scaling_factor, path_end[1]*distance_scaling_factor, '+', color='darkorange')

            if plot_centers_of_paths:
                ax.scatter(center_of_search['x'],center_of_search['y'], marker='x', color='darkorange', zorder=10000)    
            if plot_end_of_straight_paths:
                ax.scatter(path_ends_list['x'],path_ends_list['y'], marker='+', color='b', zorder=10000)    
    
            # Beautify the axes
            if show_axis:
                adjust_spines(ax, ['left', 'bottom'], x_values=np.array([-5, 5]), y_values=np.array([-5, 10]), margin=1)
            else:
                ax.set_axis_off()

            if show_labels:
                ax.set_xlabel("x (m)")
                if idx_i == 0:
                    ax.set_ylabel("y (m)")

            if show_labels:
                ax.set_title(str(conditions_labels[idx_i])+' hours')

            ax.set_aspect('equal')
            ax.grid(color="0.9", linestyle='-', linewidth=1)
    
    if show_plots:
        if filename_extra is not None:
            filename_extra = '_' + filename_extra
        else:
            filename_extra = ''

        if save_figures:
            fig.savefig(images_path + 'simulated_fullRelease_' + 'routes_vs_waiting_time' + filename_extra + save_figures_as_filetype, bbox_inches='tight', transparent=True) # , pad_inches=0
    
    if show_plots:
        return (search_dispersion_list, outbound_distance, fig, axs)
    else:
        return (search_dispersion_list, outbound_distance)


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



np.set_printoptions(threshold=np.inf)

get_center_of_path_beyond = 0


"""
The Ramer-Douglas-Peucker algorithm simplifies a polyline 
(a curve made of multiple linear segments) 
by replacing multiple segments with fewer. 
Copied from https://github.com/sebleier/RDP/
roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
and used by https://stackoverflow.com/questions/14631776/calculate-turning-points-pivot-points-in-trajectory-path
"""
from math import sqrt


def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) -
            (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d


def rdp(points, epsilon):
    """Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results


def angle(dir):
    """
    Returns the angles between vectors.

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]
    return np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

def find_turning_points(x, y, tolerance=70, min_angle = np.pi*0.3, plot=False):
    """ Detects turning points when the direction is more than min_angle away from the release-nest direction.
    x, y      : 1D lists or arrays with the paired x and y coordinates of points of the trajectory.
    tolerance : the maximum distance the simplified path can stray from the original path.
    min_angle : the minimum angle away from the homing direction considered a turning point.
    Returns   : the x,y cordinates of the points of turning more than min_angle as two lists.
    """
    
    # Get the outbound path beginning and end points
    path_start, path_end = get_start_end_of_path(outbound_path_filename)
    outbound_distance = np.hypot(path_end[0]-path_start[0], path_end[1]-path_start[1])*distance_scaling_factor

    # The outbound direction calculated end-to-end
    outbound_traj_vector = pd.DataFrame({'x': [0, path_end[0]-path_start[0]], 'y': [0, path_end[1]-path_start[1]]})
    outbound_angle = calc_angle(outbound_traj_vector, ref_dir=0)
    # The corresponding inbound direction (release point to fictive nest) is outbound_traj_vector+180deg
    homing_ref_vector = (outbound_angle + np.pi) % (2*np.pi)
    
    # Convert the two 1D lists of length N into a 2D Nx2 list.
    points = np.array(list(zip(x, y)))

    # Use the Ramer-Douglas-Peucker algorithm to simplify the path
    # http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
    # Python implementation: https://github.com/sebleier/RDP/
    simplified = np.array(rdp(points.tolist(), tolerance))
    
    sx, sy = simplified.T
    
    # compute the direction vectors on the simplified curve in respect to the home direction
    directions = pd.DataFrame({'x': simplified[:,0], 'y': simplified[:,1]})
    #print('Directions =', np.degrees(np.arctan2(directions['y'], directions['x'])))
    theta_diff = calc_angle(directions, ref_dir=homing_ref_vector)
    
    # Select the index of the heading that points at least min_angle away from the homing direction
    idx = np.where(np.abs(theta_diff)>np.abs(min_angle))[0]
    
    if plot:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        
        # Plot the start to fictious nest line
        ax.plot(x[0], y[0], '+', color='darkorange')   # Plot the start of the path
        ax.plot(path_start[0], path_start[1], 'o', color='darkorange') # Plot the fictious nest
        ax.plot([x[0], path_start[0]], [y[0], path_start[1]], '--', color='darkorange', alpha=0.5) # Plot line from release point to fictious nest
        
        # Plot the path
        #ax.plot(x, y, '-', color='black', alpha=0.3, label='original path')
        # Plot the path with varied alpha from 0 to 1
        for i in range(0, x.shape[0]-1, 100):
            ax.plot(x[i:i+100], y[i:i+100], '-', color='black', alpha=1-float(i)/(x.shape[0]+1))
        # Plot the simplified line path
        #ax.plot(sx, sy, '--', color='green', alpha=0.3, label='simplified path')
        # Plot the simplified line path with varied alpha from 0 to 1
        for i in range(0, sx.shape[0]-1):
            ax.plot(sx[i:i+2], sy[i:i+2], '--', color='black', alpha=1-float(i)/(sx.shape[0]+1))
        # Plot the detected turning points
        ax.plot(sx[idx], sy[idx], 'rx', label='turning points')
        # Plot the last detected turning point
        ax.plot(sx[-1], sy[-1], 'o', color='darkorange', label='turning points')
        #ax.invert_yaxis()
        #plt.legend(loc='best')
        #plt.show()
    
    return (sx[idx], sy[idx])


def get_start_of_search(x, y):
    """ Attempts to estimate the x,y coordinates where the ant starts searching for its nest.
        First attempts to find the point of first big turn and if it cannot estimates the center
        of the search pattern. """
    # Try to estimate the start of the search by finding the first turning point of the ant
    # A big turn is considered as a turn at least 90degrees
    #tx, ty = find_turning_points(x, y, tolerance=0.5, min_angle = np.pi*0.75, plot=False)
    tx, ty = find_turning_points(x, y, tolerance=0.5, min_angle = np.radians(91), plot=False)
    if len(tx) > 0:
        nearest_point = np.hypot(x - tx[0], y - ty[0])
        turning_point_idx = np.argmin(np.array(nearest_point))
        #return (x[turning_point_idx], y[turning_point_idx], turning_point_idx)
        
        turning_point_idx_lst = []
        for i in range(len(tx)):
            nearest_point = np.hypot(x - tx[i], y - ty[i])
            turning_point_idx_lst.append(np.argmin(np.array(nearest_point)))
        return (x[turning_point_idx], y[turning_point_idx], turning_point_idx, turning_point_idx_lst)
    else:
        # If the first turn was not found get the center of the search as a suboptimal estimate of the start of search
        x_med, y_med = get_center_of_path(x, y, beyond=2)
        return (x_med, y_med, None, None)



def get_first_cross_radius(x, y, radius=3):
    first_cross_index = np.argmax(np.abs(np.hypot(x - x[0], y - y[0]) - radius).to_numpy() < 0.05)
    # If no crossing of the circle at radius was foudn return the index of the last item
    if first_cross_index == 0: first_cross_index = len(x)-1
    return first_cross_index


# Main program

# Plot the full trajectories of simulated agents for all conditions
show_labels = True
show_axis = False
distance_scaling_factor = 3
#distance_scaling_factor = 2.87 # Corrected scaling factor that results in median homing distance of 12.79m for 0h of captivity time as reported in Ziegler1997. With factor x3 we get mean homing distance of 13.349528374417343, so the correct factor should be 3*12.7911162/13.349528374417343=2.874509684816851



# Conditions
conditions = ['FV']
conditions_labels = ['FV']

filename_extra=''

results_dict['FV'], outbound_distance = calc_stats_plot_trajectories(data_path, outbound_path_filename, num_of_files=40, show_labels=show_labels, show_axis=show_axis, distance_scaling_factor=distance_scaling_factor, conditions=conditions, conditions_labels=conditions_labels, show_plots=False)



# Plot the full trajectories of simulated agents for all conditions

# Conditions
conditions = ['ZV']
conditions_labels = ['ZV']

filename_extra=''

results_dict['ZV'], outbound_distance = calc_stats_plot_trajectories(data_path, outbound_path_filename, num_of_files=40, show_labels=show_labels, show_axis=show_axis, distance_scaling_factor=distance_scaling_factor, conditions=conditions, conditions_labels=conditions_labels, show_plots=False)



# Plot the full trajectories of simulated agents for all conditions

wait_hours = [0, 1, 24, 48, 96, 144, 192, 240, 288, 336, 384, 432] # Wait hours
wait_noise_sd_list = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.009, 0.01, 0.015, 0.02]

# First data collection: collected using wrong Nl range: data in 3_params_scanning_wrong_Nl_range/
mem_Nls = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
mem_rs = [-0.008, -0.010, -0.012, -0.014, -0.016, -0.018, -0.020, -0.022, -0.024, -0.026, -0.028, -0.030, -0.032]

# New data collection: collected using the correct Nl range: data in 3_params_scanning_correct_Nl_range/
mem_Nls = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
mem_rs = [-0.008, -0.010, -0.012, -0.014, -0.015, -0.016, -0.017, -0.018, -0.019, -0.020, -0.021, -0.022, -0.023, -0.024, -0.025, -0.026, -0.027, -0.028, -0.029, -0.030, -0.031, -0.032]

# Newest data collection: combining the last and additional collected data: data in 3_params_scanning_correct_Nl_range/
mem_Nls =  [0.0, 0.001, 0.01, 0.016, 0.018, 0.019, 0.02, 0.021, 0.022, 0.023, 0.024, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.5, 0.999, 1.3] # 25
mem_rs = [-0.008, -0.01, -0.012, -0.014, -0.015, -0.016, -0.017, -0.018, -0.019, -0.02, -0.021, -0.022, -0.023, -0.024, -0.025, -0.026, -0.027, -0.028, -0.029, -0.03, -0.031, -0.032, -0.035, -0.04, -0.045, -0.05, -0.1] # 27

# Extra data collection: explore additional region of the search space
#mem_Nls =  [0.0, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
#mem_rs =   [-0.033, -0.034, -0.035, -0.036, -0.037, -0.038, -0.039, -0.04, -0.041, -0.042]


# Add an entry with the waiting durations that correspond to the data
results_dict['wait_hours'] = wait_hours

# Get the outbound path beginning and end points, distance, direction and reverse direction
path_start, path_end, outbound_distance, outbound_angle, homing_ref_direction = get_outbound_path_details(outbound_path_filename)
# Store outbound path details
results_dict['outbound_distance'] = outbound_distance
results_dict['outbound_angle'] = outbound_angle
results_dict['homing_ref_direction'] = homing_ref_direction

for wait_noise_sd in wait_noise_sd_list:
    wait_noise_sd_str = str(wait_noise_sd)
    results_dict[wait_noise_sd_str] = {}
    for mem_Nl in mem_Nls:
        mem_Nl_str = str(mem_Nl)
        results_dict[wait_noise_sd_str][mem_Nl_str] = {}
        for mem_r in mem_rs:
            mem_r_str = str(mem_r)
            results_dict[wait_noise_sd_str][mem_Nl_str][mem_r_str] = []
            conditions_labels = wait_hours # Wait hours
            conditions = []
            # Create a list of patterns to match the relevant files
            for mem_wait in wait_hours:
                conditions.append('FVWait'+str(float(mem_wait))+'h'+'Noise'+wait_noise_sd_str+'Nl'+mem_Nl_str+'r'+mem_r_str)
            
            print('Processing for wait_noise_sd = ' + wait_noise_sd_str + ' mem_Nl = ' + mem_Nl_str + ' mem_r = ' + mem_r_str)
            #results_dict['FVWaitNoise' + wait_noise_sd_str + 'b'], outbound_distance, fig, axs = calc_stats_plot_trajectories(path, outbound_path_filename, show_labels, show_axis, distance_scaling_factor, conditions, conditions_labels, filename_extra, show_plots=False)
            # Do not bail out if files do not exist for a condition
            try:
                results_dict[wait_noise_sd_str][mem_Nl_str][mem_r_str], outbound_distance = calc_stats_plot_trajectories(data_path, outbound_path_filename, num_of_files=40, show_labels=show_labels, show_axis=show_axis, distance_scaling_factor=distance_scaling_factor, conditions=conditions, conditions_labels=conditions_labels, show_plots=False)
            except:
                print('ERROR: Files not found for wait_noise_sd = ' + wait_noise_sd_str + ' mem_Nl = ' + mem_Nl_str + ' mem_r = ' + mem_r_str)
                # Delete the key entry
                del results_dict[wait_noise_sd_str][mem_Nl_str][mem_r_str]

# Store data to file
# Store the results_dict dict which contains the calculated path statistics
# The data in this file will have the structure dict[wait_noise_sd_str][mem_Nl_str][mem_r_str][measure] = [list of values]
np.savez(output_results_filename_npz, results_dict)
